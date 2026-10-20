import os
import time
import uuid
import secrets
import tempfile
import logging
import threading
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import numpy as np
import requests

from deepface import DeepFace
from fastapi import FastAPI, Header, HTTPException, File, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from qdrant_client import QdrantClient, models

# Lets Pillow decode HEIC/HEIF (iPhone photos). Requires `pillow-heif`
# in requirements.txt.
register_heif_opener()


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diwalihaat-face-api")

TRIGGER_SECRET = os.environ["PYTHON_TRIGGER_SECRET"]
VERIFY_API_TOKEN = os.environ["VERIFY_API_TOKEN"]

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

# Optional but recommended: restricts downloads to YOUR Cloudinary account.
CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", "").strip()

COLLECTION_NAME = "student_face_embeddings"
VECTOR_SIZE = 512

MAX_IMAGE_BYTES = 10 * 1024 * 1024      # raw file size limit
# Raised from 12M: a standard 12 MP iPhone photo (4032x3024 = 12.19M px)
# was being rejected. 50 MP Android photos are still rejected; downscale
# in the app before upload if you need to support them.
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_SIDE = 1024                   # shrink before face detection/model inference

# Extra room for multipart boundaries/headers on top of the image itself.
MAX_UPLOAD_REQUEST_BYTES = MAX_IMAGE_BYTES + 1024 * 1024

# Total wall-clock budget for a Cloudinary download (requests' timeout is
# per-read only, so a slow-drip server could otherwise hold a thread).
DOWNLOAD_DEADLINE_SECONDS = 30

# How long enrollment (MongoDB trigger) waits for the model before giving up.
# Verification stays fail-fast (0) so the app can retry quickly.
ENROLL_LOCK_WAIT_SECONDS = 30

# DeepFace model inference is memory-intensive. Keep only one inference active
# per process; use one Uvicorn worker on small Railway instances.
EMBEDDING_LOCK = threading.Lock()

ALLOWED_IMAGE_HOST = "res.cloudinary.com"

# opencv is the weakest detector (false multi-face hits). If good photos get
# "exactly one face" errors, try "retinaface" or "yunet".
DETECTOR_BACKEND = os.environ.get("DETECTOR_BACKEND", "opencv")

# Matching settings. Tune these on your own data.
MATCH_TOP_K = 5
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.40"))
MIN_MATCH_MARGIN = float(os.environ.get("MIN_MATCH_MARGIN", "0.05"))

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30,
)


# ============================================================
# REQUEST MODELS
# ============================================================

class StudentPhotoJob(BaseModel):
    # Accept the existing fields plus any additional student fields sent
    # by the MongoDB trigger. Arbitrary student attributes can also be
    # grouped inside the optional `student_data` object.
    model_config = ConfigDict(extra="allow")

    student_id: str = Field(pattern=r"^[0-9a-fA-F]{24}$")
    image_url: HttpUrl
    student_data: dict = Field(default_factory=dict)


def build_student_payload(job: StudentPhotoJob, student_id: str) -> dict:
    """Build a Qdrant-compatible JSON payload containing all supplied data."""
    request_data = job.model_dump(mode="json")
    student_data = request_data.pop("student_data", {}) or {}

    payload = {
        **request_data,
        "student_id": student_id,
        "embedding_model": "ArcFace",
        "student_data": student_data,
    }
    return payload


# ============================================================
# AUTHENTICATION
# ============================================================

def authenticate(
    authorization: str | None,
    expected_token: str,
) -> None:
    expected = f"Bearer {expected_token}"

    # Compare bytes so non-ASCII headers cannot raise TypeError (500).
    if not authorization or not secrets.compare_digest(
        authorization.encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )


# ============================================================
# QDRANT POINT IDS
# Enrollment and verification must use the same ID scheme.
# ============================================================

def student_point_id(student_id: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"diwalihaat:{student_id.lower()}",
        )
    )


# ============================================================
# QDRANT COLLECTION
# ============================================================

def ensure_collection() -> None:
    if not qdrant.collection_exists(COLLECTION_NAME):
        try:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection.")
            return
        except Exception:
            # Another worker may have created it first.
            if not qdrant.collection_exists(COLLECTION_NAME):
                raise

    collection = qdrant.get_collection(COLLECTION_NAME)
    config = collection.config.params.vectors

    if not isinstance(config, models.VectorParams):
        raise RuntimeError(
            "Expected a single dense Qdrant vector."
        )

    if config.size != VECTOR_SIZE:
        raise RuntimeError(
            f"Qdrant vector size is {config.size}; "
            f"expected {VECTOR_SIZE}."
        )

    if config.distance != models.Distance.COSINE:
        raise RuntimeError(
            "Qdrant collection must use cosine distance."
        )


# ============================================================
# TEMP FILE HELPER
# ============================================================

def remove_quietly(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning("Could not remove temporary file.")


# ============================================================
# CLOUDINARY URL VALIDATION
# Enrollment only: verification accepts a camera upload.
# ============================================================

def validate_cloudinary_url(image_url: str) -> str:
    try:
        parsed = urlparse(image_url)
        port = parsed.port
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image URL.",
        )

    if (
        parsed.scheme != "https"
        or parsed.hostname != ALLOWED_IMAGE_HOST
        or port is not None
        or parsed.username
        or parsed.password
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
        or "/image/upload/" not in parsed.path
    ):
        raise HTTPException(
            status_code=400,
            detail="A valid HTTPS Cloudinary image URL is required.",
        )

    if CLOUDINARY_CLOUD_NAME and not parsed.path.startswith(
        f"/{CLOUDINARY_CLOUD_NAME}/image/upload/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Image does not belong to the expected Cloudinary account.",
        )

    return image_url


# ============================================================
# DOWNLOAD CLOUDINARY IMAGE SAFELY
# The format is NOT trusted from headers; Pillow validates the
# real file contents later in normalize_image().
# ============================================================

# Some HEIC files are served as octet-stream. Pillow does the real
# validation, so this check is deliberately loose.
_ALLOWED_CONTENT_TYPES = {"application/octet-stream", "binary/octet-stream"}


def download_image(image_url: str) -> str:
    image_url = validate_cloudinary_url(image_url)
    temp_path = None
    succeeded = False
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_SECONDS

    try:
        with requests.get(
            image_url,
            timeout=(5, 20),
            stream=True,
            allow_redirects=False,
        ) as response:

            if response.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail="Image provider could not return the image.",
                )

            content_type = (
                response.headers.get("Content-Type", "")
                .split(";")[0]
                .strip()
                .lower()
            )

            if not (
                content_type.startswith("image/")
                or content_type in _ALLOWED_CONTENT_TYPES
            ):
                raise HTTPException(
                    status_code=415,
                    detail="The URL did not return an image.",
                )

            content_length = response.headers.get("Content-Length")

            if content_length:
                try:
                    declared = int(content_length)
                except ValueError:
                    declared = None

                if declared is not None and declared > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Image exceeds the 10 MB limit.",
                    )

            total_bytes = 0

            with tempfile.NamedTemporaryFile(
                suffix=".img",
                delete=False,
            ) as temp:
                temp_path = temp.name

                for chunk in response.iter_content(8192):
                    if time.monotonic() > deadline:
                        raise HTTPException(
                            status_code=504,
                            detail="Image download timed out.",
                        )

                    if not chunk:
                        continue

                    total_bytes += len(chunk)

                    if total_bytes > MAX_IMAGE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Image exceeds the 10 MB limit.",
                        )

                    temp.write(chunk)

            if total_bytes == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Downloaded image is empty.",
                )

        succeeded = True
        return temp_path

    except HTTPException:
        raise

    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Image download timed out.",
        )

    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve the Cloudinary image.",
        )

    finally:
        if not succeeded:
            remove_quietly(temp_path)


# ============================================================
# NORMALIZE ANY SUPPORTED IMAGE TO A SMALL RGB JPEG
# Handles JPEG, PNG, WebP, HEIC/HEIF, and anything else
# Pillow can decode. Also applies EXIF rotation, flattens
# transparency, enforces a pixel limit, and downsizes.
# ============================================================

def normalize_image(image_path: str) -> str:
    output_path = None

    try:
        with Image.open(image_path) as opened:
            width, height = opened.size

            # Checked from the header, before pixels are decoded.
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large.",
                )

            # thumbnail() can use decoder-side downsampling for some formats
            # (notably JPEG). HEIC decodes fully, which is why the pixel
            # limit above is kept modest.
            opened.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            image = ImageOps.exif_transpose(opened)

            if (
                image.mode in ("RGBA", "LA", "PA")
                or "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")

            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False,
            ) as output:
                output_path = output.name

            image.save(
                output_path,
                format="JPEG",
                quality=90,
            )

        return output_path

    except HTTPException:
        remove_quietly(output_path)
        raise

    except Exception as error:
        remove_quietly(output_path)
        logger.warning(
            "Image normalization failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=415,
            detail="Unsupported, corrupted, or invalid image.",
        )


# ============================================================
# GENERATE ARCFACE EMBEDDING
# ============================================================

def get_embedding(image_path: str) -> list[float]:
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name="ArcFace",
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )

        if not results:
            raise HTTPException(
                status_code=422,
                detail="No face detected.",
            )

        if len(results) != 1:
            raise HTTPException(
                status_code=422,
                detail="The image must contain exactly one detectable face.",
            )

        vector = np.asarray(
            results[0]["embedding"],
            dtype=np.float32,
        )

        if vector.shape != (VECTOR_SIZE,):
            logger.error(
                "Unexpected embedding dimensions: %s",
                vector.shape,
            )
            raise HTTPException(
                status_code=500,
                detail="Unexpected embedding dimensions.",
            )

        if not np.all(np.isfinite(vector)):
            raise HTTPException(
                status_code=422,
                detail="Invalid face embedding.",
            )

        if np.linalg.norm(vector) == 0:
            raise HTTPException(
                status_code=422,
                detail="Generated embedding is invalid.",
            )

        return vector.tolist()

    except HTTPException:
        raise

    except ValueError:
        # DeepFace raises ValueError when enforce_detection=True finds no
        # face. This is a bad photo (client problem), not a server fault.
        raise HTTPException(
            status_code=422,
            detail="No face detected.",
        )

    except Exception as error:
        # Model load failures, out-of-memory, etc. are server problems and
        # must not be reported to the client as a bad photo.
        logger.exception(
            "Embedding generation failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Face model error.",
        )


# ============================================================
# EMBEDDING FROM A RAW IMAGE FILE (shared by both paths)
# ============================================================

def embedding_from_raw_file(
    raw_path: str,
    wait_seconds: float = 0,
) -> list[float]:
    normalized_path = None

    # Serialize the full image-processing pipeline, not just model inference.
    # wait_seconds=0 fails fast (verification); enrollment waits so a busy
    # moment does not silently drop a student.
    if wait_seconds > 0:
        acquired = EMBEDDING_LOCK.acquire(timeout=wait_seconds)
    else:
        acquired = EMBEDDING_LOCK.acquire(blocking=False)

    if not acquired:
        remove_quietly(raw_path)
        raise HTTPException(
            status_code=503,
            detail="Face recognition is busy. Please retry shortly.",
            headers={"Retry-After": "2"},
        )

    try:
        normalized_path = normalize_image(raw_path)
        return get_embedding(normalized_path)
    finally:
        remove_quietly(normalized_path)
        remove_quietly(raw_path)
        EMBEDDING_LOCK.release()


# ============================================================
# EMBEDDING FROM CLOUDINARY URL
# ============================================================

def make_embedding(image_url: str) -> list[float]:
    raw_path = download_image(image_url)
    return embedding_from_raw_file(
        raw_path,
        wait_seconds=ENROLL_LOCK_WAIT_SECONDS,
    )


# ============================================================
# EMBEDDING FROM CAMERA UPLOAD
# ============================================================

def make_embedding_from_upload(
    image: UploadFile,
) -> list[float]:
    raw_path = None

    try:
        total_bytes = 0

        with tempfile.NamedTemporaryFile(
            suffix=".img",
            delete=False,
        ) as temp:
            raw_path = temp.name

            while True:
                chunk = image.file.read(8192)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Image exceeds the 10 MB limit.",
                    )

                temp.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty.",
            )

    except BaseException:
        remove_quietly(raw_path)
        image.file.close()
        raise

    image.file.close()
    return embedding_from_raw_file(raw_path)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DiwaliHaat Face API.")
    logger.info(
        "Image limits: max_bytes=%s max_pixels=%s max_side=%s",
        MAX_IMAGE_BYTES,
        MAX_IMAGE_PIXELS,
        MAX_IMAGE_SIDE,
    )
    ensure_collection()
    logger.info("Qdrant initialization complete.")

    # Warm up ArcFace so the first real request does not pay for the
    # weight download / model build (and time out). Ideally the weights
    # are also baked into the Docker image.
    try:
        DeepFace.build_model("ArcFace")
        logger.info("ArcFace model warmed up.")
    except Exception as error:
        logger.warning(
            "ArcFace warm-up failed (will retry on first request): %s",
            type(error).__name__,
        )

    try:
        yield
    finally:
        qdrant.close()
        logger.info("Qdrant client closed.")


app = FastAPI(
    title="DiwaliHaat Face Verification API",
    version="2.2.0",
    lifespan=lifespan,
)


# ============================================================
# EARLY GUARD FOR /verify-face
# Runs BEFORE the multipart body is parsed/spooled, so:
#   - unauthenticated callers get 401 without uploading anything
#   - oversized uploads are rejected from Content-Length up front
# If a client sends chunked data with no Content-Length, the check
# is skipped here and the streaming limit in
# make_embedding_from_upload still applies (after spooling).
# ============================================================

@app.middleware("http")
async def guard_verify_upload(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/verify-face":
        try:
            authenticate(
                request.headers.get("authorization"),
                VERIFY_API_TOKEN,
            )
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )

        content_length = request.headers.get("content-length")

        if content_length:
            try:
                declared = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length."},
                )

            if declared > MAX_UPLOAD_REQUEST_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Image exceeds the 10 MB limit."},
                )

    return await call_next(request)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# ENROLL STUDENT
# MongoDB Atlas Trigger sends JSON containing at least:
# {
#   "student_id": "24-character MongoDB ObjectId",
#   "image_url": "HTTPS Cloudinary image URL",
#   "student_data": { "Name": "...", "RollNO": "...", ... }
# }
# Additional fields sent at the top level are also saved in the Qdrant payload.
# ============================================================

@app.post("/jobs/face-embedding")
def enroll_student(
    job: StudentPhotoJob,
    authorization: str | None = Header(default=None),
):
    authenticate(authorization, TRIGGER_SECRET)

    student_id = job.student_id.lower()

    logger.info("Starting face enrollment.")

    vector = make_embedding(str(job.image_url))

    try:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=student_point_id(student_id),
                    vector=vector,
                    payload=build_student_payload(job, student_id),
                )
            ],
            wait=True,
        )

    except Exception as error:
        logger.exception(
            "Qdrant enrollment failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Could not save the embedding.",
        )

    return {
        "stored": True,
        "student_id": student_id,
        "message": "Face embedding enrolled or updated.",
    }


# ============================================================
# FIND STUDENT FROM A CAMERA IMAGE
# Only the photo is sent; it is searched against all enrolled
# faces.
#
# Multipart field:
#   image
# ============================================================

@app.post("/verify-face")
def verify_face(
    image: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    # Also checked in the middleware; kept here as defense in depth.
    authenticate(authorization, VERIFY_API_TOKEN)

    submitted_vector = make_embedding_from_upload(image)

    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=submitted_vector,
            limit=MATCH_TOP_K,
            with_payload=True,
            with_vectors=False,
        ).points

    except Exception as error:
        logger.exception(
            "Qdrant search failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Could not compare the face embedding.",
        )

    # Results are sorted best-first. Strip any embedding/vector fields that
    # may have been saved inside the payload; keep nested student_data intact.
    embedding_field_names = {
        "embedding", "embeddings", "vector", "vectors",
        "face_embedding", "face_embeddings",
    }

    def looks_like_vector(value):
        # Catch-all: drop any long list of numbers, whatever its key name.
        return (
            isinstance(value, list)
            and len(value) >= 100
            and all(
                isinstance(x, (int, float)) and not isinstance(x, bool)
                for x in value
            )
        )

    def without_embeddings(value):
        if isinstance(value, dict):
            return {
                key: without_embeddings(item)
                for key, item in value.items()
                if key.lower() not in embedding_field_names
                and not looks_like_vector(item)
            }
        if isinstance(value, list):
            return [without_embeddings(item) for item in value]
        return value

    candidates = []
    for result in results:
        payload = without_embeddings(result.payload or {})
        candidates.append({
            **payload,
            "cosine_similarity": round(float(result.score), 4),
        })

    above = [
        c for c in candidates
        if c["cosine_similarity"] >= MATCH_THRESHOLD
    ]

    # If the top two are too close, the result is ambiguous.
    ambiguous = (
        len(above) >= 2
        and (
            above[0]["cosine_similarity"]
            - above[1]["cosine_similarity"]
        ) < MIN_MATCH_MARGIN
    )

    best_match = above[0] if above and not ambiguous else None

    return {
        "matched": best_match is not None,
        "best_match": best_match,
        "ambiguous": ambiguous,
        "candidates": candidates,
        "requires_review": True,
        "message": (
            "Similarity is not a probability or proof of identity. "
            "Review the result before making identity-related "
            "decisions."
        ),
    }