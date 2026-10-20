
import os
import uuid
import secrets
import tempfile
import logging
import threading
from contextlib import asynccontextmanager, contextmanager
from urllib.parse import urlparse, urljoin

import numpy as np
import requests

from deepface import DeepFace
from fastapi import FastAPI, Header, HTTPException, File, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from qdrant_client import QdrantClient, models


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("diwalihaat-face-api")

TRIGGER_SECRET = os.environ["PYTHON_TRIGGER_SECRET"]
VERIFY_API_TOKEN = os.environ["VERIFY_API_TOKEN"]

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

CLOUDINARY_CLOUD_NAME = os.environ.get(
    "CLOUDINARY_CLOUD_NAME", ""
).strip()

COLLECTION_NAME = "student_face_embeddings"
VECTOR_SIZE = 512
EMBEDDING_MODEL = "ArcFace"

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_SIDE = 1600

MAX_REDIRECTS = 3
DOWNLOAD_CONNECT_TIMEOUT = 8
DOWNLOAD_READ_TIMEOUT = 30

MATCH_TOP_K = 5
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.40"))
MIN_MATCH_MARGIN = float(os.environ.get("MIN_MATCH_MARGIN", "0.05"))

ALLOWED_IMAGE_HOST = "res.cloudinary.com"

register_heif_opener()
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30,
)

# Protect the memory-intensive DeepFace operation.
# A busy service rejects another embedding request instead of
# allowing multiple model inferences to exhaust available RAM.
embedding_lock = threading.Lock()


# ============================================================
# REQUEST MODEL
# ============================================================

class StudentPhotoJob(BaseModel):
    model_config = ConfigDict(extra="allow")

    student_id: str = Field(pattern=r"^[0-9a-fA-F]{24}$")
    image_url: HttpUrl
    student_data: dict = Field(default_factory=dict)


def build_student_payload(job: StudentPhotoJob, student_id: str) -> dict:
    data = job.model_dump(mode="json")
    student_data = data.pop("student_data", {}) or {}

    return {
        **data,
        "student_id": student_id,
        "embedding_model": EMBEDDING_MODEL,
        "student_data": student_data,
    }


# ============================================================
# AUTHENTICATION
# ============================================================

def authenticate(
    authorization: str | None,
    expected_token: str,
) -> None:
    expected = f"Bearer {expected_token}"

    if not authorization or not secrets.compare_digest(
        authorization.encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ============================================================
# QDRANT POINT ID
# Keep this ID scheme consistent with existing enrollment data.
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
        except Exception:
            # Handle a concurrent worker creating the collection.
            if not qdrant.collection_exists(COLLECTION_NAME):
                raise

    collection = qdrant.get_collection(COLLECTION_NAME)
    config = collection.config.params.vectors

    if not isinstance(config, models.VectorParams):
        raise RuntimeError(
            "Qdrant collection must use a single dense vector."
        )

    if config.size != VECTOR_SIZE:
        raise RuntimeError(
            f"Expected {VECTOR_SIZE} dimensions; found {config.size}."
        )

    if config.distance != models.Distance.COSINE:
        raise RuntimeError("Qdrant collection must use cosine distance.")

    logger.info("Qdrant collection validated.")


# ============================================================
# TEMPORARY FILES
# ============================================================

def remove_quietly(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            logger.warning("Temporary file cleanup failed.")


@contextmanager
def embedding_slot():
    acquired = embedding_lock.acquire(blocking=False)

    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="Face processing is busy. Please retry shortly.",
            headers={"Retry-After": "3"},
        )

    try:
        yield
    finally:
        embedding_lock.release()


# ============================================================
# CLOUDINARY URL VALIDATION
# Validate every URL, including redirect destinations.
# ============================================================

def validate_cloudinary_url(image_url: str) -> str:
    try:
        parsed = urlparse(image_url)
        port = parsed.port
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid image URL.")

    if (
        parsed.scheme != "https"
        or parsed.hostname != ALLOWED_IMAGE_HOST
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith("/")
        or parsed.query
        or parsed.fragment
        or "/image/upload/" not in parsed.path
    ):
        raise HTTPException(
            status_code=400,
            detail="A valid HTTPS Cloudinary image URL is required.",
        )

    if CLOUDINARY_CLOUD_NAME:
        expected_prefix = (
            f"/{CLOUDINARY_CLOUD_NAME}/image/upload/"
        )
        if not parsed.path.startswith(expected_prefix):
            raise HTTPException(
                status_code=400,
                detail="Image does not belong to the configured Cloudinary account.",
            )

    return image_url


# ============================================================
# STREAM CLOUDINARY IMAGE TO DISK
# Bounded size, timeouts, safe redirect handling.
# ============================================================

def download_image(image_url: str) -> str:
    current_url = validate_cloudinary_url(image_url)
    temp_path = None

    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            with requests.get(
                current_url,
                timeout=(
                    DOWNLOAD_CONNECT_TIMEOUT,
                    DOWNLOAD_READ_TIMEOUT,
                ),
                stream=True,
                allow_redirects=False,
                headers={"User-Agent": "DiwaliHaat-FaceAPI/1.0"},
            ) as response:

                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")

                    if not location or redirect_count >= MAX_REDIRECTS:
                        raise HTTPException(
                            status_code=502,
                            detail="Cloudinary returned an invalid or excessive redirect.",
                        )

                    next_url = urljoin(current_url, location)

                    # Never follow an unchecked redirect.
                    current_url = validate_cloudinary_url(next_url)
                    continue

                if response.status_code != 200:
                    logger.warning(
                        "Image provider returned HTTP %s.",
                        response.status_code,
                    )
                    raise HTTPException(
                        status_code=502,
                        detail="Cloudinary could not return the image.",
                    )

                content_type = (
                    response.headers.get("Content-Type", "")
                    .split(";")[0]
                    .strip()
                    .lower()
                )

                if not content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=415,
                        detail="The URL did not return an image.",
                    )

                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > MAX_IMAGE_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail="Image exceeds the 10 MB limit.",
                            )
                    except ValueError:
                        pass

                total_bytes = 0

                with tempfile.NamedTemporaryFile(
                    suffix=".img",
                    delete=False,
                ) as temp:
                    temp_path = temp.name

                    for chunk in response.iter_content(chunk_size=8192):
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

                logger.info(
                    "Image download completed (%s bytes).",
                    total_bytes,
                )
                result_path = temp_path
                temp_path = None
                return result_path

        raise HTTPException(
            status_code=502,
            detail="Too many image redirects.",
        )

    except HTTPException:
        raise

    except requests.Timeout:
        logger.warning("Cloudinary image download timed out.")
        raise HTTPException(
            status_code=504,
            detail="Image download timed out. Please retry.",
        )

    except requests.RequestException as error:
        logger.warning(
            "Cloudinary request failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve the Cloudinary image.",
        )

    finally:
        remove_quietly(temp_path)


# ============================================================
# IMAGE NORMALIZATION
# Supports formats Pillow can decode, including registered HEIC.
# ============================================================

def normalize_image(image_path: str) -> str:
    output_path = None

    try:
        with Image.open(image_path) as opened:
            width, height = opened.size

            if width <= 0 or height <= 0:
                raise HTTPException(
                    status_code=415,
                    detail="Invalid image dimensions.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large.",
                )

            opened.load()
            image = ImageOps.exif_transpose(opened)

            has_transparency = (
                image.mode in ("RGBA", "LA", "PA")
                or "transparency" in image.info
            )

            if has_transparency:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(
                    rgba,
                    mask=rgba.getchannel("A"),
                )
                image = background
            else:
                image = image.convert("RGB")

            image.thumbnail(
                (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
                Image.Resampling.LANCZOS,
            )

            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False,
            ) as output:
                output_path = output.name

            image.save(
                output_path,
                format="JPEG",
                quality=90,
                optimize=True,
            )

            image.close()

        result_path = output_path
        output_path = None
        logger.info("Image normalized successfully.")
        return result_path

    except HTTPException:
        raise

    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ) as error:
        logger.warning(
            "Invalid image: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=415,
            detail="Unsupported, corrupted, or invalid image.",
        )

    except Exception as error:
        logger.exception(
            "Unexpected image normalization failure: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Image processing failed.",
        )

    finally:
        remove_quietly(output_path)


# ============================================================
# ARCFACE EMBEDDING
# ============================================================

def get_embedding(image_path: str) -> list[float]:
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name=EMBEDDING_MODEL,
            detector_backend="opencv",
            enforce_detection=True,
        )

        if not results:
            raise HTTPException(
                status_code=422,
                detail="No face detected. Try a clearer, front-facing photo.",
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
                "Unexpected embedding shape: %s",
                vector.shape,
            )
            raise HTTPException(
                status_code=500,
                detail="Unexpected face embedding dimensions.",
            )

        if not np.all(np.isfinite(vector)):
            raise HTTPException(
                status_code=422,
                detail="The generated face embedding is invalid.",
            )

        norm = float(np.linalg.norm(vector))

        if not np.isfinite(norm) or norm == 0:
            raise HTTPException(
                status_code=422,
                detail="The generated face embedding is invalid.",
            )

        return vector.tolist()

    except HTTPException:
        raise

    except Exception as error:
        message = str(error).lower()

        # DeepFace/detector versions use different messages for
        # failures to detect or extract a face.
        face_error_terms = (
            "face could not be detected",
            "no face detected",
            "failed to detect face",
            "face detection failed",
            "cannot extract faces",
            "could not extract faces",
        )

        if any(term in message for term in face_error_terms):
            logger.info(
                "Face detection did not succeed (%s).",
                type(error).__name__,
            )
            raise HTTPException(
                status_code=422,
                detail="No usable face detected. Try a clearer photo.",
            )

        # Model initialization, missing weights and inference failures
        # are server failures, not proof that the submitted face is invalid.
        logger.exception(
            "ArcFace inference failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Face recognition is temporarily unavailable. Please retry.",
        )


# ============================================================
# SHARED IMAGE PROCESSING
# ============================================================

def embedding_from_raw_file(raw_path: str) -> list[float]:
    normalized_path = None

    try:
        normalized_path = normalize_image(raw_path)
        return get_embedding(normalized_path)
    finally:
        remove_quietly(normalized_path)
        remove_quietly(raw_path)


def make_embedding(image_url: str) -> list[float]:
    raw_path = download_image(image_url)
    return embedding_from_raw_file(raw_path)


def make_embedding_from_upload(image: UploadFile) -> list[float]:
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

        result_path = raw_path
        raw_path = None
        logger.info("Camera upload received (%s bytes).", total_bytes)

    except HTTPException:
        raise

    except Exception as error:
        logger.warning(
            "Camera upload read failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image.",
        )

    finally:
        remove_quietly(raw_path)
        image.file.close()

    return embedding_from_raw_file(result_path)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DiwaliHaat Face API.")
    ensure_collection()
    logger.info("Qdrant initialization complete.")

    try:
        yield
    finally:
        qdrant.close()
        logger.info("Qdrant client closed.")


app = FastAPI(
    title="DiwaliHaat Face Verification API",
    version="3.0.0",
    lifespan=lifespan,
)


# ============================================================
# HEALTH CHECKS
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        exists = qdrant.collection_exists(COLLECTION_NAME)

        if not exists:
            raise HTTPException(
                status_code=503,
                detail="Qdrant collection is unavailable.",
            )

        return {"status": "ready", "qdrant": "connected"}

    except HTTPException:
        raise

    except Exception:
        logger.exception("Readiness check failed.")
        raise HTTPException(
            status_code=503,
            detail="Face service dependencies are unavailable.",
        )


# ============================================================
# ENROLL / UPDATE STUDENT
# POST /jobs/face-embedding
# ============================================================

@app.post("/jobs/face-embedding")
def enroll_student(
    job: StudentPhotoJob,
    authorization: str | None = Header(default=None),
):
    authenticate(authorization, TRIGGER_SECRET)

    student_id = job.student_id.lower()
    logger.info("Starting enrollment for student ID %s.", student_id)

    with embedding_slot():
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
            detail="Could not save the face embedding. Please retry.",
        )

    logger.info("Enrollment completed for student ID %s.", student_id)

    return {
        "stored": True,
        "student_id": student_id,
        "message": "Face embedding enrolled or updated.",
    }


# ============================================================
# VERIFY CAMERA PHOTO
# POST /verify-face
# Multipart field: image
# ============================================================

@app.post("/verify-face")
def verify_face(
    image: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    authenticate(authorization, VERIFY_API_TOKEN)

    with embedding_slot():
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
            "Qdrant face search failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Could not compare the face embedding.",
        )

    candidates = []

    for result in results:
        payload = result.payload or {}
        nested_data = payload.get("student_data") or {}

        if not isinstance(nested_data, dict):
            nested_data = {}

        student_data = {
            **payload,
            **nested_data,
        }

        student_data.pop("student_data", None)
        student_data.pop("embedding_model", None)

        candidates.append({
            **student_data,
            "cosine_similarity": round(float(result.score), 4),
        })

    above = [
        candidate
        for candidate in candidates
        if candidate["cosine_similarity"] >= MATCH_THRESHOLD
    ]

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
            "Review the result before making identity-related decisions."
        ),
    }
