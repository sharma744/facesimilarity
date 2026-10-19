import os
import re
import uuid
import secrets
import tempfile
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import numpy as np
import requests

from deepface import DeepFace
from fastapi import FastAPI, Header, HTTPException, Form, File, UploadFile
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from pydantic import BaseModel, Field, HttpUrl
from qdrant_client import QdrantClient, models

# Lets Pillow decode HEIC/HEIF (iPhone photos).
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
MAX_IMAGE_PIXELS = 40_000_000           # decoded size limit (decompression-bomb guard)
MAX_IMAGE_SIDE = 1600                   # downscale before DeepFace to save memory
STUDENT_ID_RE = re.compile(r"[0-9a-fA-F]{24}")

ALLOWED_IMAGE_HOST = "res.cloudinary.com"

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30,
)


# ============================================================
# REQUEST MODELS
# ============================================================

class StudentPhotoJob(BaseModel):
    student_id: str = Field(pattern=r"^[0-9a-fA-F]{24}$")
    image_url: HttpUrl


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

def download_image(image_url: str) -> str:
    image_url = validate_cloudinary_url(image_url)
    temp_path = None
    succeeded = False

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

            # Loose sanity check only; real validation is by content.
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

                for chunk in response.iter_content(8192):
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

            opened.load()
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

            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

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
            detector_backend="opencv",
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

    except Exception as error:
        logger.exception(
            "Embedding generation failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=422,
            detail="Unable to generate a face embedding.",
        )


# ============================================================
# EMBEDDING FROM A RAW IMAGE FILE (shared by both paths)
# ============================================================

def embedding_from_raw_file(raw_path: str) -> list[float]:
    normalized_path = None

    try:
        normalized_path = normalize_image(raw_path)
        return get_embedding(normalized_path)
    finally:
        remove_quietly(normalized_path)
        remove_quietly(raw_path)


# ============================================================
# EMBEDDING FROM CLOUDINARY URL
# ============================================================

def make_embedding(image_url: str) -> list[float]:
    raw_path = download_image(image_url)
    return embedding_from_raw_file(raw_path)


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
    ensure_collection()
    logger.info("Qdrant initialization complete.")

    try:
        yield
    finally:
        qdrant.close()
        logger.info("Qdrant client closed.")


app = FastAPI(
    title="DiwaliHaat Face Verification API",
    version="2.1.0",
    lifespan=lifespan,
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# ENROLL STUDENT
# MongoDB Atlas Trigger sends JSON:
# {
#   "student_id": "24-character MongoDB ObjectId",
#   "image_url": "HTTPS Cloudinary image URL"
# }
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
                    payload={
                        "student_id": student_id,
                        "embedding_model": "ArcFace",
                    },
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
# VERIFY STUDENT USING A CAMERA IMAGE
#
# Multipart fields:
#   student_id
#   consent_confirmed
#   image
# ============================================================

@app.post("/verify-face")
def verify_claimed_student(
    student_id: str = Form(...),
    consent_confirmed: bool = Form(...),
    image: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    authenticate(authorization, VERIFY_API_TOKEN)

    if not consent_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Consent is required for face verification.",
        )

    if not STUDENT_ID_RE.fullmatch(student_id):
        raise HTTPException(
            status_code=422,
            detail="Invalid student ID.",
        )

    normalized_student_id = student_id.lower()

    submitted_vector = make_embedding_from_upload(image)

    try:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=submitted_vector,
            query_filter=models.Filter(
                must=[
                    models.HasIdCondition(
                        has_id=[
                            student_point_id(normalized_student_id)
                        ]
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        ).points

    except Exception as error:
        logger.exception(
            "Qdrant verification failed: %s",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Could not compare the face embedding.",
        )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No enrolled face found for this student ID.",
        )

    result = results[0]

    return {
        "matched_student_id": (
            result.payload or {}
        ).get("student_id"),
        "cosine_similarity": round(float(result.score), 4),
        "requires_review": True,
        "message": (
            "Similarity calculated. It is not a probability or proof "
            "of identity. Use a validated threshold and appropriate "
            "review before making identity-related decisions."
        ),
    }