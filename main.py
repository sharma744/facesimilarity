
import os
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
from pydantic import BaseModel, Field, HttpUrl
from qdrant_client import QdrantClient, models


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diwalihaat-face-api")

TRIGGER_SECRET = os.environ["PYTHON_TRIGGER_SECRET"]
VERIFY_API_TOKEN = os.environ["VERIFY_API_TOKEN"]

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

COLLECTION_NAME = "student_face_embeddings"
VECTOR_SIZE = 512
MAX_IMAGE_BYTES = 10 * 1024 * 1024

ALLOWED_IMAGE_HOST = "res.cloudinary.com"
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}

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

    if not authorization or not secrets.compare_digest(
        authorization, expected
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
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection.")
        return

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

    return image_url


# ============================================================
# DOWNLOAD CLOUDINARY IMAGE SAFELY
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

            if content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=415,
                    detail="Only JPEG, PNG, and WebP images are supported.",
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
                suffix=ALLOWED_IMAGE_TYPES[content_type],
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
        if (
            not succeeded
            and temp_path
            and os.path.exists(temp_path)
        ):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning(
                    "Could not remove partial image download."
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
# EMBEDDING FROM CLOUDINARY URL
# ============================================================

def make_embedding(image_url: str) -> list[float]:
    image_path = download_image(image_url)

    try:
        return get_embedding(image_path)
    finally:
        try:
            os.remove(image_path)
        except OSError:
            logger.warning(
                "Could not remove temporary image file."
            )


# ============================================================
# EMBEDDING FROM CAMERA UPLOAD
# ============================================================

def make_embedding_from_upload(
    image: UploadFile,
) -> list[float]:
    content_type = (
        image.content_type or ""
    ).split(";")[0].strip().lower()

    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    temp_path = None

    try:
        total_bytes = 0

        with tempfile.NamedTemporaryFile(
            suffix=ALLOWED_IMAGE_TYPES[content_type],
            delete=False,
        ) as temp:
            temp_path = temp.name

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

        return get_embedding(temp_path)

    finally:
        image.file.close()

        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning(
                    "Could not remove uploaded image file."
                )


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
    version="2.0.0",
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

    if not __import__("re").fullmatch(
        r"[0-9a-fA-F]{24}",
        student_id,
    ):
        raise HTTPException(
            status_code=422,
            detail="Invalid student ID.",
        )

    normalized_student_id = student_id.lower()

    try:
        submitted_vector = make_embedding_from_upload(image)
    finally:
        image.file.close()

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
