
import os
import re
import uuid
import secrets
import tempfile
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import numpy as np
import requests
from deepface import DeepFace
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, HttpUrl, Field
from qdrant_client import QdrantClient, models


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRIGGER_SECRET = os.environ["PYTHON_TRIGGER_SECRET"]
VERIFY_API_TOKEN = os.environ["VERIFY_API_TOKEN"]

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

COLLECTION_NAME = "student_face_embeddings"
VECTOR_SIZE = 512
MAX_IMAGE_BYTES = 10 * 1024 * 1024

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30,
)


# --------------------------------------------------
# Request models
# --------------------------------------------------

class StudentPhotoJob(BaseModel):
    student_id: str = Field(pattern=r"^[0-9a-fA-F]{24}$")
    image_url: HttpUrl


class FaceVerificationRequest(BaseModel):
    student_id: str = Field(pattern=r"^[0-9a-fA-F]{24}$")
    image_url: HttpUrl
    consent_confirmed: bool


# --------------------------------------------------
# Authentication
# --------------------------------------------------

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


# --------------------------------------------------
# Qdrant helpers
# --------------------------------------------------

def student_point_id(student_id: str) -> str:
    """
    Create a deterministic Qdrant point ID for each MongoDB ObjectId.
    """
    normalized_id = student_id.lower()

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"diwalihaat:{normalized_id}",
        )
    )


def ensure_collection() -> None:
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )


# --------------------------------------------------
# Image download and validation
# --------------------------------------------------

def download_image(image_url: str) -> str:
    parsed = urlparse(image_url)

    # Only permit HTTPS Cloudinary URLs.
    if (
        parsed.scheme != "https"
        or parsed.hostname != "res.cloudinary.com"
        or parsed.port is not None
        or parsed.username
        or parsed.password
    ):
        raise HTTPException(
            status_code=400,
            detail="Image must use an approved Cloudinary HTTPS URL.",
        )

    temp_path = None

    try:
        with requests.get(
            image_url,
            timeout=(5, 15),
            stream=True,
            allow_redirects=False,
        ) as response:
            response.raise_for_status()

            content_type = (
                response.headers.get("Content-Type", "")
                .split(";")[0]
                .lower()
            )

            extensions = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/webp": ".webp",
            }

            if content_type not in extensions:
                raise HTTPException(
                    status_code=400,
                    detail="Only JPEG, PNG, and WebP images are supported.",
                )

            total_bytes = 0

            with tempfile.NamedTemporaryFile(
                suffix=extensions[content_type],
                delete=False,
            ) as temp:
                temp_path = temp.name

                for chunk in response.iter_content(
                    chunk_size=8192
                ):
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
                    detail="The downloaded image is empty.",
                )

            return temp_path

    except HTTPException:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    except requests.RequestException:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

        raise HTTPException(
            status_code=400,
            detail="Could not download the image.",
        )


# --------------------------------------------------
# ArcFace embedding generation
# --------------------------------------------------

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
                detail="No face detected in the image.",
            )

        if len(results) != 1:
            raise HTTPException(
                status_code=422,
                detail="Image must contain exactly one detectable face.",
            )

        vector = np.asarray(
            results[0]["embedding"],
            dtype=np.float32,
        )

        if vector.shape != (VECTOR_SIZE,):
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
                detail="The generated embedding is invalid.",
            )

        return vector.tolist()

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Unable to generate a face embedding.",
        )


def make_embedding(image_url: str) -> list[float]:
    image_path = download_image(image_url)

    try:
        return get_embedding(image_path)
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass


# --------------------------------------------------
# FastAPI lifecycle
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    yield
    qdrant.close()


app = FastAPI(
    title="DiwaliHaat Face Verification API",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Enroll / update a student's face embedding
# Called by the trusted Atlas Trigger or backend.
# --------------------------------------------------

@app.post("/jobs/face-embedding", status_code=202)
def enroll_student(
    job: StudentPhotoJob,
    authorization: str | None = Header(default=None),
):
    authenticate(authorization, TRIGGER_SECRET)

    student_id = job.student_id.lower()
    vector = make_embedding(str(job.image_url))

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

    return {
        "stored": True,
        "student_id": student_id,
        "message": "Face embedding enrolled or updated.",
    }


# --------------------------------------------------
# Verify a submitted image against a claimed student ID
# --------------------------------------------------

@app.post("/verify-face")
def verify_claimed_student(
    request: FaceVerificationRequest,
    authorization: str | None = Header(default=None),
):
    authenticate(authorization, VERIFY_API_TOKEN)

    if not request.consent_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Consent is required for face verification.",
        )

    student_id = request.student_id.lower()

    # Retrieve only the claimed student's enrolled vector.
    points = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[student_point_id(student_id)],
        with_vectors=True,
        with_payload=True,
    )

    if not points:
        raise HTTPException(
            status_code=404,
            detail="No enrolled face found for this student ID.",
        )

    stored_vector = points[0].vector

    if not isinstance(stored_vector, list):
        raise HTTPException(
            status_code=500,
            detail="Stored embedding is unavailable.",
        )

    submitted_vector = np.asarray(
        make_embedding(str(request.image_url)),
        dtype=np.float32,
    )

    enrolled_vector = np.asarray(
        stored_vector,
        dtype=np.float32,
    )

    denominator = (
        np.linalg.norm(submitted_vector)
        * np.linalg.norm(enrolled_vector)
    )

    if denominator == 0:
        raise HTTPException(
            status_code=422,
            detail="Unable to compare face embeddings.",
        )

    similarity = float(
        np.dot(submitted_vector, enrolled_vector)
        / denominator
    )

    # This is the MongoDB ObjectId stored in the Qdrant payload.
    stored_student_id = (
        points[0].payload or {}
    ).get("student_id")

    return {
        "matched_student_id": stored_student_id,
        "cosine_similarity": round(similarity, 4),
        "requires_review": True,
        "message": (
            "Similarity calculated for the claimed student. "
            "The score is not a probability or proof of identity. "
            "Use validated thresholds, liveness checks, and "
            "authorized human review before making decisions."
        ),
    }
