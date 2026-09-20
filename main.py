"""Scam Message Detection API.

Train the model, then start the server:

    python train.py
    uvicorn main:app --host 0.0.0.0 --port 8765 --reload

Web UI:            http://127.0.0.1:8765/
Interactive docs:  http://127.0.0.1:8765/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sklearn.pipeline import Pipeline

from classifier import SAMPLE_MESSAGES, classify_message, load_model
from preprocessing import ensure_nltk_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Populated during application startup. Remains None if loading fails.
ml_model: Pipeline | None = None


class MessageRequest(BaseModel):
    """Incoming SMS / chat message to classify."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Raw message body (SMS, Viber, or similar).",
        examples=["URGENT: Your bank account has been locked. Verify at http://secure-bank-login.com"],
    )

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace-only")
        return value


class Signal(BaseModel):
    term: str
    weight: float = Field(..., ge=0.0)


class DetectionResponse(BaseModel):
    """Classification result for a single message."""

    original_text: str
    prediction: str = Field(..., description='Either "Scam" or "Legitimate".')
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted-class probability from Random Forest.",
    )
    scam_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability the message is a scam, regardless of the predicted label.",
    )
    signals: list[Signal] = Field(
        default_factory=list,
        description="Phrases in the message that most influenced the model.",
    )
    advice: str


class BatchDetectRequest(BaseModel):
    """One SMS per list item, for inbox-style screening."""

    messages: list[MessageRequest] = Field(..., min_length=1, max_length=50)


class BatchDetectResponse(BaseModel):
    results: list[DetectionResponse]
    scam_count: int
    legitimate_count: int


class SampleMessage(BaseModel):
    id: str
    title: str
    text: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Download NLTK data and load the classifier once at process start."""
    global ml_model

    ensure_nltk_data()
    try:
        ml_model = load_model()
        logger.info("Loaded classification pipeline.")
    except FileNotFoundError:
        logger.error(
            "scam_detector.joblib is missing. Detection endpoints "
            "will return HTTP 503 until you run `python train.py`."
        )
        ml_model = None
    except (RuntimeError, TypeError):
        logger.exception("Could not initialize the ML model.")
        ml_model = None

    yield
    ml_model = None


app = FastAPI(
    title="Scam Message Detection API",
    description=(
        "Proof-of-concept classifier that labels SMS / chat messages as "
        "Legitimate or Scam using TF-IDF features and a Random Forest. "
        "Open the web UI at / to paste a message and inspect contributing phrases."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def require_model() -> Pipeline:
    if ml_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The classification model is not loaded. "
                "Run `python train.py` to create scam_detector.joblib, then restart the server."
            ),
        )
    return ml_model


def run_detection(text: str) -> DetectionResponse:
    model = require_model()
    try:
        payload = classify_message(model, text)
    except ValueError as exc:
        logger.exception("Model produced an invalid result.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inference failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to classify the message.",
        ) from exc
    return DetectionResponse.model_validate(payload)


@app.get("/", include_in_schema=False)
def web_ui() -> FileResponse:
    """Serve the scan console."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web UI is not installed.")
    return FileResponse(index_path)


@app.get("/api", tags=["meta"])
def api_info() -> dict[str, Any]:
    """Service descriptor for API clients."""
    return {
        "name": "Scam Message Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "ui": "/",
        "detect": "POST /api/v1/detect",
        "batch": "POST /api/v1/detect/batch",
        "examples": "GET /api/v1/examples",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Report whether the process successfully loaded ``scam_detector.joblib``."""
    loaded = ml_model is not None
    return HealthResponse(
        status="ok" if loaded else "model_unavailable",
        model_loaded=loaded,
    )


@app.get("/api/v1/examples", response_model=list[SampleMessage], tags=["detection"])
def list_examples() -> list[SampleMessage]:
    """Canned SMS samples for the UI and for demos."""
    return [SampleMessage.model_validate(item) for item in SAMPLE_MESSAGES]


@app.post(
    "/api/v1/detect",
    response_model=DetectionResponse,
    tags=["detection"],
    summary="Classify a message as Scam or Legitimate",
)
def detect_scam(payload: MessageRequest) -> DetectionResponse:
    """Run the loaded pipeline on ``payload.text`` and return the prediction."""
    return run_detection(payload.text)


@app.post(
    "/api/v1/detect/batch",
    response_model=BatchDetectResponse,
    tags=["detection"],
    summary="Classify up to 50 messages in one request",
)
def detect_batch(payload: BatchDetectRequest) -> BatchDetectResponse:
    """Screen a list of messages; useful for pasted inboxes."""
    results = [run_detection(item.text) for item in payload.messages]
    scam_count = sum(1 for item in results if item.prediction == "Scam")
    return BatchDetectResponse(
        results=results,
        scam_count=scam_count,
        legitimate_count=len(results) - scam_count,
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
