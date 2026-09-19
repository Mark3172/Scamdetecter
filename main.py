"""Scam Message Detection API.

Train the model, then start the server:

    python train.py
    uvicorn main:app --host 0.0.0.0 --port 8765 --reload

Interactive docs: http://127.0.0.1:8765/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sklearn.pipeline import Pipeline

from preprocessing import ensure_nltk_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "scam_detector.joblib"
ALLOWED_LABELS = {"Legitimate", "Scam"}

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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the joblib pipeline, raising a clear error if it is missing."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Train it first with: python train.py"
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001 - surface corrupt-pickle errors to the caller
        raise RuntimeError(f"Failed to load model from '{model_path}': {exc}") from exc

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("Loaded object is not a scikit-learn classifier pipeline.")

    logger.info("Loaded model from %s", model_path)
    return model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Download NLTK data and load the classifier once at process start."""
    global ml_model

    ensure_nltk_data()
    try:
        ml_model = load_model()
    except FileNotFoundError:
        logger.error(
            "scam_detector.joblib is missing. The /api/v1/detect endpoint "
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
        "Legitimate or Scam using TF-IDF features and a Random Forest."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    """Lightweight service descriptor for humans hitting the base URL."""
    return {
        "name": "Scam Message Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "detect": "POST /api/v1/detect",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Report whether the process successfully loaded ``scam_detector.joblib``."""
    loaded = ml_model is not None
    return HealthResponse(
        status="ok" if loaded else "model_unavailable",
        model_loaded=loaded,
    )


def classify_message(model: Pipeline, text: str) -> tuple[str, float]:
    """Return (label, predicted-class probability) for a single message."""
    predicted = model.predict([text])[0]
    label = str(predicted)
    if label not in ALLOWED_LABELS:
        raise ValueError(f"Model returned unsupported label: {label!r}")

    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    try:
        class_index = classes.index(label)
    except ValueError as exc:
        raise ValueError(f"Predicted label {label!r} is not in model.classes_") from exc

    confidence = float(probabilities[class_index])
    # Guard against numeric noise outside [0, 1].
    confidence = min(1.0, max(0.0, confidence))
    return label, confidence


@app.post(
    "/api/v1/detect",
    response_model=DetectionResponse,
    tags=["detection"],
    summary="Classify a message as Scam or Legitimate",
)
def detect_scam(payload: MessageRequest) -> DetectionResponse:
    """Run the loaded pipeline on ``payload.text`` and return the prediction."""
    if ml_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The classification model is not loaded. "
                "Run `python train.py` to create scam_detector.joblib, then restart the server."
            ),
        )

    try:
        prediction, confidence = classify_message(ml_model, payload.text)
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

    return DetectionResponse(
        original_text=payload.text,
        prediction=prediction,
        confidence_score=round(confidence, 4),
    )
