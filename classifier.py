"""Inference helpers shared by the API: classify, explain, and sample prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

ALLOWED_LABELS = {"Legitimate", "Scam"}
MODEL_PATH = Path(__file__).resolve().parent / "scam_detector.joblib"
MAX_SIGNALS = 6

SAMPLE_MESSAGES: list[dict[str, str]] = [
    {
        "id": "bank-lock",
        "title": "Bank lock alert",
        "text": (
            "URGENT: Your bank account has been locked. "
            "Verify immediately at http://secure-bank-login.com"
        ),
    },
    {
        "id": "gift-card",
        "title": "Gift card prize",
        "text": "Claim your free $500 gift card before it expires. Click http://prize-win.com",
    },
    {
        "id": "dinner",
        "title": "Dinner plans",
        "text": "Hey, are we still on for dinner at 7 tonight?",
    },
    {
        "id": "pickup",
        "title": "School pickup",
        "text": "The kids are at soccer practice until 5. Can you grab them?",
    },
]


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Load the joblib pipeline, raising a clear error if it is missing."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Train it first with: python train.py"
        )

    try:
        model = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model from '{model_path}': {exc}") from exc

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("Loaded object is not a scikit-learn classifier pipeline.")

    return model


def display_term(term: str) -> str:
    """Map internal feature names to labels that make sense in the UI."""
    if term == "urlplaceholder":
        return "link / URL"
    return term.replace("urlplaceholder", "url")


def _class_probability(model: Pipeline, probabilities: np.ndarray, label: str) -> float:
    classes = list(model.classes_)
    try:
        class_index = classes.index(label)
    except ValueError as exc:
        raise ValueError(f"Label {label!r} is not in model.classes_") from exc
    score = float(probabilities[class_index])
    return min(1.0, max(0.0, score))


def explain_signals(model: Pipeline, text: str, top_k: int = MAX_SIGNALS) -> list[dict[str, Any]]:
    """Rank tokens present in the message by TF-IDF × Random Forest importance.

    This is a local explanation for the PoC, not a full SHAP attribution.
    """
    if "tfidf" not in model.named_steps or "clf" not in model.named_steps:
        return []

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]
    if not hasattr(classifier, "feature_importances_"):
        return []

    matrix = vectorizer.transform([text])
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    importances = np.asarray(classifier.feature_importances_, dtype=float)
    row = matrix.toarray()[0]
    present = np.flatnonzero(row > 0)
    if present.size == 0:
        return []

    scores = row[present] * importances[present]
    order = np.argsort(scores)[::-1]
    signals: list[dict[str, Any]] = []
    for idx in order:
        weight = float(scores[idx])
        if weight <= 0:
            continue
        term = str(feature_names[present[idx]])
        signals.append({"term": display_term(term), "weight": round(weight, 4)})
        if len(signals) >= top_k:
            break
    return signals


def advice_for(prediction: str, confidence: float) -> str:
    """Short, actionable copy shown next to the verdict."""
    if prediction == "Scam":
        if confidence >= 0.75:
            return (
                "Treat this as a scam. Do not tap links, share codes, "
                "or send money. Contact the company using a number you already trust."
            )
        return (
            "This looks suspicious. Confirm the request through an official app "
            "or a known phone number before you reply."
        )
    if confidence >= 0.75:
        return "This reads like a normal message. Stay cautious if it later asks for money or a login."
    return (
        "The model is unsure. If the sender asks for payment, passwords, "
        "or personal details, verify independently."
    )


def classify_message(model: Pipeline, text: str) -> dict[str, Any]:
    """Return label, probabilities, contributing phrases, and advice."""
    predicted = str(model.predict([text])[0])
    if predicted not in ALLOWED_LABELS:
        raise ValueError(f"Model returned unsupported label: {predicted!r}")

    probabilities = model.predict_proba([text])[0]
    confidence = _class_probability(model, probabilities, predicted)
    scam_probability = _class_probability(model, probabilities, "Scam")

    return {
        "original_text": text,
        "prediction": predicted,
        "confidence_score": round(confidence, 4),
        "scam_probability": round(scam_probability, 4),
        "signals": explain_signals(model, text),
        "advice": advice_for(predicted, confidence),
    }
