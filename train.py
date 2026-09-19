"""Train a TF-IDF + Random Forest pipeline for scam / smishing detection.

Run from the project root:

    python train.py

This writes ``scam_detector.joblib`` next to this file. Retrain whenever
you change the mock dataset or preprocessing.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import ensure_nltk_data, preprocess_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "scam_detector.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.25

# ---------------------------------------------------------------------------
# Mock SMS / chat corpus
# At least 10 legitimate and 10 scam examples, written in a realistic register.
# Labels must stay in {"Legitimate", "Scam"} — the API returns these strings.
# ---------------------------------------------------------------------------
MOCK_MESSAGES: list[dict[str, str]] = [
    # Legitimate
    {"text": "Hey, are we still on for dinner at 7 tonight?", "label": "Legitimate"},
    {"text": "Mom, don't forget to pick up milk on your way home.", "label": "Legitimate"},
    {"text": "Thanks for the birthday wishes! Had a great day.", "label": "Legitimate"},
    {"text": "The standup moved to conference room B at 10am.", "label": "Legitimate"},
    {"text": "I'll be about 10 minutes late, traffic on I-5 is awful.", "label": "Legitimate"},
    {"text": "Can you send me the notes from yesterday's class?", "label": "Legitimate"},
    {"text": "Happy anniversary! Love you. See you after work.", "label": "Legitimate"},
    {"text": "See you at the gym tomorrow morning around 6:30.", "label": "Legitimate"},
    {"text": "Did you finish the quarterly report for Friday's review?", "label": "Legitimate"},
    {"text": "Looking forward to the weekend hike. I'll bring snacks.", "label": "Legitimate"},
    {"text": "Please review the draft when you get a chance. No rush.", "label": "Legitimate"},
    {"text": "The kids are at soccer practice until 5. Can you grab them?", "label": "Legitimate"},
    {"text": "Your dental appointment is confirmed for Tuesday at 3pm.", "label": "Legitimate"},
    {"text": "Flight UA 1842 check-in is open. See you at the gate.", "label": "Legitimate"},
    {"text": "Got your text. I'll call you after the meeting wraps up.", "label": "Legitimate"},
    {"text": "Pizza night at our place Saturday? Bring whatever you like.", "label": "Legitimate"},
    {"text": "Reminder: library books are due Friday. No late fees if you renew online.", "label": "Legitimate"},
    {"text": "The plumber is coming between 2 and 4. I'll leave the side door unlocked.", "label": "Legitimate"},
    {"text": "Great game last night. Want to grab coffee and recap this afternoon?", "label": "Legitimate"},
    {"text": "Your prescription is ready for pickup at the pharmacy on Main Street.", "label": "Legitimate"},
    # Scam / smishing
    {
        "text": "Your package is held at customs. Pay a $2.99 fee now: http://bit.ly/pay-fee-now",
        "label": "Scam",
    },
    {
        "text": "URGENT: Your bank account has been locked. Verify immediately at http://secure-bank-login.com",
        "label": "Scam",
    },
    {
        "text": "Congratulations! You won a $1000 Amazon gift card. Claim it now: http://free-gift-prize.com",
        "label": "Scam",
    },
    {
        "text": "Your Apple ID was used on a new iPhone. Tap here to secure your account or it will be locked.",
        "label": "Scam",
    },
    {
        "text": "IRS notice: you owe back taxes. Pay immediately to avoid arrest. Call 1-800-555-0199 now.",
        "label": "Scam",
    },
    {
        "text": "We tried to deliver your parcel today. Reschedule delivery and pay the fee: http://fedex-delivery-fee.com",
        "label": "Scam",
    },
    {
        "text": "Your Netflix payment failed. Update billing details at http://netflix-secure-update.net to keep watching.",
        "label": "Scam",
    },
    {
        "text": "Unusual activity detected on your account. Confirm your SSN and password here to restore access.",
        "label": "Scam",
    },
    {
        "text": "You have been selected for a government grant of $8500. Reply with your bank details to receive funds.",
        "label": "Scam",
    },
    {
        "text": "Your mailbox is full. Click this link within 24 hours to avoid permanent account suspension.",
        "label": "Scam",
    },
    {
        "text": "Double your Bitcoin in 24 hours with our guaranteed crypto investment. Send BTC to this wallet now.",
        "label": "Scam",
    },
    {
        "text": "FINAL NOTICE: Your Social Security number has been suspended. Call this number immediately to reactivate.",
        "label": "Scam",
    },
    {
        "text": "You won a Publisher Clearing House prize. Pay a small shipping fee to receive your $5000 check.",
        "label": "Scam",
    },
    {
        "text": "Your car warranty is expiring today. Call now to extend coverage before it is too late.",
        "label": "Scam",
    },
    {
        "text": "Your Amazon Prime will be charged $399 unless you cancel here: http://amazon-prime-cancel.xyz",
        "label": "Scam",
    },
    {
        "text": "Your USPS package cannot be delivered. Confirm your address and card to release it: http://usps-hold.com",
        "label": "Scam",
    },
    {
        "text": "Claim your free $500 gift card before it expires. Click http://prize-win.com now.",
        "label": "Scam",
    },
    {
        "text": "Your PayPal account has been limited. Log in and verify your identity at http://paypal-secure-login.com",
        "label": "Scam",
    },
    {
        "text": "Bank alert: unusual transfer of $2,400. Confirm your PIN here to cancel the transaction.",
        "label": "Scam",
    },
    {
        "text": "You have an unpaid toll. Pay immediately or your license will be suspended: http://toll-pay-now.com",
        "label": "Scam",
    },
    {
        "text": "Your Netflix payment failed. Update your billing details to keep watching.",
        "label": "Scam",
    },
    {
        "text": "Your Netflix payment failed. Update billing now.",
        "label": "Scam",
    },
    {
        "text": "We noticed a login from a new device. Reply with your password to confirm it was you.",
        "label": "Scam",
    },
    {
        "text": "Your account will be closed in 24 hours unless you verify your identity now.",
        "label": "Scam",
    },
]


def load_dataset() -> pd.DataFrame:
    """Build a labeled DataFrame from the mock message list."""
    frame = pd.DataFrame(MOCK_MESSAGES)
    if frame.empty:
        raise ValueError("Mock dataset is empty; add labeled messages before training.")

    required_columns = {"text", "label"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"].str.len() > 0].copy()
    frame["label"] = frame["label"].astype(str)

    allowed_labels = {"Legitimate", "Scam"}
    unexpected = set(frame["label"].unique()) - allowed_labels
    if unexpected:
        raise ValueError(f"Unexpected labels in dataset: {sorted(unexpected)}")

    counts = frame["label"].value_counts()
    logger.info("Loaded %d messages (%s)", len(frame), counts.to_dict())
    if (counts.reindex(["Legitimate", "Scam"]).fillna(0) < 10).any():
        raise ValueError("Need at least 10 Legitimate and 10 Scam examples.")

    return frame.reset_index(drop=True)


def build_pipeline() -> Pipeline:
    """TF-IDF unigrams/bigrams feeding a class-balanced Random Forest."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    lowercase=False,  # already handled in preprocess_text
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=5000,
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    # Tiny TF-IDF matrices need every term eligible at each split.
                    max_features=None,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_holdout(pipeline: Pipeline, dataset: pd.DataFrame) -> None:
    """Fit on a stratified split and log a classification report."""
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            dataset["text"],
            dataset["label"],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=dataset["label"],
        )
    except ValueError as exc:
        logger.warning("Skipping holdout evaluation: %s", exc)
        return

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    report = classification_report(y_test, y_pred, digits=3, zero_division=0)
    logger.info("Holdout evaluation (%d test messages):\n%s", len(x_test), report)


def train_and_save(output_path: Path = MODEL_PATH) -> Path:
    """Train on the full mock corpus and persist the pipeline with joblib."""
    ensure_nltk_data()
    dataset = load_dataset()

    evaluate_holdout(build_pipeline(), dataset)

    pipeline = build_pipeline()
    pipeline.fit(dataset["text"], dataset["label"])
    train_accuracy = float(pipeline.score(dataset["text"], dataset["label"]))
    logger.info("Full-corpus training accuracy: %.3f", train_accuracy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Saved trained pipeline to %s", output_path)
    return output_path


def main() -> int:
    try:
        train_and_save()
    except Exception:
        logger.exception("Training failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
