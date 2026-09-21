"""Shared text preprocessing for training and inference.

The trained sklearn pipeline pickles a reference to ``preprocess_text``.
Keeping that function in this module (instead of ``__main__``) is required
so ``joblib.load`` can resolve it when the API starts.
"""

from __future__ import annotations

import logging
import re
import string
from functools import lru_cache

import nltk
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
URL_PATTERN = _URL_PATTERN
_WHITESPACE_PATTERN = re.compile(r"\s+")
_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def ensure_nltk_data() -> None:
    """Download the NLTK stopwords corpus if it is not already available."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        logger.info("NLTK stopwords corpus not found; downloading…")
        nltk.download("stopwords", quiet=True)


@lru_cache(maxsize=1)
def get_english_stopwords() -> frozenset[str]:
    """Return a cached set of English stopwords."""
    ensure_nltk_data()
    return frozenset(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """Normalize a raw message for TF-IDF vectorization.

    Steps:
        1. Coerce to string and lowercase.
        2. Replace URLs with a placeholder token (smishing often includes links).
        3. Remove punctuation.
        4. Collapse whitespace.
        5. Drop English stopwords and single-character tokens.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    # Keep a stable token so URL-heavy smishing still has a learnable signal.
    text = _URL_PATTERN.sub(" urlplaceholder ", text)
    text = text.translate(_PUNCTUATION_TABLE)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()

    stop_words = get_english_stopwords()
    tokens = [token for token in text.split() if token not in stop_words and len(token) > 1]
    return " ".join(tokens)
