"""JSONL feedback store for verdict corrections. No database required."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

FEEDBACK_PATH = Path(__file__).resolve().parent / "data" / "feedback.jsonl"
_LOCK = Lock()


def _read_all() -> list[dict[str, Any]]:
    if not FEEDBACK_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def add_feedback(
    *,
    text: str,
    predicted: str,
    actual: str,
    correct: bool,
) -> dict[str, Any]:
    record = {
        "text": text,
        "predicted": predicted,
        "actual": actual,
        "correct": correct,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _LOCK:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def list_feedback(limit: int = 50) -> list[dict[str, Any]]:
    records = _read_all()
    return list(reversed(records[-limit:]))


def feedback_summary() -> dict[str, Any]:
    records = _read_all()
    total = len(records)
    agreed = sum(1 for item in records if item.get("correct"))
    return {
        "count": total,
        "agreed": agreed,
        "disagreed": total - agreed,
        "agreement_rate": round(agreed / total, 4) if total else None,
        "recent": list_feedback(12),
    }
