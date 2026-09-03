from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def hash_pandas_frame(frame: pd.DataFrame) -> str:
    """Stable hash for a sorted reference output used by golden tests."""

    normalized = frame.sort_index().sort_index(axis=1)
    values = pd.util.hash_pandas_object(normalized, index=True).to_numpy().tobytes()
    columns = "\x1f".join(map(str, normalized.columns)).encode("utf-8")
    return hashlib.sha256(columns + values).hexdigest()


def write_golden_snapshot(path: str | Path, *, name: str, data_hash: str, signal_hash: str, notes: str = "") -> None:
    payload = {
        "name": name,
        "data_hash": data_hash,
        "signal_hash": signal_hash,
        "notes": notes,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
