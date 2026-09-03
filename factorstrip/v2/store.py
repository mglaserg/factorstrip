from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .schema import BARS_REQUIRED, SCHEMA_VERSION, SECURITY_HISTORY_REQUIRED


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    source: str
    as_of_utc: str
    bars_sha256: str
    security_history_sha256: str
    notes: str = ""


class CanonicalStore:
    """Versioned Parquet boundary between vendors and FactorStrip V2.

    Polars is imported lazily so legacy V1 remains usable until the V2 extras
    are installed.  The canonical store deliberately prevents strategy code
    from becoming coupled to Norgate/CRSP/Sharadar-specific APIs.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bars_path = self.root / "bars.parquet"
        self.security_history_path = self.root / "security_history.parquet"
        self.manifest_path = self.root / "manifest.json"

    @staticmethod
    def _pl():
        try:
            import polars as pl
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "FactorStrip V2 requires Polars. Run `uv sync` after applying "
                "the V2 patch."
            ) from exc
        return pl

    @staticmethod
    def _require_columns(columns: Iterable[str], required: set[str], label: str) -> None:
        missing = required - set(columns)
        if missing:
            raise ValueError(f"{label} is missing canonical columns: {sorted(missing)}")

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def write(self, bars: Any, security_history: Any, *, source: str, notes: str = "") -> DatasetManifest:
        pl = self._pl()
        if not isinstance(bars, pl.DataFrame) or not isinstance(security_history, pl.DataFrame):
            raise TypeError("bars and security_history must be Polars DataFrames")

        self._require_columns(bars.columns, BARS_REQUIRED, "bars")
        self._require_columns(security_history.columns, SECURITY_HISTORY_REQUIRED, "security_history")

        bars = bars.sort(["date", "asset_id"])
        security_history = security_history.sort(["date", "asset_id"])

        duplicate_bars = bars.group_by(["date", "asset_id"]).len().filter(pl.col("len") > 1)
        if duplicate_bars.height:
            raise ValueError("bars contains duplicate (date, asset_id) rows")
        duplicate_meta = security_history.group_by(["date", "asset_id"]).len().filter(pl.col("len") > 1)
        if duplicate_meta.height:
            raise ValueError("security_history contains duplicate (date, asset_id) rows")

        bars.write_parquet(self.bars_path, compression="zstd")
        security_history.write_parquet(self.security_history_path, compression="zstd")

        manifest = DatasetManifest(
            schema_version=SCHEMA_VERSION,
            source=source,
            as_of_utc=datetime.now(timezone.utc).isoformat(),
            bars_sha256=self._sha256(self.bars_path),
            security_history_sha256=self._sha256(self.security_history_path),
            notes=notes,
        )
        self.manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
        return manifest

    def read_bars(self):
        return self._pl().read_parquet(self.bars_path)

    def read_security_history(self):
        return self._pl().read_parquet(self.security_history_path)

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))
