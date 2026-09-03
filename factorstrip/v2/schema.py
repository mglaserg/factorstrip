from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = "factorstrip-v2-canonical-1"


@dataclass(frozen=True)
class CanonicalColumns:
    """Source-neutral canonical field names.

    `asset_id` must be a vendor-stable identifier that survives ticker changes.
    `total_return` is a one-period decimal return and must include distributions
    and terminal/delisting economics to the extent supplied by the source.
    """

    date: str = "date"
    asset_id: str = "asset_id"
    symbol: str = "symbol"
    close_unadjusted: str = "close_unadjusted"
    volume_unadjusted: str = "volume_unadjusted"
    total_return: str = "total_return"
    is_common_stock: str = "is_common_stock"
    is_major_exchange: str = "is_major_exchange"
    sector: str = "sector"
    industry: str = "industry"


COL = CanonicalColumns()

BARS_REQUIRED = {
    COL.date,
    COL.asset_id,
    COL.symbol,
    COL.close_unadjusted,
    COL.volume_unadjusted,
    COL.total_return,
}

SECURITY_HISTORY_REQUIRED = {
    COL.date,
    COL.asset_id,
    COL.symbol,
    COL.is_common_stock,
    COL.is_major_exchange,
}
