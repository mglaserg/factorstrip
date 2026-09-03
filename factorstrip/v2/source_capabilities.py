from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSourceCapabilities:
    name: str
    stable_asset_ids: bool
    includes_delisted_names: bool
    terminal_delisting_economics: bool
    pit_major_exchange_status: bool
    pit_sector_industry: bool
    raw_dollar_turnover: bool
    history_start_year: int | None
    notes: tuple[str, ...] = ()

    def blockers(self, *, require_sector_model: bool = True) -> list[str]:
        blockers: list[str] = []
        if not self.stable_asset_ids:
            blockers.append("stable asset IDs are required")
        if not self.includes_delisted_names:
            blockers.append("delisted securities are required")
        if not self.terminal_delisting_economics:
            blockers.append("terminal/delisting economics are unresolved")
        if not self.pit_major_exchange_status:
            blockers.append("point-in-time major-exchange status is required")
        if require_sector_model and not self.pit_sector_industry:
            blockers.append("point-in-time sector/industry classification is unresolved")
        return blockers


# Documented Norgate capabilities as of the V2 design review.  This object is
# intentionally conservative: unsupported/undocumented PIT fields are False.
NORGATE_US_PLATINUM = DataSourceCapabilities(
    name="Norgate US Platinum",
    stable_asset_ids=True,
    includes_delisted_names=True,
    terminal_delisting_economics=False,
    pit_major_exchange_status=True,
    pit_sector_industry=False,
    raw_dollar_turnover=True,
    history_start_year=1990,
    notes=(
        "Norgate AssetID is stable across symbol/exchange/delisting changes.",
        "US Platinum includes delisted securities and major-exchange history.",
        "Norgate states that it does not provide a delisting return.",
        "Python GICS classification calls are documented as security metadata, not a historical PIT classification series.",
    ),
)
