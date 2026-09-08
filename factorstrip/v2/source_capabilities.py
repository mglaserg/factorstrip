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


# RW R1000 was audited locally before the clean inferential run.  PIT membership,
# former constituents, adjusted daily OHLCV, and PIT market-cap/value fields are
# present with strong daily coverage.  We remain conservative on fields the
# archive does not explicitly document: ticker strings are not a vendor-stable
# permanent identifier and terminal/delisting return treatment is not separately
# exposed in the audited files.
ROBOT_WEALTH_R1000 = DataSourceCapabilities(
    name="Robot Wealth R1000 PIT archive",
    stable_asset_ids=False,
    includes_delisted_names=True,
    terminal_delisting_economics=False,
    pit_major_exchange_status=False,
    pit_sector_industry=False,
    raw_dollar_turnover=True,
    history_start_year=1998,
    notes=(
        "Audited OHLC: 9,438,170 rows, 2,908 historical tickers, 1998-01-02 through 2021-04-01.",
        "Audited PIT membership retained 1,674 former constituents.",
        "Audited fundamentals: daily market cap and price-to-book with ~98.9% median and ~95.1% fifth-percentile active coverage.",
        "Undated metadata is descriptive only and is not used as PIT sector exposure in the primary model.",
        "RW ticker keys include historical suffixes but are not documented here as permanent security identifiers.",
        "Terminal/delisting economics are not separately exposed by the audited R1000 files and must be resolved before final inference.",
    ),
)
