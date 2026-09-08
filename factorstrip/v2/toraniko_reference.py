from __future__ import annotations

from typing import Any


def toraniko_available() -> bool:
    try:
        import toraniko  # noqa: F401
    except ImportError:
        return False
    return True


def estimate_toraniko_reference(
    returns_df: Any,
    market_cap_df: Any,
    sector_df: Any,
    style_df: Any,
    *,
    winsor_factor: float = 0.10,
    residualize_styles: bool = False,
):
    """Legacy thin Toraniko call retained for compatibility.

    The primary V2 RW path now lives in `toraniko_engine.py`; this function is
    kept for independent/reference calls and older notebooks.
    """

    try:
        from toraniko.model import estimate_factor_returns
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Toraniko is optional. Install the `reference` extra before running "
            "this comparison path."
        ) from exc

    return estimate_factor_returns(
        returns_df,
        market_cap_df,
        sector_df,
        style_df,
        winsor_factor=winsor_factor,
        residualize_styles=residualize_styles,
    )
