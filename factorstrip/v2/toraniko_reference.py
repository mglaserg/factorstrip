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
    """Thin, optional Toraniko comparison path.

    FactorStrip does not make Toraniko authoritative.  This adapter is a
    cross-check: when both engines are fed equivalent exposures, large
    unexplained disagreements should be investigated.
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
