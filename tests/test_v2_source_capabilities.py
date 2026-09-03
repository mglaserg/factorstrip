from factorstrip.v2.source_capabilities import NORGATE_US_PLATINUM


def test_norgate_is_not_silently_certified_for_full_sector_experiment():
    blockers = NORGATE_US_PLATINUM.blockers(require_sector_model=True)
    assert "terminal/delisting economics are unresolved" in blockers
    assert "point-in-time sector/industry classification is unresolved" in blockers


def test_norgate_known_strengths_are_preserved():
    c = NORGATE_US_PLATINUM
    assert c.stable_asset_ids
    assert c.includes_delisted_names
    assert c.pit_major_exchange_status
    assert c.raw_dollar_turnover
