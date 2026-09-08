from factorstrip.v2.rw_r1000 import RwR1000Config
from factorstrip.v2.source_capabilities import ROBOT_WEALTH_R1000


def test_rw_coverage_floor_is_frozen_at_90_percent():
    cfg = RwR1000Config()
    cfg.validate()
    assert cfg.min_characteristic_coverage == 0.90
    assert cfg.characteristic_lag_days == 1


def test_rw_source_preserves_audited_strengths_and_open_terminal_issue():
    c = ROBOT_WEALTH_R1000
    assert c.includes_delisted_names
    assert c.history_start_year == 1998
    assert not c.pit_sector_industry
    assert not c.terminal_delisting_economics
