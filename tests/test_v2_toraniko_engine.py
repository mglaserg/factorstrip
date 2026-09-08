from factorstrip.v2.toraniko_engine import ToranikoEngineConfig


def test_primary_toraniko_engine_is_market_size_value_without_momentum():
    cfg = ToranikoEngineConfig()
    cfg.validate()
    assert cfg.winsor_factor == 0.01
    assert cfg.residualize_styles is True
