from factorstrip.v2.config import ResearchDesign, UniverseConfig


def test_research_design_is_frozen_at_registered_hurdles():
    design = ResearchDesign()
    design.validate()
    assert design.incremental_alpha_hurdle == 0.03
    assert design.target_vol == 0.10
    assert design.registered_trials == 2
    assert design.case_c_incremental_risk_budget == 0.25


def test_universe_has_no_index_membership_parameter():
    config = UniverseConfig()
    config.validate()
    assert "index" not in " ".join(config.__dataclass_fields__).lower()
