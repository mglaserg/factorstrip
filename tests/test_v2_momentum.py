from factorstrip.v2.momentum import MOM_12_1, MOM_6_1


def test_registered_momentum_specs_encode_actual_skip_month_design():
    MOM_12_1.validate()
    MOM_6_1.validate()
    assert MOM_12_1.formation_months == 11
    assert MOM_12_1.skip_months == 1
    assert MOM_6_1.formation_months == 5
    assert MOM_6_1.skip_months == 1
