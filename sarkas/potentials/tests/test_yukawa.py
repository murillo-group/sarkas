from numpy import array, isclose

from ..yukawa import yukawa_force, yukawa_force_pppm

from pytest import mark

@mark.parametrize("r,pot_mat,expected_potential,expected_force",[
    (2.0,array([1.0, 1.0, 0.001]),0.06766764161830635,0.10150146242745953)
    ],ids=["r-2;pot_mat-[1.0,1.0,0.001]"])
def test_yukawa_force(r,pot_mat,expected_potential,expected_force):
    """Test the calculation of the bare coulomb force."""
    potential, force = yukawa_force(r, pot_mat)

    assert isclose(potential, expected_potential)
    assert isclose(force, expected_force)

@mark.parametrize("r,pot_mat,expected_potential,expected_force",[
    (2.0,array([1.0, 0.5, 0.25, 0.001]),0.16287410244138842,0.18025091684402375)
    ],ids=["r-2;pot_mat-[1.0,0.5,0.25,0.001]"])
def test_yukawa_force_pppm(r,pot_mat,expected_potential,expected_force):
    """Test the calculation of the pp part of the coulomb force."""
    potential, force = yukawa_force_pppm(r, pot_mat)

    assert isclose(potential, expected_potential)
    assert isclose(force, expected_force)


# def test_update_params():
#     # TODO: write a test for update_params
#     pass
