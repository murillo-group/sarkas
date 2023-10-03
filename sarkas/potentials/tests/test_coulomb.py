from numpy import array, isclose, pi

from ..coulomb import coulomb_force, coulomb_force_pppm

from pytest import mark

@mark.parametrize("r,pot_mat,expected_potential,expected_force",[
    (2.0,array([1.0, 0.0, 0.001]),0.5,0.25)
    ],ids=["r-2;pot_mat-[1.0,0.0,0.001]"])
def test_coulomb_force(r,pot_mat,expected_potential,expected_force):
    """Test the calculation of the bare coulomb force."""
    potential, force = coulomb_force(r, pot_mat)

    assert isclose(potential, expected_potential)
    assert isclose(force, expected_force)

@mark.parametrize("r,pot_mat,expected_potential,expected_force",[
    (2.0,array([1.0, 0.5, 0.001]),0.07864960352514257,0.14310167611771996)
    ],ids=["r-2;pot_mat-[1.0,0.0,0.001]"])
def test_coulomb_force_pppm(r,pot_mat,expected_potential,expected_force):
    """Test the calculation of the pp part of the coulomb force."""
    potential, force = coulomb_force_pppm(r, pot_mat)

    assert isclose(potential, expected_potential)
    assert isclose(force, expected_force)


# def test_update_params():
#     # TODO: write a test for update_params
#     pass
