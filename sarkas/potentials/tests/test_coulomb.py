from numpy import isclose, array, pi
from ..coulomb import coulomb_force, coulomb_force_pppm


def test_coulomb_force():
    """Test the calculation of the bare coulomb force."""
    r = 2.0
    pot_mat = array([1.0, 0.0, 0.001])

    potential, force = coulomb_force(r, pot_mat)

    assert isclose(potential, 0.5)

    assert isclose(force, 0.25)


def test_coulomb_force_pppm():
    """Test the calculation of the pp part of the coulomb force."""
    r = 2.0
    pot_mat = array([1.0, 0.5, 0.001])

    potential, force = coulomb_force_pppm(r, pot_mat)

    assert isclose(potential, 0.07864960352514257)

    assert isclose(force, 0.14310167611771996)


# def test_update_params():
#     # TODO: write a test for update_params
#     pass