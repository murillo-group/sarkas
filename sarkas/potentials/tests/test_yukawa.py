from numpy import array, isclose

from ..yukawa import yukawa_force, yukawa_force_pppm


def test_yukawa_force():
    """Test the calculation of the bare coulomb force."""
    r = 2.0
    pot_mat = array([1.0, 1.0, 0.001])

    potential, force = yukawa_force(r, pot_mat)

    assert isclose(potential, 0.06766764161830635)

    assert isclose(force, 0.10150146242745953)


def test_yukawa_force_pppm():
    """Test the calculation of the pp part of the coulomb force."""
    r = 2.0
    pot_mat = array([1.0, 0.5, 0.25, 0.001])

    potential, force = yukawa_force_pppm(r, pot_mat)

    assert isclose(potential, 0.16287410244138842)

    assert isclose(force, 0.18025091684402375)


# def test_update_params():
#     # TODO: write a test for update_params
#     pass
