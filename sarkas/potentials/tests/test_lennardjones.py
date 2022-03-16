from numpy import array, isclose

from ..lennardjones import lj_force


def test_lj_force():
    """Test the calculation of the lj force and potential."""

    pot_const = 4.0 * 1.656e-21  # 4*epsilon
    sigma = 3.4e-10
    high_pow, low_pow = 12.0, 6.0
    short_cutoff = 0.0001 * sigma
    pot_mat = array([pot_const, sigma, high_pow, low_pow, short_cutoff])
    r = 15.0 * sigma  # particles distance
    potential, force = lj_force(r, pot_mat)

    assert isclose(potential, -5.815308131440668e-28)

    assert isclose(force, -6.841538377536503e-19)


# def test_update_params():
#     # TODO: write a test for update_params
#     pass
