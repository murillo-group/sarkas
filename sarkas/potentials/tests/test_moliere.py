from numpy import array, isclose, zeros
from scipy.constants import elementary_charge, epsilon_0, pi

from ..moliere import moliere_force


def test_moliere_force():
    """Test the calculation of the moliere force and potential."""

    charge = 4.0 * elementary_charge  # = 4e [C] mks units
    coul_const = 1.0 / (4.0 * pi * epsilon_0)

    screening_charges = array([0.5, -0.5, 1.0])
    screening_lengths = array([5.99988000e-11, 1.47732309e-11, 1.47732309e-11])  # [m]
    params_len = len(screening_lengths)

    pot_mat = zeros(2 * params_len + 1)
    pot_mat[0] = coul_const * charge**2
    pot_mat[1 : params_len + 1] = screening_charges.copy()
    pot_mat[params_len + 1 :] = 1.0 / screening_lengths

    r = 6.629755e-10  # [m] particles distance
    potential, force = moliere_force(r, pot_mat)

    assert isclose(potential, 4.423663010052846e-23)

    assert isclose(force, 6.672438139145769e-14)


# def test_update_params():
#     # TODO: write a test for update_params
#     pass
