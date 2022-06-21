from numpy import array, isclose, linspace, pi, sqrt, zeros
from scipy.constants import elementary_charge, epsilon_0, pi

from ..maths import force_error_analytic_lcl, yukawa_green_function

# def test_fd_integral():
#     """Test the calculation of the unnormalized FD integral."""
#     # Values take from Tutorial notebooks
#     p = 0.5
#     eta = -0.2860
#     I = fd_integral(eta, p)
#
#     assert isclose(I, 0.5381222613853245)
#


def test_yukawa_gk():
    """Test the calculation of yukawa_green_function."""

    k = linspace(0, 2, 100)
    alpha = 0.2
    kappa = 0.5
    G_k = yukawa_green_function(k=k, alpha=alpha, kappa=kappa)

    assert isclose(G_k[0], 10.53621750372248)


def test_yukawa_force_analytic_lcl():

    # Look more about potential matrix
    potential_matrix = zeros((2, 2, 2))
    kappa = 2.0  # in units of a_ws
    potential_matrix[1, :, :] = kappa
    rc = 6.0  # in units of a_ws
    const = 1.0  # Rescaling const
    f = force_error_analytic_lcl("yukawa", rc, potential_matrix, const)

    assert isclose(f, 2.1780665692875655e-05)


def test_lj_force_analytic_lcl():
    """Test lennard jones force error."""

    potential_matrix = zeros((5, 2, 2))
    sigma = 3.4e-10
    pot_const = 4.0 * 1.656e-21  # 4*epsilon
    high_pow, low_pow = 12, 6
    potential_matrix[0] = pot_const
    potential_matrix[1] = sigma
    potential_matrix[2] = high_pow
    potential_matrix[3] = low_pow
    rc = 10 * sigma
    f = force_error_analytic_lcl("lj", rc, potential_matrix, 1.0)

    assert isclose(f, 1.4590050212983888e-16)


def test_moliere_force_analytic_lcl():
    """Test moliere force error."""

    charge = 4.0 * elementary_charge  # = 4e [C] mks units
    coul_const = 1.0 / (4.0 * pi * epsilon_0)
    screening_charges = array([0.5, -0.5, 1.0])
    screening_lengths = array([5.99988000e-11, 1.47732309e-11, 1.47732309e-11])  # [m]
    params_len = len(screening_lengths)
    pot_mat = zeros((2 * params_len + 1, 2, 2))
    pot_mat[0] = coul_const * charge**2
    pot_mat[1 : params_len + 1] = screening_charges.reshape((3, 1, 1))
    pot_mat[params_len + 1 :] = 1.0 / screening_lengths.reshape((3, 1, 1))
    rc = 6.629e-10

    f = force_error_analytic_lcl("moliere", rc, pot_mat, 1.0)

    assert isclose(f, 2.1223648580087958e-14)
