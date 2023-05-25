r"""
Module for handling custom potential of the form given below.

Average Atom Fit Potential
**************************

The form of the potential is

.. math::
    U(r) = \frac{q_i q_j}{4 \pi \epsilon_0} \frac{e^{- \kappa r} }{r} \frac{1}{a + b e^{c (r - d)}},

where :math:`\kappa, a, b, c, d` are fit parameters to be passed. Remember to use the correct units.

Force Error
***********

The force error is calculated from the ratio

.. math::

    \Delta F = \frac{F(r_c)}{F(2a_{ws})},

where :math:`F(x)` is the force between two particles at distance :math:`x`, and :math:`a_{ws}` is the Wigner-Seitz radius.

Potential Attributes
********************
The elements of the :attr:`sarkas.potentials.core.Potential.matrix` are:

.. code-block:: python

    pot_matrix[0] = q_iq_je^2/(4 pi eps_0) Force factor between two particles.
    pot_matrix[1] = kappa
    pot_matrix[2] = a
    pot_matrix[3] = b
    pot_matrix[4] = c
    pot_matrix[5] = d
    pot_matrix[6] = a_rs. Short-range cutoff.

"""
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import array, exp, inf, pi, sqrt, zeros
from scipy.integrate import quad


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def fit_force(r, pot_matrix):
    """
    Numba'd function to calculate the PP force between particles using the Moliere Potential.

    Parameters
    ----------
    r : float
        Particles' distance.

    pot_matrix : numpy.ndarray
        Slice of `sarkas.potentials.Potential.matrix` containing the potential parameters.

    Returns
    -------
    u_r : float
        Potential.

    f_r : float
        Force between two particles.

    """

    # Unpack the parameters
    q2_e0 = pot_matrix[0]
    kappa = pot_matrix[1]
    a = pot_matrix[2]
    b = pot_matrix[3]
    c = pot_matrix[4]
    d = pot_matrix[5]

    yukawa = exp(-kappa * r) / r
    denom = a + b * exp(c * r - c * d)

    u_r = yukawa / denom

    # d/dr 1/r
    f1 = u_r / r
    # d/dr exp(-kappa r)
    f2 = kappa * u_r
    # d/dr denom
    f3 = u_r / denom * (b * c * exp(c * r - c * d))

    force = f1 + f2 + f3
    force *= q2_e0
    u_r *= q2_e0

    return u_r, force


def potential_derivatives(r, pot_matrix):
    """Calculate the first and second derivative of the potential.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables.

    Returns
    -------
    u_r : float, numpy.ndarray
        Potential value.

    dv_dr : float, numpy.ndarray
        First derivative of the potential.

    d2v_dr2 : float, numpy.ndarray
        Second derivative of the potential.

    """

    # Unpack the parameters
    q2_e0 = pot_matrix[0]
    kappa = pot_matrix[1]
    a = pot_matrix[2]
    b = pot_matrix[3]
    c = pot_matrix[4]
    d = pot_matrix[5]

    yukawa = exp(-kappa * r) / r
    denom = a + b * exp(c * r - c * d)

    u_r = yukawa / denom

    # d/dr 1/r
    f1 = -u_r / r
    # d/dr exp(-kappa r)
    f2 = -kappa * u_r
    # d/dr denom
    f3 = -u_r / denom * (b * c * exp(c * r - c * d))

    dv_dr = q2_e0 * (f1 + f2 + f3)

    # d/dr f1
    h1 = 2.0 * f1 / r + f1 / denom * (b * c * exp(c * r - c * d)) + kappa * f1
    # d/dr f2
    h2 = kappa * f2 + f2 / r + f2 / denom * (b * c * exp(c * r - c * d))
    # d/dr f3
    h3 = f3 / r - f3 * (c - kappa) + 2 * f3 / denom * (b * c * exp(c * r - c * d))

    d2v_dr2 = q2_e0 * (h1 + h2 + h3)

    u_r *= q2_e0

    return u_r, dv_dr, d2v_dr2


def pretty_print_info(potential):
    """
    Print potential specific parameters in a user-friendly way.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """
    msg = (
        f"screening type : {potential.screening_length_type}\n"
        f"screening length = {potential.screening_length:.6e} {potential.units_dict['length']}\n"
        f"kappa = {potential.a_ws / potential.screening_length:.4f}\n"
        f"Fit params:\n"
        f"kappa = {potential.matrix[1, 0,0]*potential.a_ws:.3f} / a_ws = {potential.matrix[1, 0,0]:.6e} {potential.units_dict['inverse length']}\n"
        f"a = {potential.matrix[2,0,0]}\n"
        f"b = {potential.matrix[3,0,0]}\n"
        f"c = {potential.matrix[4,0,0] * potential.a_ws:.3f} /a_ws => {potential.matrix[4, 0,0]:.6e} {potential.units_dict['inverse length']}\n"
        f"d = {potential.matrix[5,0,0] / potential.a_ws:.3f} a_ws => {potential.matrix[4, 0,0]:.6e} {potential.units_dict['length']}\n"
        f"Gamma_eff = {potential.coupling_constant:.2f}"
    )
    print(msg)


def update_params(potential):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """
    potential.fit_params = array(potential.fit_params)
    params_len = len(potential.fit_params)

    potential.matrix = zeros((params_len + 2, potential.num_species, potential.num_species))

    for i, q1 in enumerate(potential.species_charges):
        for j, q2 in enumerate(potential.species_charges):
            potential.matrix[0, i, j] = q1 * q2 / potential.fourpie0
            potential.matrix[1 : params_len + 1, i, j] = potential.fit_params
            potential.matrix[-1, i, j] = potential.a_rs

    potential.force = fit_force
    # _, f_rc = fit_force(potential.rc, potential.matrix[:, 0, 0])
    # _, f_2a = fit_force(2.0 * potential.a_ws, potential.matrix[:, 0, 0])
    # potential.force_error = f_rc / f_2a
    potential.potential_derivatives = potential_derivatives
    potential.force_error = calc_force_error_quad(potential.a_ws, potential.rc, potential.matrix[:, 0, 0])


def force_error_integrand(r, pot_matrix):
    r"""Auxiliary function to be used in `scipy.integrate.quad` to calculate the integrand.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        Slice of the `sarkas.potentials.core.Potential.matrix` containing the necessary potential parameters.

    Returns
    -------
    _ : float
        Integrand :math:`4\pi r^2 ( d r\phi(r)/dr )^2`

    """

    _, dv_dr, _ = potential_derivatives(r, pot_matrix)

    return 4.0 * pi * r**2 * dv_dr**2


def calc_force_error_quad(a, rc, pot_matrix):
    r"""
    Calculate the force error by integrating the square modulus of the force over the neglected volume.\n
    The force error is calculated from

    .. math::
        \Delta F =  \left [ 4 \pi \int_{r_c}^{\infty} dr \, r^2  \left ( \frac{d\phi(r)}{r} \right )^2 ]^{1/2}

    where :math:`\phi(r)` is only the radial part of the potential, :math:`r_c` is the cutoff radius, and :math:`r` is scaled by the input parameter `a`.\n
    The integral is calculated using `scipy.integrate.quad`. The derivative of the potential is obtained from :meth:`potential_derivatives`.

    Parameters
    ----------
    a : float
        Rescaling length. Usually it is the Wigner-Seitz radius.

    rc : float
        Cutoff radius to be used as the lower limit of the integral. The lower limit is actually `rc /a`.

    pot_matrix: numpy.ndarray
        Slice of the `sarkas.potentials.Potential.matrix` containing the parameters of the potential. It must be a 1D-array.

    Returns
    -------
    f_err: float
        Force error. It is the sqrt root of the integral. It is calculated using `scipy.integrate.quad`  and :func:`potential_derivatives`.

    """

    params = pot_matrix.copy()
    params[0] = 1
    # Un-dimensionalize the screening length.
    params[1] *= a
    params[4] *= a
    params[5] /= a

    r_c = rc / a

    result, _ = quad(force_error_integrand, a=r_c, b=inf, args=(params,))

    f_err = sqrt(result)

    return f_err
