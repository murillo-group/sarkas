r"""
Module for handling custom potential of the form given below.

Average Atom Fit Potential
**************************

The form of the potential is

.. math::
    U(r) = \frac{q_i q_j}{4 \pi \epsilon_0} a \frac{e^{- \kappa r} }{r} \frac{1}{a + b e^{c (r - d)}} + h \cos\left ( (r-i) j e^{-kr} \right ) e^{-l r},

where :math:`\kappa, a, b, c, d, h, i, j, k, l` are fit parameters to be passed. Remember to use the correct units.

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
from numpy import array, cos, exp, inf, pi, sin, sqrt, zeros
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
    a = pot_matrix[0]
    b = pot_matrix[1]
    c = pot_matrix[2]
    d = pot_matrix[3]
    e = pot_matrix[4]
    f = pot_matrix[5]
    g = pot_matrix[6]
    h = pot_matrix[7]
    i = pot_matrix[8]
    j = pot_matrix[9]
    k = pot_matrix[10]
    l = pot_matrix[11]

    yukawa = a * exp(-b * r) / r
    dyuk_dr = -(1.0 / r + b) * yukawa

    sigmoid = 1.0 / (1.0 + exp(c * (r - d)))
    dsig_dr = c * sigmoid**2

    angle = (r - f) * g * exp(-h * r)
    dangle_dr = -h * angle + g * exp(-h * r)
    cos_term = e * cos(angle) * exp(-i * r)

    arg = -((k - r) ** 2) / l
    gaussian_term = j * exp(arg)

    u_r = yukawa * sigmoid + cos_term + gaussian_term

    # derivative of the yukawa term
    f1 = -dyuk_dr * sigmoid - yukawa * dsig_dr

    # derivative of the cos term
    f2 = e * sin(angle) * (dangle_dr) * exp(-i * r) + i * cos_term

    # derivative of the exp term
    f3 = -2.0 * (k - r) / l * gaussian_term

    force = f1 + f2 + f3

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
    a = pot_matrix[0]
    b = pot_matrix[1]
    c = pot_matrix[2]
    d = pot_matrix[3]
    e = pot_matrix[4]
    f = pot_matrix[5]
    g = pot_matrix[6]
    h = pot_matrix[7]
    i = pot_matrix[8]
    j = pot_matrix[9]
    k = pot_matrix[10]
    l = pot_matrix[11]

    yukawa = a * exp(-b * r) / r
    dyuk_dr = -(1.0 / r + b) * yukawa
    d2yuk_dr2 = (1.0 / r**2) * yukawa + dyuk_dr * dyuk_dr

    sigmoid = 1.0 / (1.0 + exp(c * (r - d)))
    dsig_dr = c * sigmoid**2
    d2sig_dr2 = 2.0 * c * sigmoid * dsig_dr

    angle = (r - f) * g * exp(-h * r)
    dangle_dr = -h * angle + g * exp(-h * r)
    d2angle_dr2 = -h * dangle_dr - h * g * exp(-h * r)
    cos_term = e * cos(angle) * exp(-i * r)

    arg = -((k - r) ** 2) / l
    gaussian_term = j * exp(arg)

    u_r = yukawa * sigmoid + cos_term + gaussian_term

    # derivative of the yukawa term
    f1 = dyuk_dr * sigmoid + yukawa * dsig_dr

    # derivative of the cos term
    f2 = -e * sin(angle) * dangle_dr * exp(-i * r) - i * cos_term

    # derivative of the exp term
    f3 = 2.0 * (k - r) / l * gaussian_term

    dv_dr = f1 + f2 + f3

    # Derivative of f1
    v1 = d2yuk_dr2 * sigmoid + dyuk_dr * dsig_dr + dyuk_dr * dsig_dr + yukawa * d2sig_dr2

    # derivative of f2
    v2 = -e * (cos(angle) * dangle_dr**2 + sin(angle) * (d2angle_dr2 - i * dangle_dr)) * exp(-i * r) - i * f2

    # derivative of f3
    v3 = -2.0 / l * gaussian_term + (2.0 * (k - r) / l) * f3

    d2v_dr2 = v1 + v2 + v3

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
        f"Gamma_eff = {potential.coupling_constant:.2f}\n"
        f"Fit params:\n"
        f"a = {potential.fit_params[0]:6e} beta/a_ws = {potential.matrix[0, 0, 0]:.6e} {potential.units_dict['energy']}\n"
        f"b = {potential.fit_params[1]:.6e} / a_ws = {potential.matrix[1,0,0]:.6e} {potential.units_dict['inverse length']}\n"
        f"c = {potential.fit_params[2]:.6e} / a_ws = {potential.matrix[2,0,0]:.6e} {potential.units_dict['inverse length']}\n"
        f"d = {potential.fit_params[3]:.6e} a_ws = {potential.matrix[3,0,0]:.6e} {potential.units_dict['length']}\n"
        f"e = {potential.fit_params[4]:.6e} beta = {potential.matrix[4,0,0]:.6e} {potential.units_dict['energy']}\n"
        f"f = {potential.fit_params[5]:.6e} a_ws = {potential.matrix[5,0,0]:.6e} {potential.units_dict['length']}\n"
        f"g = {potential.fit_params[6]:.6e} / a_ws = {potential.matrix[6,0,0]:.6e} {potential.units_dict['inverse length']}\n"
        f"h = {potential.fit_params[7]:.6e} / a_ws = {potential.matrix[7,0,0]:.6e} {potential.units_dict['inverse length']}\n"
        f"i = {potential.fit_params[8]:.6e} / a_ws = {potential.matrix[8,0,0]:.6e} {potential.units_dict['inverse length']}\n"
        f"j = {potential.fit_params[9]:.6e} beta = {potential.matrix[9,0,0]:.6e} {potential.units_dict['energy']}\n"
        f"k = {potential.fit_params[10]:.6e} a_ws = {potential.matrix[10,0,0]:.6e} {potential.units_dict['length']}\n"
        f"l = {potential.fit_params[11]:.6e} a_ws^2 = {potential.matrix[11,0,0]:.6e} {potential.units_dict['length']}"
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

    potential.matrix = zeros((params_len + 1, potential.num_species, potential.num_species))
    beta = 1.0 / (potential.kB * potential.electron_temperature)
    for i, q1 in enumerate(potential.species_charges):
        for j, q2 in enumerate(potential.species_charges):
            potential.matrix[0, i, j] = potential.fit_params[0] * potential.a_ws / beta  # a
            potential.matrix[1, i, j] = potential.fit_params[1] / potential.a_ws  # b
            potential.matrix[2, i, j] = potential.fit_params[2] / potential.a_ws  # c
            potential.matrix[3, i, j] = potential.fit_params[3] * potential.a_ws  # d
            potential.matrix[4, i, j] = potential.fit_params[4] / beta  # e
            potential.matrix[5, i, j] = potential.fit_params[5] * potential.a_ws  # f
            potential.matrix[6, i, j] = potential.fit_params[6] / potential.a_ws  # g
            potential.matrix[7, i, j] = potential.fit_params[7] / potential.a_ws  # h
            potential.matrix[8, i, j] = potential.fit_params[8] / potential.a_ws  # i
            potential.matrix[9, i, j] = potential.fit_params[9] / beta  # j
            potential.matrix[10, i, j] = potential.fit_params[10] * potential.a_ws  # k
            potential.matrix[11, i, j] = potential.fit_params[11] * potential.a_ws**2  # l
            potential.matrix[12, i, j] = potential.a_rs
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
    params[0] /= a  # a
    params[1] *= a  # b
    params[2] *= a  # c
    params[3] /= a  # d
    params[5] /= a  # f
    params[6] *= a  # g
    params[7] *= a  # h
    params[8] *= a  # i
    params[10] /= a  # k
    params[11] /= a**2  # l

    r_c = rc / a

    result, _ = quad(force_error_integrand, a=r_c, b=inf, args=(params,))

    f_err = sqrt(result)

    return f_err
