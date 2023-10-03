r"""
Module for handling Moliere Potential.

Potential
*********

The Moliere Potential is defined as

.. math::
    U(r) = \frac{q_i q_j}{4 \pi \epsilon_0} \frac{1}{r} \sum_{\alpha} C_{\alpha} e^{- b_{\alpha} r}.

For more details see :cite:`Wilson1977`. Note that the parameters :math:`b` are not normalized by the Bohr radius.
They should be passed with the correct units [m] if mks or [cm] if cgs.

Force Error
***********

The force error is calculated from the Yukawa's formula with the smallest screening length.

.. math::

    \Delta F = \frac{q^2}{4 \pi \epsilon_0} \sqrt{2 \pi n b_{\textrm min} }e^{-b_{\textrm min} r_c},

This overestimates it, but it doesn't matter.

Potential Attributes
********************
The elements of the :attr:`sarkas.potentials.core.Potential.matrix` are:

.. code-block:: python

    pot_matrix[0] = q_iq_je^2/(4 pi eps_0) Force factor between two particles.
    pot_matrix[1] = C_1
    pot_matrix[2] = C_2
    pot_matrix[3] = C_3
    pot_matrix[4] = b_1
    pot_matrix[5] = b_2
    pot_matrix[6] = b_3

"""
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import array, exp, inf, pi, sqrt, zeros
from scipy.integrate import quad


def update_params(potential):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """
    potential.screening_lengths = array(potential.screening_lengths)
    potential.screening_charges = array(potential.screening_charges)
    params_len = len(potential.screening_lengths)

    potential.matrix = zeros((potential.num_species, potential.num_species, 2 * params_len + 1))

    for i, q1 in enumerate(potential.species_charges):
        for j, q2 in enumerate(potential.species_charges):
            potential.matrix[i, j, 0] = q1 * q2 / potential.fourpie0
            potential.matrix[i, j, 1 : params_len + 1] = potential.screening_charges
            potential.matrix[i, j, params_len + 1 :] = 1.0 / potential.screening_lengths

    potential.force = moliere_force
    potential.potential_derivatives = potential_derivatives

    potential.force_error = calc_force_error_quad(potential.a_ws, potential.rc, potential.matrix[0, 0])


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def moliere_force(r, pot_matrix):
    """
    Numba'd function to calculate the PP force between particles using the Moliere Potential.

    Parameters
    ----------
    r : float
        Particles' distance.

    pot_matrix : numpy.ndarray
        Moliere potential parameters. \n
        Shape = (7, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    U : float
        Potential.

    force : float
        Force between two particles.

    Examples
    --------
    >>> from scipy.constants import epsilon_0, pi, elementary_charge
    >>> from numpy import array, zeros
    >>> charge = 4.0 * elementary_charge  # = 4e [C] mks units
    >>> coul_const = 1.0/ (4.0 * pi * epsilon_0)
    >>> screening_charges = array([0.5, -0.5, 1.0])
    >>> screening_lengths = array([5.99988000e-11, 1.47732309e-11, 1.47732309e-11])  # [m]
    >>> params_len = len(screening_lengths)
    >>> pot_mat = zeros(2 * params_len + 1)
    >>> pot_mat[0] = coul_const * charge**2
    >>> pot_mat[1: params_len + 1] = screening_charges.copy()
    >>> pot_mat[params_len + 1:] = 1./screening_lengths
    >>> r = 6.629755e-10  # [m] particles distance
    >>> moliere_force(r, pot_mat)
    (4.423663010052846e-23, 6.672438139145769e-14)

    """

    u_r = 0.0
    force = 0.0

    for i in range(int(len(pot_matrix[:-1]) / 2)):
        factor1 = r * pot_matrix[i + 4]
        factor2 = pot_matrix[i + 1] / r
        u_r += factor2 * exp(-factor1)
        force += exp(-factor1) * factor2 * (1.0 / r + pot_matrix[i + 1])

    force *= pot_matrix[0]
    u_r *= pot_matrix[0]

    return u_r, force


def potential_derivatives(r, pot_matrix):
    """
    Calculate the first and second derivative of the potential.

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

    u_r = 0.0
    dv_dr = 0.0
    d2v_dr2 = 0.0

    for i in range(int(len(pot_matrix[:-1]) / 2)):
        bi = pot_matrix[i + 4]
        Ci = pot_matrix[i + 1]

        ui_r = Ci * exp(-bi * r) / r
        dui_dr = -(1.0 / r + bi) * ui_r
        d2ui_dr2 = -(1.0 / r + bi) * dui_dr + ui_r / r**2

        u_r += ui_r
        dv_dr += dui_dr
        d2v_dr2 += d2ui_dr2

    u_r *= pot_matrix[0]
    dv_dr *= pot_matrix[0]
    d2v_dr2 *= pot_matrix[0]

    return u_r, dv_dr, d2v_dr2


def force_error_integrand(r, pot_matrix):
    r"""Auxiliary function to be used in `scipy.integrate.quad` to calculate the integrand.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        Slice of the `sarkas.potentials.Potential.matrix` containing the necessary potential parameters.

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
    for i in range(int(len(pot_matrix[:-1]) / 2)):
        params[i + 4] *= a

    r_c = rc / a
    result, _ = quad(force_error_integrand, a=r_c, b=inf, args=(params,))

    f_err = sqrt(result)

    return f_err
