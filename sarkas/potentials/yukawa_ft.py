r"""
Module for handling Yukawa-Friedel tail potential.

Potential
*********

The Yukawa potential between two charges :math:`q_i` and :math:`q_j` at distant :math:`r` is defined as

.. math::
    U_{YFT}(r) = A_Y \frac{e^{- \kappa_Y r} }{r} + B_F \frac{e^{- \kappa_F r} }{r}\cos(Q_F r + \phi_F)

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.matrix` are:

.. code-block:: python

    pot_matrix[0] = A_Y
    pot_matrix[1] = kappa_Y
    pot_matrix[2] = B_F
    pot_matrix[3] = kappa_F
    pot_matrix[4] = Q_F
    pot_matrix[5] = phi_F
    pot_matrix[6] = a_rs

"""
from math import erfc
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import exp, inf, pi, sqrt, zeros, cos, sin
from scipy.integrate import quad
from warnings import warn

from ..utilities.maths import force_error_analytic_lcl, force_error_analytic_pp

__all__ = ["yukawa_force", "potential_derivatives", "pretty_print_info", "update_params", "calc_force_error_quad"]

@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def yukawa_ft_force(r_in, pot_matrix):
    """
    Numba'd function to calculate Potential and Force between two particles.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. Shape = 7


    Returns
    -------
    u_r : float
        Potential.

    f_r : float
        Force between two particles.

    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([ 1.0, 1.0, 0.0001])
    >>> yukawa_ft_force(r, pot_matrix)
    (0.06766764161830635, 0.10150146242745953)

    """
    # Short-range cutoff to deal with divergence of the Coulomb potential
    rs = pot_matrix[-1]
    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    u_y = pot_matrix[0] * exp(-pot_matrix[1] * r) / r
    f_y = u_y * (1.0 / r + pot_matrix[1])

    u_ft_exp = pot_matrix[2] * exp(-pot_matrix[3] * r) / r**3
    u_ft = u_ft_exp * cos(pot_matrix[4] * r + pot_matrix[5])

    f_ft =  u_ft * (3.0/r + pot_matrix[3]) 
    f_ft += pot_matrix[4] * u_ft_exp * sin(pot_matrix[4] * r + pot_matrix[5]) 

    u_r = u_y + u_ft
    f_r = f_y + f_ft

    return u_r, f_r


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
    U : float, numpy.ndarray
        Potential value.

    dv_dr : float, numpy.ndarray
        First derivative of the potential.

    d2v_dr2 : float, numpy.ndarray
        Second derivative of the potential.

    """

    u_y = pot_matrix[0] * exp(-pot_matrix[1] * r) / r
    f_y = - u_y * (1.0 / r + pot_matrix[1])

    u_ft_exp = pot_matrix[2] * exp(-pot_matrix[3] * r) / r**3
    u_ft = u_ft_exp * cos(pot_matrix[4] * r + pot_matrix[5])

    f_ft =  -u_ft * (3.0/r + pot_matrix[3]) 
    f_ft += -pot_matrix[4] * u_ft_exp * sin(pot_matrix[4] * r + pot_matrix[5]) 

    u_r = u_y + u_ft
    dv_dr = f_y + f_ft
    
    d2v_dr2 = 0.0

    return u_r, dv_dr, d2v_dr2


def pretty_print_info(potential):
    """
    Print potential specific parameters in a user-friendly way.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """
    kTa = potential.kB * potential.T_desired * potential.a_ws
    msg = (
        f"screening type : {potential.screening_length_type}\n"
        f"screening length = {potential.screening_length:.6e} {potential.units_dict['length']}\n"
        f"a_ws/screening length = {potential.a_ws / potential.screening_length:.4f}\n"
        f"A = {potential.matrix[0,0,0]/kTa:.4f} beta/a_ws = {potential.matrix[0,0,0]:.6e} {potential.units_dict['energy']}\n"
        f"kappa_y = {potential.a_ws  * potential.matrix[0,0,1]:.4f} / a_ws  = {potential.matrix[0,0,1]:.6e} {potential.units_dict['inverse length']}\n"
        f"B = {potential.matrix[0,0,2]/kTa/potential.a_ws**2:.4f} beta/a_ws^3 = {potential.matrix[0,0,2]:.6e} {potential.units_dict['energy']}\n"
        f"kappa_ft = {potential.a_ws *  potential.matrix[0,0,3]:.4f} / a_ws = {potential.matrix[0,0,3]:.6e} {potential.units_dict['inverse length']}\n"
        f"q_ft = {potential.a_ws * potential.matrix[0,0,4]:.4f} / a_ws = {potential.matrix[0,0,4]:.6e} {potential.units_dict['inverse length']}\n"
        f"phase = {potential.matrix[0,0,5]:.6e} [rad]\n"
        f"Gamma_eff = {potential.coupling_constant:.2f}"
    )
    print(msg)


def update_params(potential, species):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """

    potential.matrix = zeros((potential.num_species, potential.num_species, 7))

    # potential.matrix[:, :, 1] = 1.0 / potential.screening_length

    # potential.matrix[:, :, 0] = potential.species_charges.reshape((len(potential.species_charge), 1))
    # * potential.species_charges / potential.fourpie0
    # the above line is the Python version of the for loops below. I believe that the for loops are easier to understand
    for i, _ in enumerate(potential.species_charges):
        for j, _ in enumerate(potential.species_charges):
            potential.matrix[i, j, :-1] = potential.ft_params

    potential.matrix[:, :, -1] = potential.a_rs

    potential.potential_derivatives = potential_derivatives

    assert potential.method == "pp" , "The Yukawa-Friedel tail potential is implemented only for PP method."

    # The rescaling constant is sqrt ( na^4 ) = sqrt( 3 a/(4pi) )
    potential.force = yukawa_ft_force

    potential.force_error = calc_force_error_quad(potential.a_ws, potential.rc, potential.matrix[0, 0])

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

    Examples
    --------
    >>> import numpy as np
    >>> potential_matrix = np.zeros(2)
    >>> a = 1.0 # Wigner-seitz radius
    >>> kappa = 2.0 # in units of a_ws
    >>> potential_matrix[1] = kappa
    >>> rc = 6.0 # in units of a_ws
    >>> calc_force_error_quad(a, rc, potential_matrix)
    6.636507826720378e-06

    """

    params = pot_matrix.copy()

    params[2] /= params[0]
    params[0] = 1
    
    # Un-dimensionalize the screening length.
    params[1] *= a
    params[3] *= a**3
    params[4] *= a
    
    r_c = rc / a
    result, _ = quad(force_error_integrand, a=r_c, b=inf, args=(params,))

    f_err = sqrt(result)

    return f_err
