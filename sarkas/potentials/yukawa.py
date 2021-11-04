"""
Module for handling Yukawa potential.

Potential
*********

The Yukawa potential between two charges :math:`q_i` and :math:`q_j` at distant :math:`r` is defined as

.. math::
    U_{ab}(r) = \\frac{q_a q_b}{4 \\pi \\epsilon_0} \\frac{e^{- \\kappa r} }{r}.

where :math:`\\kappa = 1/\\lambda` is the screening parameter.

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.pot_matrix` are:

.. code-block:: python

    pot_matrix[0] = q_iq_j^2/(4 pi eps0)
    pot_matrix[1] = 1/lambda
    pot_matrix[2] = Ewald screening parameter

"""
import numpy as np
from numba import njit
import math as mt
from sarkas.utilities.maths import force_error_analytic_pp


@njit
def yukawa_force_pppm(r, pot_matrix):
    """
    Calculates Potential and Force between two particles when the pppm algorithm is chosen.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    U : float
        Potential value

    fr : float
        Force between two particles calculated using eq.(22) in :cite:`Dharuman2017`.

    """
    kappa = pot_matrix[1]
    alpha = pot_matrix[2]  # Ewald parameter alpha

    kappa_alpha = kappa / alpha
    alpha_r = alpha * r
    kappa_r = kappa * r
    U = (
        pot_matrix[0]
        * (0.5 / r)
        * (
            np.exp(kappa_r) * mt.erfc(alpha_r + 0.5 * kappa_alpha)
            + np.exp(-kappa_r) * mt.erfc(alpha_r - 0.5 * kappa_alpha)
        )
    )
    # Derivative of the exponential term and 1/r
    f1 = (0.5 / r) * np.exp(kappa * r) * mt.erfc(alpha_r + 0.5 * kappa_alpha) * (1.0 / r - kappa)
    f2 = (0.5 / r) * np.exp(-kappa * r) * mt.erfc(alpha_r - 0.5 * kappa_alpha) * (1.0 / r + kappa)
    # Derivative of erfc(a r) = 2a/sqrt(pi) e^{-a^2 r^2}* (x/r)
    f3 = (alpha / np.sqrt(np.pi) / r) * (
        np.exp(-((alpha_r + 0.5 * kappa_alpha) ** 2)) * np.exp(kappa_r)
        + np.exp(-((alpha_r - 0.5 * kappa_alpha) ** 2)) * np.exp(-kappa_r)
    )
    fr = pot_matrix[0] * (f1 + f2 + f3)

    return U, fr


@njit
def yukawa_force(r, pot_matrix):
    """
    Calculates Potential and Force between two particles.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    U : float
        Potential.

    force : float
        Force between two particles.

    """
    U = pot_matrix[0] * np.exp(-pot_matrix[1] * r) / r
    force = U * (1.0 / r + pot_matrix[1])

    return U, force


@njit
def force_deriv(r, pot_matrix):
    """Calculate the second derivative of the potential.

    Parameters
    ----------

    r : float
        Distance between particles

    pot_matrix : numpy.ndarray
        Values of the potential constants. \n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    f_dev : float
        Second derivative of potential.

    """
    kappa_r = pot_matrix[1] * r
    U2 = pot_matrix[0] * np.exp(-kappa_r) / r ** 3
    f_dev = U2 * (2.0 * (1.0 + kappa_r) + kappa_r ** 2)
    return f_dev


def update_params(potential, params):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : sarkas.potentials.core.Potential
        Class handling potential form.

    params: sarkas.core.Parameters
        Simulation's parameters.

    """

    if potential.method == "pppm":
        potential.matrix = np.zeros((3, params.num_species, params.num_species))
    else:
        potential.matrix = np.zeros((2, params.num_species, params.num_species))

    potential.matrix[1, :, :] = 1.0 / params.lambda_TF

    for i, q1 in enumerate(params.species_charges):
        for j, q2 in enumerate(params.species_charges):
            potential.matrix[0, i, j] = q1 * q2 / params.fourpie0

    if potential.method == "pp":
        # The rescaling constant is sqrt ( na^4 ) = sqrt( 3 a/(4pi) )
        potential.force = yukawa_force
        params.force_error = force_error_analytic_pp(
            potential.type, potential.rc, potential.matrix, np.sqrt(3.0 * params.a_ws / (4.0 * np.pi))
        )
        # # Force error calculated from eq.(43) in Ref.[1]_
        # params.force_error = np.sqrt( TWOPI / params.lambda_TF) * np.exp(- potential.rc / params.lambda_TF)
        # # Renormalize
        # params.force_error *= params.a_ws ** 2 * np.sqrt(params.total_num_ptcls / params.pbox_volume)
    elif potential.method == "pppm":
        potential.force = yukawa_force_pppm
        potential.matrix[2, :, :] = potential.pppm_alpha_ewald
        # PP force error calculation. Note that the equation was derived for a single component plasma.
        kappa_over_alpha = -0.25 * (potential.matrix[1, 0, 0] / potential.matrix[2, 0, 0]) ** 2
        alpha_times_rcut = -((potential.matrix[2, 0, 0] * potential.rc) ** 2)
        params.pppm_pp_err = 2.0 * np.exp(kappa_over_alpha + alpha_times_rcut) / np.sqrt(potential.rc)
        params.pppm_pp_err *= np.sqrt(params.total_num_ptcls) * params.a_ws ** 2 / np.sqrt(params.pbox_volume)
