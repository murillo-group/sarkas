"""
Module for handling Hard-Sphere Yukawa potential.

Potential
*********

The Hard-Sphere Yukawa potential between two charges :math:`q_i` and :math:`q_j` at distant :math:`r` is defined as

.. math::
    U_{ab}(r) = \\left ( \\frac{\\sigma}{r} \\right )^{n} \\frac{q_a q_b}{4 \\pi \\epsilon_0} \\frac{e^{- \\kappa r} }{r}.

where :math:`\\kappa = 1/\\lambda` is the screening parameter.

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.pot_matrix` are:

.. code-block:: python

    pot_matrix[0] = q_iq_j^2/(4 pi eps0)
    pot_matrix[1] = 1/lambda
    pot_matrix[2] = Ewald screening parameter
    pot_matrix[3] = short range cutoff
"""

import numpy as np
from numba import njit
from sarkas.utilities.maths import force_error_analytic_pp


@njit
def hs_yukawa_force(r, pot_matrix):
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
    U_y = pot_matrix[0] * np.exp(-pot_matrix[1] * r) / r
    U_hs = (pot_matrix[2] / r) ** (50)
    U = U_y + U_hs
    force = U_y * (1.0 / r + pot_matrix[1]) + (50.0) * U_hs / r

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

    potential.matrix = np.zeros((3, params.num_species, params.num_species))
    potential.matrix[1, :, :] = 1.0 / params.lambda_TF

    for i, q1 in enumerate(params.species_charges):
        for j, q2 in enumerate(params.species_charges):
            potential.matrix[0, i, j] = q1 * q2 / params.fourpie0

    potential.matrix[2, :, :] = potential.hs_radius

    if potential.method == "pp":
        # The rescaling constant is sqrt ( na^4 ) = sqrt( 3 a/(4pi) )
        potential.force = hs_yukawa_force
        params.force_error = force_error_analytic_pp(
            "yukawa", potential.rc, potential.matrix, np.sqrt(3.0 * params.a_ws / (4.0 * np.pi))
        )
        # # Force error calculated from eq.(43) in Ref.[1]_
        # params.force_error = np.sqrt( TWOPI / params.lambda_TF) * np.exp(- potential.rc / params.lambda_TF)
        # # Renormalize
        # params.force_error *= params.a_ws ** 2 * np.sqrt(params.total_num_ptcls / params.pbox_volume)
    elif potential.method == "pppm":
        raise ValueError("PPPM algorithm not supported.")
