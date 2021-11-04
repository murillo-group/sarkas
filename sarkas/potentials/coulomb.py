"""
Module for handling Coulomb interaction.

Potential
*********

The Coulomb potential between two particles :math:`a,b` is

.. math::
   U_{ab}(r) = \\frac{q_{a}q_b}{4 \\pi \\epsilon_0 r}.

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.pot_matrix` are:

.. code-block::

    pot_matrix[0] : qi qj/(4pi esp0) Force factor between two particles.
    pot_matrix[1] : Ewald parameter in the case of pppm Algorithm. Same value for all species.
    pot_matrix[2] : Short-range cutoff. Same value for all species.

"""

import numpy as np
from numba import njit
import math as mt


def update_params(potential, params):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential: sarkas.potentials.core.Potential
        Potential's information

    params: sarkas.core.Parameters
        Simulation's parameters


    """

    potential.matrix = np.zeros((3, params.num_species, params.num_species))

    for i, q1 in enumerate(params.species_charges):
        for j, q2 in enumerate(params.species_charges):
            potential.matrix[0, i, j] = q1 * q2 / params.fourpie0

    if potential.method == "PP":
        potential.matrix[2, :, :] = potential.a_rs
        potential.force = coulomb_force
        params.force_error = 0.0  # TODO: Implement force error in PP case
    elif potential.method == "pppm":
        potential.matrix[1, :, :] = potential.pppm_alpha_ewald
        potential.matrix[2, :, :] = potential.a_rs
        # Calculate the (total) plasma frequency
        potential.force = coulomb_force_pppm

        # PP force error calculation. Note that the equation was derived for a single component plasma.
        alpha_times_rcut = -((potential.pppm_alpha_ewald * potential.rc) ** 2)
        params.pppm_pp_err = 2.0 * np.exp(alpha_times_rcut) / np.sqrt(potential.rc)
        params.pppm_pp_err *= np.sqrt(params.total_num_ptcls) * params.a_ws ** 2 / np.sqrt(params.pbox_volume)


@njit
def coulomb_force_pppm(r_in, pot_matrix):
    """
    Calculate Potential and Force between two particles when the pppm algorithm is chosen.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables.\n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`) .

    Returns
    -------
    U : float
        Potential value.

    fr : float
        Force between two particles.

    """

    # Short-range cutoff to deal with divergence of the Coulomb potential
    rs = pot_matrix[2]
    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    alpha = pot_matrix[1]  # Ewald parameter alpha
    alpha_r = alpha * r
    r2 = r * r
    U = pot_matrix[0] * mt.erfc(alpha_r) / r
    f1 = mt.erfc(alpha_r) / r2
    f2 = (2.0 * alpha / np.sqrt(np.pi) / r) * np.exp(-(alpha_r ** 2))
    fr = pot_matrix[0] * (f1 + f2)

    return U, fr


@njit
def coulomb_force(r_in, pot_matrix):
    """
    Calculate the bare coulomb potential and force between two particles.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`) .

    Returns
    -------
    U : float
        Potential value.

    fr : float
        Force between two particles.

    """

    # Short-range cutoff to deal with divergence of the Coulomb potential
    rs = pot_matrix[2]
    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    U = pot_matrix[0] / r
    fr = U / r

    return U, fr
