""" 
Module for handling Coulomb interaction
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
    """
    Dev Notes:
    -----
    Coulomb_matrix[0,i,j] : qi qj/(4pi esp0) Force factor between two particles.
    Coulomb_matrix[1,i,j] : Ewald parameter in the case of P3M Algorithm. Same value for all species
    """
    potential.matrix = np.zeros((2, params.num_species, params.num_species))

    for i, q1 in enumerate(params.species_charges):
        for j, q2 in enumerate(params.species_charges):
            potential.matrix[0, i, j] = q1 * q2 / params.fourpie0

    potential.matrix[1, :, :] = potential.pppm_alpha_ewald
    # Calculate the (total) plasma frequency
    potential.force = coulomb_force_pppm

    # PP force error calculation. Note that the equation was derived for a single component plasma.
    alpha_times_rcut = - (potential.pppm_alpha_ewald * potential.rc) ** 2
    params.pppm_pp_err = 2.0 * np.exp(alpha_times_rcut) / np.sqrt(potential.rc)
    params.pppm_pp_err *= np.sqrt(params.total_num_ptcls) * params.a_ws ** 2 / np.sqrt(params.box_volume)


@njit
def coulomb_force_pppm(r, pot_matrix):
    """ 
    Calculate Potential and Force between two particles when the P3M algorithm is chosen.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables.

    Returns
    -------
    U_s_r : float
        Potential value.
                
    fr : float
        Force between two particles. 
    
    """

    alpha = pot_matrix[1]  # Ewald parameter alpha
    alpha_r = alpha * r
    r2 = r * r
    U = pot_matrix[0] * mt.erfc(alpha_r) / r
    f1 = mt.erfc(alpha_r) / r2
    f2 = (2.0 * alpha / np.sqrt(np.pi) / r) * np.exp(- alpha_r ** 2)
    fr = pot_matrix[0] * (f1 + f2)

    return U, fr


@njit
def coulomb_force(r, pot_matrix):
    """
    Calculate the coulomb potential and force between two particles.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables.

    Returns
    -------
    U : float
        Potential value.

    fr : float
        Force between two particles.

    """

    U = pot_matrix[0] / r
    fr = U / r

    return U, fr
