"""
Module for handling Moliere Potential
"""
import numpy as np
import numba as nb
import sys
import yaml


def update_params(potential, params):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : sarkas.potential.Potential
        Class handling potential form.

    params : sarkas.base.Parameters
        Simulation's parameters.

    """
    potential.screening_lengths = np.array(potential.screening_lengths)
    potential.screening_charges = np.array(potential.screening_charges)
    params_len = len(potential.screening_lengths)

    Moliere_matrix = np.zeros((2 * params_len + 1, params.num_species, params.num_species))

    for i, q1 in enumerate(params.species_charges):
        for j, q2 in enumerate(params.species_charges):

            Moliere_matrix[0, i, j] = q1 * q2 / params.fourpie0
            Moliere_matrix[1:params_len + 1, i, j] = potential.screening_charges
            Moliere_matrix[params_len + 1:, i, j] = potential.screening_lengths

    # Effective Coupling Parameter in case of multi-species
    # see eq.(3) in Haxhimali et al. Phys Rev E 90 023104 (2014)
    potential.matrix = Moliere_matrix

    potential.force = Moliere_force_PP

    # Force error calculated from eq.(43) in Ref.[1]_
    params.force_error = np.sqrt(2.0 * np.pi / potential.screening_lengths.min()) \
                    * np.exp(- potential.rc / potential.screening_lengths.min())
    # Renormalize
    params.force_error *= params.aws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)


@nb.njit
def Moliere_force_PP(r, pot_matrix):
    """ 
    Calculates the PP force between particles using the Moliere Potential.
    
    Parameters
    ----------
    r : float
        Particles' distance.

    pot_matrix : numpy.ndarray
        Moliere potential parameters. 


    Returns
    -------
    phi : float
        Potential

    fr : float
        Force
    """
    """
    Notes
    -----
    See Wilson et al. PRB 15 2458 (1977) for parameters' details
    pot_matrix[0] = Z_1Z_2e^2/(4 np.pi eps_0)
    pot_matrix[1] = C_1
    pot_matrix[2] = C_2
    pot_matrix[3] = C_3
    pot_matrix[4] = b_1
    pot_matrix[5] = b_2
    pot_matrix[6] = b_3
    """

    U = 0.0
    force = 0.0

    for i in range(int(len(pot_matrix[:-1]) / 2)):
        factor1 = r * pot_matrix[i + 4]
        factor2 = pot_matrix[i + 1] / r
        U += factor2 * np.exp(-factor1)
        force += np.exp(-factor1) * factor2 * (1.0 / r + pot_matrix[i + 1])

    force = force * pot_matrix[0]
    U = U * pot_matrix[0]

    return U, force
