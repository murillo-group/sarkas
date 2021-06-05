"""
Module for handling Moliere Potential
"""
import numpy as np
import numba as nb


def update_params(potential, params):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : sarkas.potentials.core.Potential
        Class handling potential form.

    params : sarkas.core.Parameters
        Simulation's parameters.

    """
    potential.screening_lengths = np.array(potential.screening_lengths)
    potential.screening_charges = np.array(potential.screening_charges)
    params_len = len(potential.screening_lengths)

    moliere_matrix = np.zeros((2 * params_len + 1, params.num_species, params.num_species))

    for i, q1 in enumerate(params.species_charges):
        for j, q2 in enumerate(params.species_charges):

            moliere_matrix[0, i, j] = q1 * q2 / params.fourpie0
            moliere_matrix[1:params_len + 1, i, j] = potential.screening_charges
            moliere_matrix[params_len + 1:, i, j] = potential.screening_lengths

    potential.matrix = moliere_matrix
    potential.force = moliere_force

    # Force error calculated from eq.(43) in Ref.[1]_
    params.force_error = np.sqrt(2.0 * np.pi / potential.screening_lengths.min()) \
                    * np.exp(- potential.rc / potential.screening_lengths.min())
    # Renormalize
    params.force_error *= params.a_ws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)


@nb.njit
def moliere_force(r, pot_matrix):
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

    force *= pot_matrix[0]
    U *= pot_matrix[0]

    return U, force
