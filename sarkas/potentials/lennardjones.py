"""
Module for handling Lennard-Jones interaction
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
    potential.matrix = np.zeros((4, params.num_species, params.num_species))
    # See Lima Physica A 391 4281 (2012) for the following definitions
    if not hasattr(potential, 'powers'):
        potential.powers = np.array([12, 6])

    exponent = potential.powers[0] / (potential.powers[1] - potential.powers[0])
    lj_constant = potential.powers[1]/(potential.powers[0] - potential.powers[1])
    lj_constant *= (potential.powers[1]/potential.powers[0]) ** exponent

    # Use the Lorentz-Berthelot mixing rules.
    # Lorentz: sigma_12 = 0.5 * (sigma_1 + sigma_2)
    # Berthelot: epsilon_12 = sqrt( eps_1 eps2)
    sigma2 = 0.0
    epsilon_tot = 0.0
    # Recall that species_charges contains sqrt(epsilon)
    for i, q1 in enumerate(params.species_charges):
        for j, q2 in enumerate(params.species_charges):
            potential.matrix[0, i, j] = lj_constant * q1 * q2
            potential.matrix[1, i, j] = 0.5 * (params.species_lj_sigmas[i] + params.species_lj_sigmas[j])
            potential.matrix[2, i, j] = potential.powers[0]
            potential.matrix[3, i, j] = potential.powers[1]
            sigma2 += params.species_lj_sigmas[i]
            epsilon_tot += q1 * q2

    potential.force = lj_force

    # Calculate the force error
    deriv_exp = (potential.powers[0] + 1)
    params.force_error = np.sqrt(np.pi * sigma2 ** potential.powers[0])
    params.force_error /= np.sqrt(deriv_exp * potential.rc ** deriv_exp)
    params.force_error *= np.sqrt(params.total_num_ptcls / params.box_volume) * params.a_ws ** 2


@nb.njit
def lj_force(r, pot_matrix_ij):
    """
    Calculates the PP force between particles using Lennard-Jones Potential.
    
    Parameters
    ----------
    pot_matrix_ij : array
        LJ potential parameters. 

    r : float
        Particles' distance.


    Returns
    -------
    U : float
        Potential.

    force : float
        Force.
    """
    """
    Notes
    -----
    pot_matrix[0] = epsilon_12 * lj_constant
    pot_matrix[1] = sigmas
    pot_matrix[2] = highest power
    pot_matrix[3] = lowest power
    
    """

    epsilon = pot_matrix_ij[0]
    sigma = pot_matrix_ij[1]
    s_over_r = sigma / r
    s_over_r_high = s_over_r ** pot_matrix_ij[2]
    s_over_r_low = s_over_r ** pot_matrix_ij[3]

    U = epsilon * (s_over_r_high - s_over_r_low)
    force = epsilon * (pot_matrix_ij[2] * s_over_r_high - pot_matrix_ij[3] * s_over_r_low) / r

    return U, force
