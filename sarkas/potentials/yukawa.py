"""
Module for handling Yukawa interaction
"""
import numpy as np
from numba import njit
import math as mt


@njit
def yukawa_force_pppm(r, pot_matrix):
    """ 
    Calculates Potential and Force between two particles when the P3M algorithm is chosen.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : array
        Potential matrix. See setup function above.

    Returns
    -------
    U_s_r : float
        Potential value
                
    fr : float
        Force between two particles calculated using eq.(22) in Ref. [Dharuman2017] .

    References
    ----------
    .. [Dharuman2017]: `Dharuman et al. J. Chem. Phys. 146, 024112 (2017) <https://doi.org/10.1063/1.4973842>`_

    """
    kappa = pot_matrix[1]
    alpha = pot_matrix[2]  # Ewald parameter alpha

    kappa_alpha = kappa / alpha
    alpha_r = alpha * r
    kappa_r = kappa * r
    U_s_r = pot_matrix[0] * (0.5 / r) * (np.exp(kappa_r) * mt.erfc(alpha_r + 0.5 * kappa_alpha)
                                         + np.exp(-kappa_r) * mt.erfc(alpha_r - 0.5 * kappa_alpha))
    # Derivative of the exponential term and 1/r
    f1 = (0.5 / r) * np.exp(kappa * r) * mt.erfc(alpha_r + 0.5 * kappa_alpha) * (1.0 / r - kappa)
    f2 = (0.5 / r) * np.exp(-kappa * r) * mt.erfc(alpha_r - 0.5 * kappa_alpha) * (1.0 / r + kappa)
    # Derivative of erfc(a r) = 2a/sqrt(pi) e^{-a^2 r^2}* (x/r)
    f3 = (alpha / np.sqrt(np.pi) / r) * (np.exp(-(alpha_r + 0.5 * kappa_alpha) ** 2) * np.exp(kappa_r)
                                              + np.exp(-(alpha_r - 0.5 * kappa_alpha) ** 2) * np.exp(-kappa_r))
    fr = pot_matrix[0] * (f1 + f2 + f3)

    return U_s_r, fr


@njit
def yukawa_force(r, pot_matrix):
    """ 
    Calculates Potential and Force between two particles.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : array
        It contains potential dependent variables.

    Returns
    -------
    U : float
        Potential.
                
    force : float
        Force between two particles.
    
    """
    U = pot_matrix[0] * np.exp(-pot_matrix[1] * r) / r
    force = U * (1 / r + pot_matrix[1])

    return U, force


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

    if potential.method == "P3M":
        potential.matrix = np.zeros((3, params.num_species, params.num_species))
    else:
        potential.matrix = np.zeros((2, params.num_species, params.num_species))

    twopi = 2.0 * np.pi
    potential.matrix[1, :, :] = 1.0 / params.lambda_TF

    for i, q1 in enumerate(params.species_charges):
        for j, q2 in enumerate(params.species_charges):
            potential.matrix[0, i, j] = q1 * q2 / params.fourpie0

    if potential.method == "PP":
        potential.force = yukawa_force
        # Force error calculated from eq.(43) in Ref.[1]_
        params.force_error = np.sqrt( twopi / params.lambda_TF) * np.exp(- potential.rc / params.lambda_TF)
        # Renormalize
        params.force_error *= params.a_ws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)
    elif potential.method == "P3M":
        potential.force = yukawa_force_pppm
        potential.matrix[2, :, :] = potential.pppm_alpha_ewald
        # PP force error calculation. Note that the equation was derived for a single component plasma.
        kappa_over_alpha = - 0.25 * (potential.matrix[1, 0, 0] / potential.matrix[2, 0, 0]) ** 2
        alpha_times_rcut = - (potential.matrix[2, 0, 0] * potential.rc) ** 2
        params.pppm_pp_err = 2.0 * np.exp(kappa_over_alpha + alpha_times_rcut) / np.sqrt(potential.rc)
        params.pppm_pp_err *= np.sqrt(params.total_num_ptcls) * params.a_ws ** 2 / np.sqrt(params.box_volume)
