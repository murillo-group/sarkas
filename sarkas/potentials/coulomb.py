""" 
Module for handling Coulomb interaction
"""

import numpy as np
from numba import njit
import math as mt
from sarkas.algorithm.force_pm import force_optimized_green_function as gf_opt


def setup(params):
    """
    Update ``params`` class with Coulomb's parameters.

    Parameters
    ----------
    params : class
        Simulation's parameters. See ``S_params.py`` for more info.

    """
    """
    Dev Notes:
    -----
    Coulomb_matrix[0,i,j] : qi qj/(4pi esp0) Force factor between two particles.
    Coulomb_matrix[1,i,j] : Ewald parameter in the case of P3M Algorithm. Same value for all species
    """

    # Do a bunch of checks
    # P3M algorithm only
    if not params.Potential.method == "P3M":
        raise AttributeError('QSP interaction can only be calculated using P3M algorithm.')

    Coulomb_matrix = np.zeros((2, params.num_species, params.num_species))

    beta_i = 1.0 / (params.kB * params.Ti)

    # Create the Potential Matrix
    Z53 = 0.0
    Z_avg = 0.0
    for i in range(params.num_species):
        if hasattr(params.species[i], "Z"):
            Zi = params.species[i].Z
        else:
            Zi = 1.0

        Z53 += Zi ** (5. / 3.) * params.species[i].concentration
        Z_avg += Zi * params.species[i].concentration

        for j in range(params.num_species):
            if hasattr(params.species[j], "Z"):
                Zj = params.species[j].Z
            else:
                Zj = 1.0

            Coulomb_matrix[0, i, j] = Zi * Zj * params.qe ** 2 / params.fourpie0

    # Effective Coupling Parameter in case of multi-species
    # see eq.(3) of T. Haxhimali et al. Phys Rev E 90 023104 (2014) <https://doi.org/10.1103/PhysRevE.90.023104>
    params.Potential.Gamma_eff = Z53 * Z_avg ** (1. / 3.) * params.qe ** 2 * beta_i / (params.fourpie0 * params.aws)
    params.QFactor = params.QFactor / params.fourpie0
    params.Potential.matrix = Coulomb_matrix

    # Calculate the (total) plasma frequency
    wp_tot_sq = 0.0
    for i in range(params.num_species):
        wp2 = 4.0 * np.pi * params.species[i].charge ** 2 * params.species[i].num_density / (
                    params.species[i].mass * params.fourpie0)
        params.species[i].wp = np.sqrt(wp2)
        wp_tot_sq += wp2

    params.wp = np.sqrt(wp_tot_sq)

    params.force = Coulomb_force_P3M
    # P3M parameters
    params.P3M.hx = params.Lx / params.P3M.Mx
    params.P3M.hy = params.Ly / params.P3M.My
    params.P3M.hz = params.Lz / params.P3M.Mz
    params.Potential.matrix[1, :, :] = params.P3M.G_ew
    # Calculate the Optimized Green's Function
    constants = np.array([0.0, params.P3M.G_ew, params.fourpie0])
    params.P3M.G_k, params.P3M.kx_v, params.P3M.ky_v, params.P3M.kz_v, params.P3M.PM_err = gf_opt(
        params.P3M.MGrid, params.P3M.aliases, params.Lv, params.P3M.cao, constants)
    # Complete PM Force error calculation
    params.P3M.PM_err *= np.sqrt(params.N) * params.aws ** 2 * params.fourpie0 / params.box_volume ** (2. / 3.)

    # PP force error calculation. Note that the equation was derived for a single component plasma.
    alpha_times_rcut = - (params.Potential.matrix[1, 0, 0] * params.Potential.rc) ** 2
    params.P3M.PP_err = 2.0 * np.exp(alpha_times_rcut) / np.sqrt(params.Potential.rc)
    params.P3M.PP_err *= np.sqrt(params.N) * params.aws ** 2 / np.sqrt(params.box_volume)

    # Total Force Error
    params.P3M.F_err = np.sqrt(params.P3M.PM_err ** 2 + params.P3M.PP_err ** 2)
    return


@njit
def Coulomb_force_P3M(r, pot_matrix):
    """ 
    Calculate Potential and Force between two particles when the P3M algorithm is chosen.

    Parameters
    ----------
    r : real
        Distance between two particles.

    pot_matrix : array
        It contains potential dependent variables.
        pot_matrix[0] = q1*q2/(4*pi*eps0)
        pot_matrix[1] = Ewald parameter alpha

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
    U_s_r = pot_matrix[0] * mt.erfc(alpha_r) / r
    f1 = mt.erfc(alpha_r) / r2
    f2 = (2.0 * alpha / np.sqrt(np.pi) / r) * np.exp(- alpha_r ** 2)
    fr = pot_matrix[0] * (f1 + f2)/r

    return U_s_r, fr
