""" 
Module for handling Coulomb interaction
"""

import numpy as np
from numba import njit
import yaml  # IO
import math as mt
from sarkas.algorithm.force_pm import force_optimized_green_function as gf_opt


def setup(params, read_input=True):
    """
    Update ``params`` class with Coulomb's parameters.

    Parameters
    ----------
    params : object
        Simulation's parameters

    read_input: bool
        Flag to read inputs from YAML input file.
    """
    # Do a bunch of checks
    # P3M algorithm only
    assert params.potential.method == "P3M",  'QSP interaction can only be calculated using P3M algorithm.'

    # open the input file to read Yukawa parameters
    if read_input:
        with open(params.input_file, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)
            for lkey in dics:
                if lkey == "Potential":
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if key == "rc":  # cutoff
                                params.potential.rc = float(value)

    update_params(params)


def update_params(params):
    """
    Create potential dependent simulation's parameters.

    Parameters
    ----------
    params : object
        Simulation's parameters

    """
    """
    Dev Notes:
    -----
    Coulomb_matrix[0,i,j] : qi qj/(4pi esp0) Force factor between two particles.
    Coulomb_matrix[1,i,j] : Ewald parameter in the case of P3M Algorithm. Same value for all species
    """

    if not params.BC.open_axes:
        params.potential.LL_on = True  # linked list on
        if not hasattr(params.potential, "rc"):
            print("\nWARNING: The cut-off radius is not defined. L/2 = ", params.Lv.min() / 2, "will be used as rc")
            params.potential.rc = params.Lv.min() / 2.
            params.potential.LL_on = False  # linked list off

        if params.potential.method == "PP" and params.potential.rc > params.Lv.min() / 2.:
            print("\nWARNING: The cut-off radius is > L/2. L/2 = ", params.Lv.min() / 2, "will be used as rc")
            params.potential.rc = params.Lv.min() / 2.
            params.potential.LL_on = False  # linked list off

    Coulomb_matrix = np.zeros((2, params.num_species, params.num_species))

    beta_i = 1.0 / (params.kB * params.Ti)

    # Create the Potential Matrix
    Z53 = 0.0
    Z_avg = 0.0
    for i, sp1 in enumerate(params.species):
        if hasattr(sp1, "Z"):
            Zi = sp1.Z
        else:
            Zi = 1.0

        Z53 += Zi ** (5. / 3.) * sp1.concentration
        Z_avg += Zi * sp1.concentration

        for j, sp2 in enumerate(params.species):
            if hasattr(sp2, "Z"):
                Zj = sp2.Z
            else:
                Zj = 1.0

            Coulomb_matrix[0, i, j] = Zi * Zj * params.qe ** 2 / params.fourpie0

    # Effective Coupling Parameter in case of multi-species
    # see eq.(3) of T. Haxhimali et al. Phys Rev E 90 023104 (2014) <https://doi.org/10.1103/PhysRevE.90.023104>
    params.potential.Gamma_eff = Z53 * Z_avg ** (1. / 3.) * params.qe ** 2 * beta_i / (params.fourpie0 * params.aws)
    params.QFactor = params.QFactor / params.fourpie0
    params.potential.matrix = Coulomb_matrix

    # Calculate the (total) plasma frequency
    # Calculate the (total) plasma frequency
    wp_tot_sq = 0.0
    for i, sp in enumerate(params.species):
        wp2 = 4.0 * np.pi * sp.charge ** 2 * sp.num_density / (sp.mass * params.fourpie0)
        sp.wp = np.sqrt(wp2)
        wp_tot_sq += wp2

    params.wp = np.sqrt(wp_tot_sq)

    params.force = Coulomb_force_P3M
    # P3M parameters
    params.pppm.hx = params.Lx / params.pppm.Mx
    params.pppm.hy = params.Ly / params.pppm.My
    params.pppm.hz = params.Lz / params.pppm.Mz
    params.potential.matrix[1, :, :] = params.pppm.G_ew
    # Calculate the Optimized Green's Function
    constants = np.array([0.0, params.pppm.G_ew, params.fourpie0])
    params.pppm.G_k, params.pppm.kx_v, params.pppm.ky_v, params.pppm.kz_v, params.pppm.PM_err = gf_opt(
        params.pppm.MGrid, params.pppm.aliases, params.Lv, params.pppm.cao, constants)
    # Complete PM Force error calculation
    params.pppm.PM_err *= np.sqrt(params.total_num_ptcls) * params.aws ** 2 * params.fourpie0 / params.box_volume ** (2. / 3.)

    # PP force error calculation. Note that the equation was derived for a single component plasma.
    alpha_times_rcut = - (params.potential.matrix[1, 0, 0] * params.potential.rc) ** 2
    params.pppm.PP_err = 2.0 * np.exp(alpha_times_rcut) / np.sqrt(params.potential.rc)
    params.pppm.PP_err *= np.sqrt(params.total_num_ptcls) * params.aws ** 2 / np.sqrt(params.box_volume)

    # Total Force Error
    params.pppm.F_err = np.sqrt(params.pppm.PM_err ** 2 + params.pppm.PP_err ** 2)


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
    fr = pot_matrix[0] * (f1 + f2) / r

    return U_s_r, fr
