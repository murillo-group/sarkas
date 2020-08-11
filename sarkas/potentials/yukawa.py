"""
Module for handling Yukawa interaction
"""
import numpy as np
from numba import njit
import math as mt
import yaml  # IO
import fdint  # Fermi integrals calculation
from sarkas.algorithm.force_pm import force_optimized_green_function as gf_opt


@njit
def Yukawa_force_P3M(r, pot_matrix):
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
        Force between two particles calculated using eq.(22) in Ref. [Dharuman2017]_ .

    """
    kappa = pot_matrix[1]
    alpha = pot_matrix[2]  # Ewald parameter alpha

    kappa_alpha = kappa / alpha
    alpha_r = alpha * r
    kappa_r = kappa * r
    U_s_r = pot_matrix[0] * (0.5 / r) * (np.exp(kappa_r) * mt.erfc(alpha_r + 0.5 * kappa_alpha)
                                         + np.exp(-kappa_r) * mt.erfc(alpha_r - 0.5 * kappa_alpha))
    # Derivative of the exponential term and 1/r
    f1 = (0.5 / r ** 2) * np.exp(kappa * r) * mt.erfc(alpha_r + 0.5 * kappa_alpha) * (1.0 / r - kappa)
    f2 = (0.5 / r ** 2) * np.exp(-kappa * r) * mt.erfc(alpha_r - 0.5 * kappa_alpha) * (1.0 / r + kappa)
    # Derivative of erfc(a r) = 2a/sqrt(pi) e^{-a^2 r^2}* (x/r)
    f3 = (alpha / np.sqrt(np.pi) / r ** 2) * (np.exp(-(alpha_r + 0.5 * kappa_alpha) ** 2) * np.exp(kappa_r)
                                              + np.exp(-(alpha_r - 0.5 * kappa_alpha) ** 2) * np.exp(-kappa_r))
    fr = pot_matrix[0] * (f1 + f2 + f3)

    return U_s_r, fr


@njit
def Yukawa_force_PP(r, pot_matrix):
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
    force = U * (1 / r + pot_matrix[1]) / r

    return U, force


def setup(params, read_input=True):
    """ 
    Updates ``params`` class with Yukawa's parameters.

    Parameters
    ----------
    read_input: bool
        Flag to read inputs from YAML input file.

    params: object
        Simulation's parameters.

    """

    """
    Dev Notes
    ---------
    Yukawa_matrix[0,i,j] : qi qj/(4pi esp0) Force factor between two particles.
    Yukawa_matrix[1,:,:] : kappa = 1.0/lambda_TF or given as input. Same value for all species.
    Yukawa_matrix[2,i,j] : Ewald parameter in the case of P3M Algorithm. Same value for all species
    """

    # open the input file to read Yukawa parameters
    if read_input:
        with open(params.input_file, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)
            for key, value in dics["Potential"].items():
                if key == "kappa":  # screening
                    params.potential.kappa = float(value)

                # electron temperature for screening parameter calculation
                if key == "elec_temperature":
                    params.Te = float(value)

                if key == "elec_temperature_eV":
                    params.Te = params.eV2K * float(value)

    update_params(params)


def update_params(params):
    """
    Create potential dependent simulation's parameters.

    Parameters
    ----------
    params: object
        Simulation's parameters.

    References
    ----------
    .. [Stanton2015] `Stanton and Murillo Phys Rev E 91 033104 (2015) <https://doi.org/10.1103/PhysRevE.91.033104>`_
    .. [Haxhimali2014] `T. Haxhimali et al. Phys Rev E 90 023104 (2014) <https://doi.org/10.1103/PhysRevE.90.023104>`_
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

    if params.pppm.on:
        Yukawa_matrix = np.zeros((3, params.num_species, params.num_species))
    else:
        Yukawa_matrix = np.zeros((2, params.num_species, params.num_species))

    if hasattr(params.potential, "kappa") and hasattr(params.potential, "Te"):
        print(
            "\nWARNING: You have provided both kappa and Te while only one is needed. kappa will be used to calculate "
            "the screening parameter.")

    twopi = 2.0 * np.pi
    beta_i = 1.0 / (params.kB * params.Ti)

    if hasattr(params.potential, "kappa"):
        # Thomas-Fermi Length
        lambda_TF = params.aws / params.potential.kappa
        Yukawa_matrix[0, :, :] = 1.0 / lambda_TF

    else:  # if kappa is not given calculate it from the electron temperature
        if not hasattr(params, "Te"):
            print("\nElectron temperature is not defined. 1st species temperature ", params.species[0].temperature,
                  "will be used as the electron temperature.")
            params.Te = params.species[0].temperature

        fdint_fdk_vec = np.vectorize(fdint.fdk)
        fdint_ifd1h_vec = np.vectorize(fdint.ifd1h)
        beta = 1. / (params.kB * params.Te)
        thermal_wavelength = np.sqrt(2.0 * np.pi * params.hbar2 * beta / params.me)
        lambda3 = thermal_wavelength ** 3
        # chemical potential of electron gas/(kB T). See eq.(4) in Ref.[Stanton2015]_
        eta = fdint_ifd1h_vec(lambda3 * np.sqrt(np.pi) * params.ne / 4.0)
        # Thomas-Fermi length obtained from compressibility. See eq.(10) in Ref. [Stanton2015]_
        lambda_TF = np.sqrt(params.fourpie0 * np.sqrt(np.pi) * lambda3 / (
                8.0 * np.pi * params.qe ** 2 * beta * fdint_fdk_vec(k=-0.5, phi=eta)))
    # Calculate the Potential Matrix
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

            Yukawa_matrix[0, i, j] = (Zi * Zj) * params.qe ** 2 / params.fourpie0

    # Effective Coupling Parameter in case of multi-species
    # see eq.(3) in Ref.[Haxhimali2014]_
    params.potential.Gamma_eff = Z53 * Z_avg ** (1. / 3.) * params.qe ** 2 * beta_i / (params.fourpie0 * params.aws)
    params.QFactor /= params.fourpie0

    params.lambda_TF = lambda_TF
    Yukawa_matrix[1, :, :] = 1.0 / params.lambda_TF  # kappa/ai
    params.potential.matrix = Yukawa_matrix

    # Calculate the (total) plasma frequency
    wp_tot_sq = 0.0
    for i, sp in enumerate(params.species):
        wp2 = 4.0 * np.pi * sp.charge ** 2 * sp.number_density / (sp.mass * params.fourpie0)
        sp.wp = np.sqrt(wp2)
        wp_tot_sq += wp2

    params.wp = np.sqrt(wp_tot_sq)

    if params.potential.method == "PP":
        params.force = Yukawa_force_PP
        # Force error calculated from eq.(43) in Ref.[1]_
        params.PP_err = np.sqrt(twopi / params.lambda_TF) * np.exp(-params.potential.rc / params.lambda_TF)
        # Renormalize
        params.PP_err *= params.aws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)
    elif params.potential.method == "P3M":
        params.force = Yukawa_force_P3M
        # P3M parameters
        params.pppm.hx = params.Lx / float(params.pppm.Mx)
        params.pppm.hy = params.Ly / float(params.pppm.My)
        params.pppm.hz = params.Lz / float(params.pppm.Mz)
        params.potential.matrix[-1, :, :] = params.pppm.G_ew
        # Calculate the Optimized Green's Function
        constants = np.array([params.potential.matrix[1, 0, 0], params.pppm.G_ew, params.fourpie0])
        params.pppm.G_k, params.pppm.kx_v, params.pppm.ky_v, params.pppm.kz_v, params.pppm.PM_err = gf_opt(
            params.pppm.MGrid, params.pppm.aliases, params.Lv, params.pppm.cao, constants)
        # Complete PM Force error calculation
        params.pppm.PM_err *= np.sqrt(params.total_num_ptcls) * params.aws ** 2 * params.fourpie0 / params.box_volume ** (2. / 3.)

        # PP force error calculation. Note that the equation was derived for a single component plasma.
        kappa_over_alpha = - 0.25 * (params.potential.matrix[1, 0, 0] / params.potential.matrix[2, 0, 0]) ** 2
        alpha_times_rcut = - (params.potential.matrix[2, 0, 0] * params.potential.rc) ** 2
        params.pppm.PP_err = 2.0 * np.exp(kappa_over_alpha + alpha_times_rcut) / np.sqrt(params.potential.rc)
        params.pppm.PP_err *= np.sqrt(params.total_num_ptcls) * params.aws ** 2 / np.sqrt(params.box_volume)

        # Total force error
        params.pppm.F_err = np.sqrt(params.pppm.PM_err ** 2 + params.pppm.PP_err ** 2)
