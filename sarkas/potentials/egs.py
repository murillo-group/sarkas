"""
Module for handling EGS Potential as described in Ref. [Stanton2015]_
"""
import numpy as np
import numba as nb
import sys
import fdint
import yaml


def setup(params, read_input=True):
    """
    Updates ``params`` class with EGS potential parameters.

    Parameters
    ----------
    read_input: bool
        Flag to read inputs from YAML input file.

    params : object
        Simulation's parameters

    """

    # open the input file to read EGS parameters
    if read_input:
        with open(params.input_file, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)
            for lkey in dics:
                if lkey == "Potential":
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if key == "rc":  # cutoff
                                params.potential.rc = float(value)

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
    params : object
        Simulation's parameters

    """

    """
    EGS_matrix[0,i,j] : kappa = 1.0/lambda_TF or given as input. Same value for all species.
    EGS_matrix[1,i,j] : nu = eq.(14) of Ref
    EGS_matrix[2,i,j] : qi qj/(4 pi esp0) Force factor between two particles.
    EGS_matrix[3:6,i,j] : other parameters see below.
    EGS_matrix[7,i,j] : Ewald parameter in the case of P3M Algorithm. Same value for all species
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

    # constants and conversion factors
    twopi = 2.0 * np.pi
    beta_i = 1.0 / (params.kB * params.Ti)

    if not hasattr(params, "Te"):
        print("Electron temperature is not defined. 1st species temperature ", params.species[0].temperature,
              "will be used as the electron temperature.")
        params.Te = params.species[0].temperature

    # lambda factor : 1 = von Weizsaecker, 1/9 = Thomas-Fermi
    lmbda = 1.0 / 9.0
    params.potential.lmbda = lmbda

    fdint_fdk_vec = np.vectorize(fdint.fdk)
    fdint_ifd1h_vec = np.vectorize(fdint.ifd1h)
    beta = 1. / (params.kB * params.Te)
    lambda_DB = np.sqrt(twopi * params.hbar2 * beta / params.me)
    lambda3 = lambda_DB ** 3
    # chemical potential of electron gas/(kB T). See eq.(4) in Ref.[3]_
    eta = fdint_ifd1h_vec(lambda3 * np.sqrt(np.pi) * params.ne / 4.0)
    # Thomas-Fermi length obtained from compressibility. See eq.(10) in Ref. [3]_
    lambda_TF = np.sqrt(params.fourpie0 * np.sqrt(np.pi) * lambda3 / (
            8.0 * np.pi * params.qe ** 2 * beta * fdint_fdk_vec(k=-0.5, phi=eta)))

    # eq. (14) of Ref. [1]_
    nu = - 3.0 * lmbda * (4.0 * np.pi * params.qe ** 2 * beta / params.fourpie0) / (
                4.0 * np.pi * lambda_DB) * fdint_fdk_vec(k=-1.5, phi=eta)
    # Fermi Energy
    E_F = (params.hbar2 / (2.0 * params.me)) * (3.0 * np.pi ** 2 * params.ne) ** (2. / 3.)
    # Degeneracy Parameter
    theta = 1.0 / (beta * E_F)
    if 0.1 <= theta <= 12:
        # Regime of validity of the following approximation Perrot et al. Phys Rev A 302619 (1984)
        # eq. (33) of Ref. [1]_
        Ntheta = 1.0 + 2.8343 * theta ** 2 - 0.2151 * theta ** 3 + 5.2759 * theta ** 4
        # eq. (34) of Ref. [1]_
        Dtheta = 1.0 + 3.9431 * theta ** 2 + 7.9138 * theta ** 4
        # eq. (32) of Ref. [1]_
        h = Ntheta / Dtheta * np.tanh(1.0 / theta)
        # grad h(x)
        gradh = (-(Ntheta / Dtheta) / np.cosh(1 / theta) ** 2 / (theta ** 2)  # derivative of tanh(1/x)
                 - np.tanh(1.0 / theta) * (
                             Ntheta * (7.8862 * theta + 31.6552 * theta ** 3) / Dtheta ** 2  # derivative of 1/Dtheta
                             + (5.6686 * theta - 0.6453 * theta ** 2 + 21.1036 * theta ** 3) / Dtheta)) # derivative of Ntheta
        # eq.(31) of Ref. [1]_
        b = 1.0 - 1.0 / 8.0 * theta * (h - 2.0 * theta * gradh)  # *(params.hbar2/lambda_TF**2)/params.me
    else:
        b = 1.0

    params.potential.b = b
    params.potential.theta = theta

    params.lambda_TF = lambda_TF
    # Monotonic decay
    if nu <= 1:
        # eq. (29) of Ref. [1]_
        params.potential.lambda_p = lambda_TF * np.sqrt(nu / (2.0 * b + 2.0 * np.sqrt(b ** 2 - nu)))
        params.potential.lambda_m = lambda_TF * np.sqrt(nu / (2.0 * b - 2.0 * np.sqrt(b ** 2 - nu)))
        params.potential.alpha = b / np.sqrt(b - nu)

    # Oscillatory behavior
    if nu > 1:
        # eq. (29) of Ref. [1]_
        params.potential.gamma_m = lambda_TF * np.sqrt(nu / (np.sqrt(nu) - b))
        params.potential.gamma_p = lambda_TF * np.sqrt(nu / (np.sqrt(nu) + b))
        params.potential.alphap = b / np.sqrt(nu - b)

    params.potential.nu = nu

    # Calculate the (total) plasma frequency
    wp_tot_sq = 0.0
    for i, sp in enumerate(params.species):
        wp2 = 4.0 * np.pi * sp.charge ** 2 * sp.num_density / (sp.mass * params.fourpie0)
        sp.wp = np.sqrt(wp2)
        wp_tot_sq += wp2

    params.wp = np.sqrt(wp_tot_sq)

    if params.pppm.on:
        EGS_matrix = np.zeros((7, params.num_species, params.num_species))
    else:
        EGS_matrix = np.zeros((8, params.num_species, params.num_species))

    EGS_matrix[0, :, :] = 1.0 / params.lambda_TF
    EGS_matrix[1, :, :] = params.potential.nu

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

            if nu <= 1:
                EGS_matrix[2, i, j] = (Zi * Zj) * params.qe * params.qe / (2.0 * params.fourpie0)
                EGS_matrix[3, i, j] = (1.0 + params.potential.alpha)
                EGS_matrix[4, i, j] = (1.0 - params.potential.alpha)
                EGS_matrix[5, i, j] = params.potential.lambda_m
                EGS_matrix[6, i, j] = params.potential.lambda_p

            if nu > 1:
                EGS_matrix[2, i, j] = (Zi * Zj) * params.qe * params.qe / params.fourpie0
                EGS_matrix[3, i, j] = 1.0
                EGS_matrix[4, i, j] = params.potential.alphap
                EGS_matrix[5, i, j] = params.potential.gamma_m
                EGS_matrix[6, i, j] = params.potential.gamma_p

    # Effective Coupling Parameter in case of multi-species
    # see eq.(3) in Haxhimali et al. Phys Rev E 90 023104 (2014)
    params.potential.Gamma_eff = Z53 * Z_avg ** (1. / 3.) * params.qe ** 2 * beta_i / (params.fourpie0 * params.aws)
    params.QFactor = params.QFactor / params.fourpie0

    params.potential.matrix = EGS_matrix

    if params.potential.method == "PP":
        params.force = EGS_force_PP
        params.PP_err = np.sqrt(twopi / params.lambda_TF) * np.exp(-params.potential.rc / params.lambda_TF)
        # Renormalize
        params.PP_err *= params.aws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)

    if params.potential.method == "P3M":
        print("\nERROR: P3M Algorithm not implemented yet. Good Bye!")
        sys.exit()


@nb.njit
def EGS_force_PP(r, pot_matrix):
    """ 
    Calculates Potential and force between particles using the EGS Potential.
    
    Parameters
    ----------
    r : float
        Particles' distance.

    pot_matrix : array
        EGS potential parameters. 

    Return
    ------

    U : float
        Potential.

    fr : float
        Force.

    """
    nu = pot_matrix[1]
    if nu <= 1.0:
        # pot_matrix[2] = Charge factor
        # pot_matrix[3] = 1 + alpha
        # pot_matrix[4] = 1 - alpha
        # pot_matrix[5] = lambda_minus
        # pot_matrix[6] = lambda_plus

        temp1 = pot_matrix[3] * np.exp(-r / pot_matrix[5])
        temp2 = pot_matrix[4] * np.exp(-r / pot_matrix[6])
        U = (temp1 + temp2) * pot_matrix[2] / r
        fr = U / r + pot_matrix[2] * (temp1 / pot_matrix[5] + temp2 / pot_matrix[6])/r
        fr /= r
    else:
        # pot_matrix[2] = Charge factor
        # pot_matrix[3] = 1.0
        # pot_matrix[4] = alpha prime
        # pot_matrix[5] = gamma_minus
        # pot_matrix[6] = gamma_plus
        cos = np.cos(r / pot_matrix[5])
        sin = np.sin(r / pot_matrix[5])
        exp = pot_matrix[2] * np.exp(-r / pot_matrix[6])
        U = (pot_matrix[3] * cos + pot_matrix[4] * sin) * exp / r
        fr1 = U / r   # derivative of 1/r
        fr3 = U / pot_matrix[6]   # derivative of exp
        fr2 = (sin - pot_matrix[4] * cos) * exp / (r * pot_matrix[5])
        fr = (fr1 + fr2 + fr3)/r

    return U, fr
