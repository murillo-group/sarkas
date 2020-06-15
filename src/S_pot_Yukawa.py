"""
Module for handling Yukawa interaction
"""
import numpy as np
import numba as nb
import math as mt
import yaml  # IO
import fdint  # Fermi integrals calculation


@nb.njit
def Yukawa_force_P3M(r, pot_matrix_ij):
    """ 
    Calculates Potential and Force between two particles when the P3M algorithm is chosen.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix_ij : array
        Potential matrix. See setup function above.

    Returns
    -------
    U_s_r : float
        Potential value
                
    fr : float
        Force between two particles calculated using eq.(22) in Ref. [1]_

    References
    ----------
    .. [1] `G. Dharuman et al. J Chem Phys 146 024112 (2017) <https://doi.org/10.1063/1.4973842>`_
    """
    kappa = pot_matrix_ij[0]

    G = pot_matrix_ij[3]  # Ewald parameter alpha

    U_s_r = pot_matrix_ij[2] * (0.5 / r) * (np.exp(kappa * r) * mt.erfc(G * r + 0.5 * kappa / G)
                                            + np.exp(-kappa * r) * mt.erfc(G * r - 0.5 * kappa / G))
    # Derivative of the exponential term and 1/r
    f1 = (0.5 / r ** 2) * np.exp(kappa * r) * mt.erfc(G * r + 0.5 * kappa / G) * (1.0 / r - kappa)
    f2 = (0.5 / r ** 2) * np.exp(-kappa * r) * mt.erfc(G * r - 0.5 * kappa / G) * (1.0 / r + kappa)
    # Derivative of erfc(a r) = 2a/sqrt(pi) e^{-a^2 r^2}* (x/r)
    f3 = (G / np.sqrt(np.pi) / r ** 2) * (np.exp(-(G * r + 0.5 * kappa / G) ** 2) * np.exp(kappa * r)
                                          + np.exp(-(G * r - 0.5 * kappa / G) ** 2) * np.exp(-kappa * r))
    fr = pot_matrix_ij[2] * (f1 + f2 + f3)

    return U_s_r, fr


@nb.njit
def Yukawa_force_PP(r, pot_matrix_ij):
    """ 
    Calculates Potential and Force between two particles.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix_ij : array
        It contains potential dependent variables.
                    
    Returns
    -------
    U : float
        Potential.
                
    force : float
        Force between two particles.
    
    """

    """
    Dev Notes
    -----    
    pot_matrix_ij[0,:,:] = 1/lambda_TF
    pot_matrix_ij[1,i,j] = Gamma_ij
    pot_matrix_ij[2,i,j] = q1*q2/(4*pi*eps0)

    """

    U = pot_matrix_ij[2] * np.exp(-pot_matrix_ij[0] * r) / r
    force = U * (1 / r + pot_matrix_ij[0]) / r

    return U, force


@nb.njit
def gf_opt(MGrid, aliases, BoxLv, p, pot_matrix, rcut, fourpie0):
    """ 
    Calculate the Optimized Green Function given by eq.(22) of Ref. [2]_.

    Parameters
    ----------
    MGrid : array
        number of mesh points in x,y,z

    aliases : array
        number of aliases in each direction

    BoxLv : array
        Length of simulation's box in each direction

    p : int
        Charge assignment order (CAO)

    pot_matrix : array
        Potential matrix. It contains screening parameter and Ewald parameter. See potential matrix above.

    rcut : float
        Cutoff distance for the PP calculation

    fourpie0 : float
        Potential factor.

    Returns
    -------
    G_k : array_like
        optimal Green Function

    kx_v : array_like
       array of reciprocal space vectors along the x-axis

    ky_v : array_like
       array of reciprocal space vectors along the y-axis

    kz_v : array_like
       array of reciprocal space vectors along the z-axis

    PM_err : float
        Error in the force calculation due to the optimized Green's function. eq.(28) of Ref. [2]_

    PP_err : float
        Error in the force calculation due to the distance cutoff. eq.(30) of Ref. [1]_
   
    References
    ----------
    .. [2] `H.A. Stern et al. J Chem Phys 128, 214006 (2008) <https://doi.org/10.1063/1.2932253>`_
    """
    kappa = pot_matrix[0, 0, 0]  # params.Potential.matrix[0,0,0]
    Gew = pot_matrix[-1, 0, 0]  # params.Potential.matrix[3,0,0]
    rcut2 = rcut * rcut
    mx_max = aliases[0]  # params.P3M.mx_max
    my_max = aliases[1]  # params.P3M.my_max
    mz_max = aliases[2]  # params.P3M.mz_max
    Mx = MGrid[0]  # params.P3M.Mx
    My = MGrid[1]  # params.P3M.My
    Mz = MGrid[2]  # params.P3M.Mz
    Lx = BoxLv[0]  # params.Lx
    Ly = BoxLv[1]  # params.Ly
    Lz = BoxLv[2]  # params.Lz
    hx = Lx / float(Mx)
    hy = Ly / float(My)
    hz = Lz / float(Mz)

    kappa_sq = kappa * kappa
    Gew_sq = Gew * Gew

    G_k = np.zeros((Mz, My, Mx))

    if np.mod(Mz, 2) == 0:
        nz_mid = Mz / 2
    else:
        nz_mid = (Mz - 1) / 2

    if np.mod(My, 2) == 0:
        ny_mid = My / 2
    else:
        ny_mid = (My - 1) / 2

    if np.mod(Mx, 2) == 0:
        nx_mid = Mx / 2
    else:
        nx_mid = (Mx - 1) / 2

    nx_v = np.arange(Mx).reshape((1, Mx))
    ny_v = np.arange(My).reshape((My, 1))
    nz_v = np.arange(Mz).reshape((Mz, 1, 1))

    kx_v = 2.0 * np.pi * (nx_v - nx_mid) / Lx
    ky_v = 2.0 * np.pi * (ny_v - ny_mid) / Ly
    kz_v = 2.0 * np.pi * (nz_v - nz_mid) / Lz

    PM_err = 0.0

    if fourpie0 == 1.0:
        four_pi = 4.0*np.pi
    else:
        four_pi = 4.0*np.pi/fourpie0

    for nz in range(Mz):
        nz_sh = nz - nz_mid
        kz = 2.0 * np.pi * nz_sh / Lz

        for ny in range(My):
            ny_sh = ny - ny_mid
            ky = 2.0 * np.pi * ny_sh / Ly

            for nx in range(Mx):
                nx_sh = nx - nx_mid
                kx = 2.0 * np.pi * nx_sh / Lx

                k_sq = kx * kx + ky * ky + kz * kz

                if k_sq != 0.0:

                    U_k_sq = 0.0
                    U_G_k = 0.0

                    # Sum over the aliases
                    for mz in range(-mz_max, mz_max + 1):
                        for my in range(-my_max, my_max + 1):
                            for mx in range(-mx_max, mx_max + 1):

                                kx_M = 2.0 * np.pi * (nx_sh + mx * Mx) / Lx
                                ky_M = 2.0 * np.pi * (ny_sh + my * My) / Ly
                                kz_M = 2.0 * np.pi * (nz_sh + mz * Mz) / Lz

                                k_M_sq = kx_M ** 2 + ky_M ** 2 + kz_M ** 2

                                if kx_M != 0.0:
                                    U_kx_M = np.sin(0.5 * kx_M * hx) / (0.5 * kx_M * hx)
                                else:
                                    U_kx_M = 1.0

                                if ky_M != 0.0:
                                    U_ky_M = np.sin(0.5 * ky_M * hy) / (0.5 * ky_M * hy)
                                else:
                                    U_ky_M = 1.0

                                if kz_M != 0.0:
                                    U_kz_M = np.sin(0.5 * kz_M * hz) / (0.5 * kz_M * hz)
                                else:
                                    U_kz_M = 1.0

                                U_k_M = (U_kx_M * U_ky_M * U_kz_M) ** p
                                U_k_M_sq = U_k_M * U_k_M

                                G_k_M = four_pi * np.exp(-0.25 * (kappa_sq + k_M_sq) / Gew_sq) / (kappa_sq + k_M_sq)

                                k_dot_k_M = kx * kx_M + ky * ky_M + kz * kz_M

                                U_G_k += (U_k_M_sq * G_k_M * k_dot_k_M)
                                U_k_sq += U_k_M_sq

                    # eq.(22) of Ref.[2]_
                    G_k[nz, ny, nx] = U_G_k / ((U_k_sq ** 2) * k_sq)
                    Gk_hat = four_pi * np.exp(-0.25 * (kappa_sq + k_sq) / Gew_sq) / (kappa_sq + k_sq)

                    # eq.(28) of Ref.[2]_
                    PM_err = PM_err + Gk_hat * Gk_hat * k_sq - U_G_k ** 2 / ((U_k_sq ** 2) * k_sq)

    PP_err = 2.0 / np.sqrt(Lx * Ly * Lz * rcut) * np.exp(-0.25 * kappa_sq / Gew_sq) * np.exp(-Gew_sq * rcut2)
    PM_err = np.sqrt(PM_err) / (Lx * Ly * Lz) ** (1. / 3.)

    return G_k, kx_v, ky_v, kz_v, PM_err, PP_err


def setup(params, filename):
    """ 
    Updates ``params`` class with Yukawa's parameters.

    Parameters
    ----------
    params : class
        Simulation's parameters. See S_params.py for more info.

    filename : str
        Input filename.

    References
    ----------
    .. [3] `Stanton and Murillo Phys Rev E 91 033104 (2015) <https://doi.org/10.1103/PhysRevE.91.033104>`_
    .. [4] `T. Haxhimali et al. Phys Rev E 90 023104 (2014) <https://doi.org/10.1103/PhysRevE.90.023104>`_
    """

    """
    Dev Notes
    ---------
    Yukawa_matrix[0,:,:] : kappa = 1.0/lambda_TF or given as input. Same value for all species.
    Yukawa_matrix[1,i,j] : Gamma = qi qj/(4pi esp0 kb T), Coupling parameter between particles' species.
    Yukawa_matrix[2,i,j] : qi qj/(4pi esp0) Force factor between two particles.
    Yukawa_matrix[3,i,j] : Ewald parameter in the case of P3M Algorithm. Same value for all species
    """

    twopi = 2.0 * np.pi
    beta_i = 1.0 / (params.kB * params.Ti)

    # open the input file to read Yukawa parameters
    with open(filename, 'r') as stream:
        dics = yaml.load(stream, Loader=yaml.FullLoader)
        for lkey in dics:
            if lkey == "Potential":
                for keyword in dics[lkey]:
                    for key, value in keyword.items():
                        if key == "kappa":  # screening
                            params.Potential.kappa = float(value)

                        # electron temperature for screening parameter calculation
                        if key == "elec_temperature":
                            params.Te = float(value)

                        if key == "elec_temperature_eV":
                            T_eV = float(value)
                            params.Te = params.eV2K * float(value)

    if params.P3M.on:
        Yukawa_matrix = np.zeros((4, params.num_species, params.num_species))
    else:
        Yukawa_matrix = np.zeros((3, params.num_species, params.num_species))

    if hasattr(params.Potential, "kappa") and hasattr(params.Potential, "Te"):
        print(
            "\nWARNING: You have provided both kappa and Te while only one is needed. kappa will be used to calculate "
            "the screening parameter.")

    if hasattr(params.Potential, "kappa"):
        # Thomas-Fermi Length
        lambda_TF = params.aws / params.Potential.kappa
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
        # chemical potential of electron gas/(kB T). See eq.(4) in Ref.[3]_
        eta = fdint_ifd1h_vec(lambda3 * np.sqrt(np.pi) * params.ne / 4.0)
        # Thomas-Fermi length obtained from compressibility. See eq.(10) in Ref. [3]_
        lambda_TF = np.sqrt(params.fourpie0 * np.sqrt(np.pi) * lambda3 / (
                    8.0 * np.pi * params.qe ** 2 * beta * fdint_fdk_vec(k=-0.5, phi=eta)))
    # Calculate the Potential Matrix
    Z53 = 0.0
    Z_avg = 0.0

    for i in range(params.num_species):
        if hasattr(params.species[i], "Z"):
            Zi = params.species[i].Z
        else:
            Zi = 1.0

        Z53 += (Zi) ** (5. / 3.) * params.species[i].concentration
        Z_avg += Zi * params.species[i].concentration

        for j in range(params.num_species):
            if hasattr(params.species[j], "Z"):
                Zj = params.species[j].Z
            else:
                Zj = 1.0

            Yukawa_matrix[1, i, j] = (Zi * Zj) * params.qe ** 2 * beta_i / (params.fourpie0 * params.aws)  # Gamma_ij
            Yukawa_matrix[2, i, j] = (Zi * Zj) * params.qe ** 2 / params.fourpie0

    # Effective Coupling Parameter in case of multi-species
    # see eq.(3) in Ref.[4]_
    params.Potential.Gamma_eff = Z53 * Z_avg ** (1. / 3.) * params.qe ** 2 * beta_i / (params.fourpie0 * params.aws)
    params.QFactor /= params.fourpie0
    params.Potential.matrix = Yukawa_matrix

    # Calculate the (total) plasma frequency
    params.lambda_TF = lambda_TF
    Yukawa_matrix[0, :, :] = 1.0 / params.lambda_TF  # kappa/ai
    wp_tot_sq = 0.0
    for i in range(params.num_species):
        wp2 = 4.0 * np.pi * params.species[i].charge ** 2 * params.species[i].num_density / (
                    params.species[i].mass * params.fourpie0)
        params.species[i].wp = np.sqrt(wp2)
        wp_tot_sq += wp2

    params.wp = np.sqrt(wp_tot_sq)

    if params.Potential.method == "PP" or params.Potential.method == "brute":
        params.force = Yukawa_force_PP
        # Force error calculated from eq.(43) in Ref.[1]_
        params.PP_err = np.sqrt(twopi / params.lambda_TF) * np.exp(-params.Potential.rc / params.lambda_TF)
        # Renormalize
        params.PP_err = params.PP_err * params.aws ** 2 * np.sqrt(params.N / params.box_volume)

    if params.Potential.method == "P3M":
        params.force = Yukawa_force_P3M
        # P3M parameters
        params.P3M.hx = params.Lx / float(params.P3M.Mx)
        params.P3M.hy = params.Ly / float(params.P3M.My)
        params.P3M.hz = params.Lz / float(params.P3M.Mz)
        params.Potential.matrix[-1, :, :] = params.P3M.G_ew
        # Optimized Green's Function
        params.P3M.G_k, params.P3M.kx_v, params.P3M.ky_v, params.P3M.kz_v, params.P3M.PM_err, params.P3M.PP_err = gf_opt(
            params.P3M.MGrid, params.P3M.aliases, params.Lv, params.P3M.cao, params.Potential.matrix,
            params.Potential.rc, params.fourpie0)

        # Include the charges in the Force errors. Prefactor in eq.(29) Ref.[1]_
        # Notice that the equation was derived for a single component plasma. 
        params.P3M.PP_err *= np.sqrt(params.N) * params.aws ** 2 * params.fourpie0
        params.P3M.PM_err *= np.sqrt(params.N) * params.aws ** 2 * params.fourpie0 / params.box_volume ** (2. / 3.)
        # Total force error
        params.P3M.F_err = np.sqrt(params.P3M.PM_err ** 2 + params.P3M.PP_err ** 2)

    return
