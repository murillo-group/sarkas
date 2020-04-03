""" 
Module for handling Coulomb interaction
"""

import numpy as np
import numba as nb
import math as mt
import sys


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
    Coulomb_matrix[0,i,j] : Gamma = qi qj/(4pi esp0*kb T), Coupling parameter between particles' species.
    Coulomb_matrix[1,i,j] : qi qj/(4pi esp0) Force factor between two particles.
    Coulomb_matrix[2,i,j] : Ewald parameter in the case of P3M Algorithm. Same value for all species
    """

    if params.P3M.on:
        Coulomb_matrix = np.zeros((3, params.num_species, params.num_species))
    else:
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

        Z53 += (Zi) ** (5. / 3.) * params.species[i].concentration
        Z_avg += Zi * params.species[i].concentration

        for j in range(params.num_species):
            if hasattr(params.species[j], "Z"):
                Zj = params.species[j].Z
            else:
                Zj = 1.0

            Coulomb_matrix[0, i, j] = Zi * Zj * params.qe ** 2 * beta_i / (params.fourpie0 * params.aws)
            Coulomb_matrix[1, i, j] = Zi * Zj * params.qe ** 2 / params.fourpie0

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

    if params.Potential.method == "PP" or params.Potential.method == "brute":
        print('\nERROR: Coulomb interaction cannot be implemented using LCL algorithm. Please choose P3M')
        sys.exit()

    if params.Potential.method == "P3M":
        params.force = Coulomb_force_P3M
        # P3M parameters
        params.P3M.hx = params.Lx / params.P3M.Mx
        params.P3M.hy = params.Ly / params.P3M.My
        params.P3M.hz = params.Lz / params.P3M.Mz
        params.Potential.matrix[2, :, :] = params.P3M.G_ew
        # Optimized Green's Function
        params.P3M.G_k, params.P3M.kx_v, params.P3M.ky_v, params.P3M.kz_v, params.P3M.PM_err, params.P3M.PP_err = gf_opt(
            params.P3M.MGrid, params.P3M.aliases, params.Lv, params.P3M.cao, params.P3M.G_ew, params.Potential.rc,
            params.fourpie0)

        # Notice that the equation was derived for a single component plasma. 
        params.P3M.PP_err *= np.sqrt(params.N) * params.aws ** 2 * params.fourpie0
        params.P3M.PM_err *= np.sqrt(params.N) * params.aws ** 2 * params.fourpie0 / params.box_volume ** (2. / 3.)
        # Total Force Error
        params.P3M.F_err = np.sqrt(params.P3M.PM_err ** 2 + params.P3M.PP_err ** 2)
    return


@nb.njit
def Coulomb_force_P3M(r, pot_matrix_ij):
    """ 
    Calculate Potential and Force between two particles when the P3M algorithm is chosen.

    Parameters
    ----------
    r : real
        Distance between two particles.

    pot_matrix_ij : array
        It contains potential dependent variables.
                    
    Returns
    -------
    U_s_r : float
        Potential value.
                
    fr : float
        Force between two particles. 
    
    """

    """
    pot_matrix_ij[0,:,:] = Gamma_ij
    pot_matrix_ij[1,:,:] = q1*q2/(4*pi*eps0)
    pot_matrix_ij[2,:,:] = Ewald parameter alpha
    """
    alpha = pot_matrix_ij[2]  # Ewald parameter alpha
    a2 = alpha * alpha
    r2 = r * r
    U_s_r = pot_matrix_ij[1] * mt.erfc(alpha * r) / r
    f1 = mt.erfc(alpha * r) / r2
    f2 = (2.0 * alpha / np.sqrt(np.pi) / r) * np.exp(- a2 * r2)
    fr = pot_matrix_ij[1] * (f1 + f2)

    return U_s_r, fr


@nb.njit
def gf_opt(MGrid, aliases, BoxLv, p, alpha, rcut, fourpie0):
    """ 
    Calculate the Optimized Green Function given by eq.(22) of Ref. [1]_ . 

    Parameters
    ----------
    MGrid : array
            number of mesh points in x,y,z.

    aliases : array
            number of aliases in each direction.

    BoxLv : array
            Length of simulation's box in each direction.

    p : int
        charge assignment order (CAO).

    alpha : float
        Ewald parameter.

    rcut : float
        Cutoff distance for the PP calculation.

    fourpie0 : float
        Potential factor.

    Returns
    -------
    G_k : array_like
        optimal Green Function.

    kx_v : array_like
        array of reciprocal space vectors along the x-axis.

    ky_v : array_like
        array of reciprocal space vectors along the y-axis.

    kz_v : array_like
        array of reciprocal space vectors along the z-axis.

    PM_err : float
        Error in the force calculation due to the optimized Green's function.

    PP_err : float
        Error in the force calculation due to the distance cutoff.
    
    References
    ----------
    .. [1] `H.A. Stern et al. J Chem Phys 128, 214006 (2008) <https://doi.org/10.1063/1.2932253>`_
    """
    Gew = alpha  # params.Potential.matrix[3,0,0]
    # p = params.P3M.cao
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

    Gew_sq = Gew * Gew

    CoulombFactor = 1.0 / fourpie0

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

                                G_k_M = CoulombFactor * np.exp(-0.25 * k_M_sq / Gew_sq) / k_M_sq

                                k_dot_k_M = kx * kx_M + ky * ky_M + kz * kz_M

                                U_G_k += (U_k_M_sq * G_k_M * k_dot_k_M)
                                U_k_sq += U_k_M_sq

                    # eq.(22) of Ref. [1]_
                    G_k[nz, ny, nx] = U_G_k / ((U_k_sq ** 2) * k_sq)
                    Gk_hat = CoulombFactor * np.exp(-0.25 * k_sq / Gew_sq) / k_sq

                    # eq.(28) of Ref. [1]_ 
                    PM_err += Gk_hat * Gk_hat * k_sq - U_G_k ** 2 / ((U_k_sq ** 2) * k_sq)

    PP_err = 2.0 * CoulombFactor / np.sqrt(Lx * Ly * Lz) * np.exp(-Gew_sq * rcut2) / np.sqrt(rcut)
    PM_err = np.sqrt(PM_err) / (Lx * Ly * Lz) ** (1. / 3.)

    return G_k, kx_v, ky_v, kz_v, PM_err, PP_err
