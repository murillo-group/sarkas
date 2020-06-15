""" Module for handling the Quantum Statistical Potential as given by Ref [1]_

Note
----
Notice that in Ref. [1] the DeBroglie wavelength is defined as

.. math::
   \lambda_{ee} = \dfrac{\hbar}{\sqrt{2 \pi \mu_{ee} k_{B} T} },

while in statistical physics textbooks and Ref. [2]_ is defined as

.. math::
   \lambda_{ee} = \dfrac{h}{\sqrt{2 \pi \mu_{ee} k_{B} T} },

References
----------
.. [1] `J.P. Hansen and I.R. McDonald, Phys Rev A 23 2041 (1981) <https://doi.org/10.1103/PhysRevA.23.2041>`_
.. [2] `J.N. Glosli et al. Phys Rev E 78 025401(R) (2008) <https://doi.org/10.1103/PhysRevE.78.025401>`_
"""
import numpy as np
import numba as nb
import yaml
import math as mt
import sys


def setup(params, filename):
    """ 
    Update the ``params`` class with QSP Potential parameters. 
    The QSP Potential is given by eq.(5) in Ref. [2]_ .

    Parameters
    ----------
    filename : str
        Input file's name.

    params : class
        Simulation parameters. See ``S_params.py`` for more info.

    """
    """
    Dev Notes
    for more info and a description of the potential's parameters.
    QSP_matrix[0,:,:] = de Broglie wavelengths,
    QSP_matrix[1,:,:] = qi*qj/4*pi*eps0
    QSP_matrix[2,:,:] = 2pi/deBroglie
    QSP_matrix[3,:,:] = e-e Hansen-Pauli factor term factor
    QSP_matrix[4,:,:] = e-e exp factor 4pi/deBroglie/ln(2) 
    """
    # Default attributes
    params.Potential.QSP_type = 'Deutsch'
    params.Potential.QSP_Pauli = True
    # open the input file to read Yukawa parameters
    with open(filename, 'r') as stream:
        dics = yaml.load(stream, Loader=yaml.FullLoader)
        for lkey in dics:
            if lkey == "Potential":
                for keyword in dics[lkey]:
                    for key, value in keyword.items():
                        if key == "QSP_type":  # screening
                            params.Potential.QSP_type = value
                        if key == "QSP_Pauli":
                            params.Potential.QSP_Pauli = value

    if params.P3M.on:
        QSP_matrix = np.zeros((6, params.num_species, params.num_species))
    else:
        QSP_matrix = np.zeros((5, params.num_species, params.num_species))
        # open the input file to read Yukawa parameters

    if params.Control.units == "cgs":
        # constants and conversion factors
        params.ne = params.species[0].num_density
        params.ae = (3.0 / (4.0 * np.pi * params.ne)) ** (1.0 / 3.0)  # e WS
        params.rs = params.ae / params.a0  # e coupling parameter

    else:
        # e,i simulation parameters
        params.ne = params.species[0].num_density
        params.ae = (3.0 / (4.0 * np.pi * params.ne)) ** (1.0 / 3.0)  # e WS

    twopi = 2.0 * np.pi
    params.rs = params.ae / params.a0  # e coupling parameter
    params.Te = params.species[0].temperature
    params.Ti = params.species[1].temperature

    beta_e = 1.0 / (params.kB * params.Te)
    beta_i = 1.0 / (params.kB * params.Ti)

    for ic in range(params.num_species):
        params.species[ic].charge = params.qe
        params.species[ic].concentration = params.species[ic].num_density / params.total_num_density

        if hasattr(params.species[ic], "Z"):
            params.species[ic].charge = params.qe * params.species[ic].Z

    for i in range(params.num_species):
        m1 = params.species[i].mass
        q1 = params.species[i].charge

        for j in range(params.num_species):
            m2 = params.species[j].mass
            q2 = params.species[j].charge
            reduced = (m1 * m2) / (m1 + m2)
            if i == 0:
                Lambda_dB = np.sqrt(2.0 * np.pi * beta_e * params.hbar2 / reduced)
                QSP_matrix[0, i, j] = Lambda_dB
                if j == i:  # e-e
                    QSP_matrix[3, i, j] = np.log(2.0) / beta_e
                    QSP_matrix[4, i, j] = 4.0 * np.pi / (np.log(2.0) * Lambda_dB ** 2)
            else:
                Lambda_dB = np.sqrt(2.0 * np.pi * beta_i * params.hbar2 / reduced)
                QSP_matrix[0, i, j] = Lambda_dB

            QSP_matrix[1, i, j] = q1 * q2 / params.fourpie0
            QSP_matrix[2, i, j] = twopi / Lambda_dB

    params.QFactor = params.QFactor / params.fourpie0
    if params.Potential.QSP_Pauli == 0:
        QSP_matrix[3, :, :] = 0.0

    params.Potential.matrix = QSP_matrix

    # Calculate the (total) plasma frequency
    wp_tot_sq = 0.0
    for i in range(params.num_species):
        wp2 = 4.0 * np.pi * params.species[i].charge ** 2 * params.species[i].num_density / (
                params.species[i].mass * params.fourpie0)
        params.species[i].wp = np.sqrt(wp2)
        wp_tot_sq += wp2

    params.wp = np.sqrt(wp_tot_sq)

    # Calculate the Wigner-Seitz Radius
    params.aws = (3.0 / (4.0 * np.pi * params.total_num_density)) ** (1. / 3.)

    params.ni = 0.0
    for j in range(params.num_species - 1):
        params.ni += params.species[j + 1].num_density

    params.ai = (3.0 / (4.0 * np.pi * params.ni)) ** (1.0 / 3.0)  # Ion WS
    params.Potential.Gamma_eff = abs(params.Potential.matrix[1, 0, 1]) / (params.ai * params.kB * params.Ti)
    # Rescale all the Lengths by the ion's WS Radius instead of the total WS radius. 
    params.L = params.ai * (4.0 * np.pi * params.total_num_ptcls / 3.0) ** (1.0 / 3.0)  # box length
    L = params.L
    params.Lx = L
    params.Ly = L
    params.Lz = L
    params.Lv = np.array([L, L, L])  # box length vector
    params.d = np.count_nonzero(params.Lv)  # no. of dimensions

    params.box_volume = L * L * L
    params.Lmax_v = np.array([L, L, L])
    params.Lmin_v = np.array([0.0, 0.0, 0.0])

    if params.Potential.method == "P3M":

        if params.Potential.QSP_type == "Deutsch":
            params.force = Deutsch_force_P3M
        elif params.Potential.QSP_type == "Kelbg":
            params.force = Kelbg_force_P3M
        # P3M parameters
        params.P3M.hx = params.Lx / params.P3M.Mx
        params.P3M.hy = params.Ly / params.P3M.My
        params.P3M.hz = params.Lz / params.P3M.Mz
        params.Potential.matrix[5, :, :] = params.P3M.G_ew
        # Optimized Green's Function
        params.P3M.G_k, params.P3M.kx_v, params.P3M.ky_v, params.P3M.kz_v, params.P3M.PM_err, params.P3M.PP_err = gf_opt(
            params.P3M.MGrid, params.P3M.aliases, params.Lv, params.P3M.cao, params.Potential.matrix,
            params.Potential.rc, params.fourpie0)

        params.P3M.PP_err *= np.sqrt(params.N) * params.aws ** 2 * params.fourpie0
        params.P3M.PM_err *= np.sqrt(params.N) * params.aws ** 2 * params.fourpie0 / (params.box_volume ** (2. / 3.))
        params.P3M.F_err = np.sqrt(params.P3M.PM_err ** 2 + params.P3M.PP_err ** 2)

    else:
        raise AttributeError('QSP interaction can only be calculated using P3M algorithm.')

    return


@nb.njit
def Deutsch_force_P3M(r, pot_matrix):
    """ 
    Calculates the Deutsch QSP Force between two particles.

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
    """
    pot_matrix[0] = de Broglies wavelength,
    pot_matrix[1] = qi*qj/4*pi*eps0
    pot_matrix[2] = 2pi/deBroglie
    pot_matrix[3] = e-e term factor
    pot_matrix[4] = e-e exp factor 4pi/L_dB^2/ln(2)
    """
    A = pot_matrix[1]
    C = pot_matrix[2]
    D = pot_matrix[3]
    F = pot_matrix[4]
    alpha = pot_matrix[5]  # Ewald parameter alpha

    # Ewald short-range force and potential
    a2 = alpha * alpha
    r2 = r * r
    U_s_r = A * mt.erfc(alpha * r) / r

    # Diffraction term
    f1 = mt.erfc(alpha * r) / r2 / r
    f2 = (2.0 * alpha / np.sqrt(np.pi) / r2) * np.exp(- a2 * r2)
    fterm1 = A * (f1 + f2)

    # Exponential terms of the potential and force
    U_pp = -A * np.exp(-C * r) / r
    ee_pot_term = D * np.exp(-F * r2)

    fterm2 = U_pp / r2
    fterm3 = -A * C * np.exp(-C * r) / r2
    fterm4 = 2.0 * D * F * np.exp(-F * r2)

    U = U_s_r + U_pp + ee_pot_term
    force = fterm1 + fterm2 + fterm3 + fterm4

    return U, force


@nb.njit
def Kelbg_force_P3M(r, pot_matrix):
    """ 
    Calculates the QSP Force between two particles when the P3M algorithm is chosen.

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
    """
    pot_matrix[0] = de Broglies wavelength,
    pot_matrix[1] = qi*qj/4*pi*eps0
    pot_matrix[2] = 2pi/deBroglie
    pot_matrix[3] = e-e term factor
    pot_matrix[4] = e-e exp factor 4pi/L_dB^2/ln(2)
    """

    # Ewald force corresponding to the 1/r term of the potential

    A = pot_matrix[1]
    C = pot_matrix[2]
    C2 = C * C
    D = pot_matrix[3]
    F = pot_matrix[4]
    alpha = pot_matrix[5]  # Ewald parameter alpha

    # Ewald short-range force and potential
    a2 = alpha * alpha
    r2 = r * r
    U_s_r = A * mt.erfc(alpha * r) / r

    f1 = mt.erfc(alpha * r) / r2 / r
    f2 = (2.0 * alpha / np.sqrt(np.pi) / r2) * np.exp(- a2 * r2)
    fterm1 = A * (f1 + f2)

    # Diffraction terms of the potential and force
    U_pp = -A * np.exp(-C2 * r2 / np.pi) / r
    U_pp2 = + A * C * mt.erfc(C * r / np.sqrt(pi))

    # Pauli Term
    ee_pot_term = D * np.exp(-F * r2)

    fterm_pp = -A * (2.0 * C2 * r2 + np.pi) * np.exp(- C2 * r2 / np.pi) / (np.pi * r * r2)
    # erfc derivative
    fterm_pp2 = 2.0 * A * C2 * np.exp(- C2 * r2 / np.pi) / r / np.pi
    fterm_ee = 2.0 * D * F * np.exp(-F * r2)

    U = U_s_r + U_pp + U_pp2 + ee_pot_term
    force = fterm1 + fterm_pp + fterm_pp2 + fterm_ee

    return U, force


@nb.njit
def gf_opt(MGrid, aliases, BoxLv, p, pot_matrix, rcut, fourpie0):
    """
    Calculates the Optimized Green Function given by eq.(22) of Ref.[3]_ .

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

    pot_matrix : array
        Potential matrix. It contains screening parameter and Ewald parameter. See potential matrix above.

    rcut : float
        Cutoff distance for the PP calculation.

    fourpie0 : float
        Potential factor.

    Returns
    -------
    G_k : array
        optimal Green Function.

    kx_v : array
       array of reciprocal space vectors along the x-axis.

    ky_v : array
       array of reciprocal space vectors along the y-axis.

    kz_v : array
       array of reciprocal space vectors along the z-axis.

    PM_err : float
        Error in the force calculation due to the optimized Green's function. eq.(28) of Ref.[3]_ .

    PP_err : float
        Error in the e-e force calculation due to the distance cutoff see eq.(30) of Ref.[4]_ .
   
    References
    ----------
    .. [3] `H.A. Stern et al. J Chem Phys 128, 214006 (2008) <https://doi.org/10.1063/1.2932253>`_
    .. [4] `G. Dharuman et al. J Chem Phys 146 024112 (2017) <https://doi.org/10.1063/1.4973842>`_

    """

    # Grab the e-e interaction parameters only. 
    C = pot_matrix[2, 0, 0]
    D = pot_matrix[3, 0, 0]
    F = pot_matrix[4, 0, 0]

    Gew = pot_matrix[-1, 0, 0]  # params.Potential.matrix[5,0,0]

    rcut2 = rcut * rcut
    mx_max = aliases[0]  # params.P3M.mx_max
    my_max = aliases[1]  # params.P3M.my_max
    mz_max = aliases[2]  # params.P3M.mz_max
    Mx = MGrid[0]  # params.P3M.Mx
    My = MGrid[1]  # params.P3M.My
    Mz = MGrid[2]  # params.P3M.Mz
    # hx = params.P3M.hx
    # hy = params.P3M.hy
    # hz = params.P3M.hz
    Lx = BoxLv[0]  # params.Lx
    Ly = BoxLv[1]  # params.Ly
    Lz = BoxLv[2]  # params.Lz
    hx = Lx / float(Mx)
    hy = Ly / float(My)
    hz = Lz / float(Mz)

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
        four_pi = 4.0 * np.pi
    else:
        four_pi = 4.0 * np.pi / fourpie0

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

                                G_k_M = four_pi * np.exp(-0.25 * k_M_sq / Gew_sq) / k_M_sq

                                k_dot_k_M = kx * kx_M + ky * ky_M + kz * kz_M

                                U_G_k += (U_k_M_sq * G_k_M * k_dot_k_M)
                                U_k_sq += U_k_M_sq

                    # eq.(22) of Ref. [3]_                                                  
                    G_k[nz, ny, nx] = U_G_k / ((U_k_sq ** 2) * k_sq)

                    # eq. (9) of Ref. [3]_
                    Gk_hat = four_pi * np.exp(-0.25 * k_sq / Gew_sq) / k_sq

                    # eq.(28) of Ref. [3]_
                    PM_err += Gk_hat * Gk_hat * k_sq - U_G_k ** 2 / ((U_k_sq ** 2) * k_sq)

    # Calculate the PP error for the e-e interaction only. 
    # This is because the electron's DeBroglie wavelength is much shorter than the ion's, hence longer-range force.     
    PP_err_exp = 2.0 * np.pi * np.exp(- 2.0 * C * rcut) * (C * rcut + 2) / rcut
    PP_err_ee = -np.pi / 4. * (3.0 * np.sqrt(2.0 * np.pi / F) * mt.erf(np.sqrt(2.0 * F * rcut2))
                               + 4.0 * rcut * np.exp(-2.0 * F * rcut2) * (4.0 * F * rcut2 + 3))

    PP_err_ee *= D ** 2 * fourpie0 ** 2

    PP_err_Ew = 4.0 * np.exp(-2.0 * Gew_sq * rcut2) / rcut
    PP_err_tot = np.sqrt(PP_err_Ew + PP_err_ee + PP_err_exp) / (fourpie0 * np.sqrt(Lx * Ly * Lz))

    PM_err_tot = np.sqrt(PM_err) / (Lx * Ly * Lz) ** (1. / 3.)

    return G_k, kx_v, ky_v, kz_v, PM_err_tot, PP_err_tot
