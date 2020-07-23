"""
Module for handling the Particle-Mesh part of the force and potential calculation.
"""

import numpy as np
from numba import jit, njit
import pyfftw

# These "ignore" are needed because numba does not support pyfftw yet
from numba.core.errors import NumbaWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@njit
def force_optimized_green_function(MGrid, aliases, BoxLv, p, constants):
    """
    Calculate the Optimized Green Function given by eq.(22) of Ref. [Stern2008]_.

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

    constants : array
        Screening parameter, Ewald parameter, 4 pi eps0.

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
    .. [Stern2008] `H.A. Stern et al. J Chem Phys 128, 214006 (2008) <https://doi.org/10.1063/1.2932253>`_
    """
    kappa = constants[0]
    Gew = constants[1]
    fourpie0 = constants[2]
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

                                G_k_M = four_pi * np.exp(-0.25 * (kappa_sq + k_M_sq) / Gew_sq) / (kappa_sq + k_M_sq)

                                k_dot_k_M = kx * kx_M + ky * ky_M + kz * kz_M

                                U_G_k += (U_k_M_sq * G_k_M * k_dot_k_M)
                                U_k_sq += U_k_M_sq

                    # eq.(22) of Ref.[2]_
                    G_k[nz, ny, nx] = U_G_k / ((U_k_sq ** 2) * k_sq)
                    Gk_hat = four_pi * np.exp(-0.25 * (kappa_sq + k_sq) / Gew_sq) / (kappa_sq + k_sq)

                    # eq.(28) of Ref.[2]_
                    PM_err += Gk_hat * Gk_hat * k_sq - U_G_k ** 2 / ((U_k_sq ** 2) * k_sq)

    PM_err = np.sqrt(PM_err) / (Lx * Ly * Lz) ** (1. / 3.)

    return G_k, kx_v, ky_v, kz_v, PM_err


@njit
def assgnmnt_func(cao, x):
    """ 
    Calculate the charge assignment function as given in Ref. [1]_ .
    
    Parameters
    ----------
    cao : int
        Charge assignment order.

    x : float
        Distance to closest mesh point if cao is even.
    
    Returns
    ------
    W : array
        Charge Assignment Function. 

    References
    ----------
    .. [1] `M. Deserno and C. Holm J Chem Phys 108, 7678 (1998) <https://doi.org/10.1063/1.477414>`_

    """
    W = np.zeros(cao)

    if cao == 1:

        W[0] = 1

    elif cao == 2:

        W[0] = 0.5 * (1. - 2. * x)
        W[1] = 0.5 * (1. + 2. * x)

    elif cao == 3:

        W[0] = (1. - 4. * x + 4. * x ** 2) / 8.
        W[1] = (3. - 4. * x ** 2) / 4.
        W[2] = (1. + 4. * x + 4. * x ** 2) / 8.

    elif cao == 4:

        W[0] = (1. - 6. * x + 12. * x ** 2 - 8. * x ** 3) / 48.
        W[1] = (23. - 30. * x - 12. * x ** 2 + 24. * x ** 3) / 48.
        W[2] = (23. + 30. * x - 12. * x ** 2 - 24. * x ** 3) / 48.
        W[3] = (1. + 6. * x + 12. * x ** 2 + 8. * x ** 3) / 48.

    elif cao == 5:

        W[0] = (1. - 8. * x + 24. * x ** 2 - 32. * x ** 3 + 16. * x ** 4) / 384.
        W[1] = (19. - 44. * x + 24. * x ** 2 + 16. * x ** 3 - 16. * x ** 4) / 96.
        W[2] = (115. - 120. * x ** 2 + 48. * x ** 4) / 192.
        W[3] = (19. + 44. * x + 24. * x ** 2 - 16. * x ** 3 - 16. * x ** 4) / 96.
        W[4] = (1. + 8. * x + 24. * x ** 2 + 32. * x ** 3 + 16. * x ** 4) / 384.

    elif cao == 6:
        W[0] = (1. - 10. * x + 40. * x ** 2 - 80. * x ** 3 + 80. * x ** 4 - 32. * x ** 5) / 3840.
        W[1] = (237. - 750. * x + 840. * x ** 2 - 240. * x ** 3 - 240. * x ** 4 + 160. * x ** 5) / 3840.
        W[2] = (841. - 770. * x - 440. * x ** 2 + 560. * x ** 3 + 80. * x ** 4 - 160. * x ** 5) / 1920.
        W[3] = (841. + 770. * x - 440. * x ** 2 - 560. * x ** 3 + 80. * x ** 4 + 160. * x ** 5) / 1920.
        W[4] = (237. + 750. * x + 840. * x ** 2 + 240. * x ** 3 - 240. * x ** 4 - 160. * x ** 5) / 3840.
        W[5] = (1. + 10. * x + 40. * x ** 2 + 80. * x ** 3 + 80. * x ** 4 + 32. * x ** 5) / 3840.

    elif cao == 7:

        W[0] = (1. - 12. * x + 60. * x * 2 - 160. * x ** 3 + 240. * x ** 4 - 192. * x ** 5 + 64. * x ** 6) / 46080.

        W[1] = (361. - 1416. * x + 2220. * x ** 2 - 1600. * x ** 3 + 240. * x ** 4
                + 384. * x ** 5 - 192. * x ** 6) / 23040.

        W[2] = (10543. - 17340. * x + 4740. * x ** 2 + 6880. * x ** 3 - 4080. * x ** 4
                - 960. * x ** 5 + 960. * x ** 6) / 46080.

        W[3] = (5887. - 4620. * x ** 2 + 1680. * x ** 4 - 320. * x ** 6) / 11520.

        W[4] = (10543. + 17340. * x + 4740. * x ** 2 - 6880. * x ** 3 - 4080. * x ** 4
                + 960. * x ** 5 + 960. * x ** 6) / 46080.

        W[5] = (361. + 1416. * x + 2220. * x ** 2 + 1600. * x ** 3 + 240. * x ** 4
                - 384. * x ** 5 - 192. * x ** 6) / 23040.

        W[6] = (1. + 12. * x + 60. * x ** 2 + 160. * x ** 3 + 240. * x ** 4 + 192. * x ** 5 + 64. * x ** 6) / 46080.

    return W


@njit
def calc_charge_dens(pos, Z, N, cao, Mx, My, Mz, hx, hy, hz):
    """ 
    Assigns Charges to Mesh Points.

    Parameters
    ----------
    pos : array
        Particles' positions.
    
    Z : array
        Particles' charges.
    
    N : int
        Number of particles.

    cao : int
        Charge assignment order.

    Mx : int
        Number of mesh points along x-axis.

    My : int
        Number of mesh points along y-axis.

    Mz : int
        Number of mesh points along z-axis.

    hx : int
        Distance between mesh points along x-axis.

    hy : int
        Distance between mesh points along y-axis.

    hz : int
        Distance between mesh points along z-axis.

    Returns
    -------
    rho_r : array
        Charge density distributed on mesh.

    """

    rho_r = np.zeros((Mz, My, Mz))

    # Mid point calculation
    if cao % 2 == 0:
        # Choose the midpoint between the two closest mesh point to the particle's position
        mid = 0.5
        pshift = int(cao / 2 - 1)
    else:
        # Choose the mesh point closes to the particle
        mid = 0.0
        pshift = int(cao / float(2.0))

    for ipart in range(N):

        # ix = x-coord of the (left) closest mesh point
        # (ix + 0.5)*hx = midpoint between the two mesh points closest to the particle
        # x = the difference between the particle's position and the midpoint
        # Rescale

        ix = int(pos[ipart, 0] / hx)
        x = pos[ipart, 0] - (ix + mid) * hx
        x = x / hx

        iy = int(pos[ipart, 1] / hy)
        y = pos[ipart, 1] - (iy + mid) * hy
        y = y / hy

        iz = int(pos[ipart, 2] / hz)
        z = pos[ipart, 2] - (iz + mid) * hz
        z = z / hz

        wx = assgnmnt_func(cao, x)
        wy = assgnmnt_func(cao, y)
        wz = assgnmnt_func(cao, z)

        izn = iz - pshift  # min. index along z-axis

        for g in range(cao):

            if izn < 0:
                r_g = izn + Mz
            elif izn > (Mz - 1):
                r_g = izn - Mz
            else:
                r_g = izn

            iyn = iy - pshift  # min. index along y-axis

            for i in range(cao):

                if iyn < 0:
                    r_i = iyn + My
                elif iyn > (My - 1):
                    r_i = iyn - My
                else:
                    r_i = iyn

                ixn = ix - pshift  # min. index along x-axis

                for j in range(cao):

                    if ixn < 0:
                        r_j = ixn + Mx
                    elif ixn > (Mx - 1):
                        r_j = ixn - Mx
                    else:
                        r_j = ixn

                    rho_r[r_g, r_i, r_j] += Z[ipart] * wz[g] * wy[i] * wx[j]

                    ixn += 1

                iyn += 1

            izn += 1

    return rho_r


@njit
def calc_field(phi_k, kx_v, ky_v, kz_v):
    """ 
    Calculates the Electric field in Fourier space.

    Parameters
    ----------
    phi_k : array
        3D array of the Potential.

    kx_v : array
        3D array containing the values of kx.

    ky_v : array
        3D array containing the values of ky.

    kz_v : array
        3D array containing the values of kz.
    
    Returns
    -------
    E_kx : array 
       Electric Field along kx-axis.

    E_ky : array
       Electric Field along ky-axis.

    E_kz : array
       Electric Field along kz-axis.
    
    """

    E_kx = -1j * kx_v * phi_k
    E_ky = -1j * ky_v * phi_k
    E_kz = -1j * kz_v * phi_k

    return E_kx, E_ky, E_kz


@njit
def calc_acc_pm(E_x_r, E_y_r, E_z_r, pos, Z, N, cao, Mass, Mx, My, Mz, hx, hy, hz):
    """ 
    Calculates the long range part of particles' accelerations. 
    
    Parameters
    ----------
    E_x_r : array
        Electric field along x-axis.

    E_y_r : array
        Electric field along y-axis.
    
    E_z_r : array
        Electric field along z-axis.
    
    pos : array
        Particles' positions.
    
    Z : array
        Particles' charges.
    
    N : int
        Number of particles.

    cao : int
        Charge assignment order.
    
    Mass : array
        Particles' masses.

    Mx : int
        Number of mesh points along x-axis.

    My : int
        Number of mesh points along y-axis.

    Mz : int
        Number of mesh points along z-axis.

    hx : int
        Distance between mesh points along x-axis.

    hy : int
        Distance between mesh points along y-axis.

    hz : int
        Distance between mesh points along z-axis.

    Returns
    -------

    acc : array
          Acceleration from Electric Field.

    """
    E_x_p = np.zeros(N)
    E_y_p = np.zeros(N)
    E_z_p = np.zeros(N)

    acc = np.zeros((N, 3))

    # Mid point calculation
    if cao % 2 == 0:
        # Choose the midpoint between the two closest mesh point to the particle's position
        mid = 0.5
        # Number of points to the left of the chosen one
        pshift = int(cao / 2 - 1)
    else:
        # Choose the mesh point closes to the particle
        mid = 0.0
        # Number of points to the left of the chosen one
        pshift = int(cao / float(2.0))

    for ipart in range(N):

        ix = int(pos[ipart, 0] / hx)
        x = pos[ipart, 0] - (ix + mid) * hx
        x = x / hx

        iy = int(pos[ipart, 1] / hy)
        y = pos[ipart, 1] - (iy + mid) * hy
        y = y / hy

        iz = int(pos[ipart, 2] / hz)
        z = pos[ipart, 2] - (iz + mid) * hz
        z = z / hz

        wx = assgnmnt_func(cao, x)
        wy = assgnmnt_func(cao, y)
        wz = assgnmnt_func(cao, z)

        izn = iz - pshift  # min. index along z-axis

        for g in range(cao):

            if izn < 0:
                r_g = izn + Mz
            elif izn > (Mz - 1):
                r_g = izn - Mz
            else:
                r_g = izn

            iyn = iy - pshift  # min. index along y-axis

            for i in range(cao):

                if iyn < 0:
                    r_i = iyn + My
                elif iyn > (My - 1):
                    r_i = iyn - My
                else:
                    r_i = iyn

                ixn = ix - pshift  # min. index along x-axis

                for j in range(cao):

                    if ixn < 0:
                        r_j = ixn + Mx
                    elif ixn > (Mx - 1):
                        r_j = ixn - Mx
                    else:
                        r_j = ixn

                    ZM = Z[ipart] / Mass[ipart]
                    E_x_p[ipart] += ZM * E_x_r[r_g, r_i, r_j] * wz[g] * wy[i] * wx[j]
                    E_y_p[ipart] += ZM * E_y_r[r_g, r_i, r_j] * wz[g] * wy[i] * wx[j]
                    E_z_p[ipart] += ZM * E_z_r[r_g, r_i, r_j] * wz[g] * wy[i] * wx[j]

                    ixn += 1

                iyn += 1

            izn += 1

    acc[:, 0] = E_x_p
    acc[:, 1] = E_y_p
    acc[:, 2] = E_z_p

    return acc


# FFTW version
@jit  # Numba does not support pyfftw yet, however, this decorator still speeds up the function.
def update(pos, Z, Mass, MGrid, Lv, G_k, kx_v, ky_v, kz_v, cao):
    """ 
    Calculate the long range part of particles' accelerations.

    Parameters
    ----------
    pos : array
        Particles' positions.

    Z : array
        Particles' charges, it corresponds to ``ptcls.charge``.

    Mass : array
        Particles' masses, it corresponds to ``ptcls.mass``. 

    MGrid : array
        Mesh grid points.

    Lv : array
        Box length in each direction.

    G_k : array
        Optimized Green's function.

    kx_v : array
        Array of kx values.

    ky_v : array
        Array of ky values.

    kz_v : array
        Array of kz values.
    
    cao : int
        Charge order parameter.

    Returns
    -------
    U_f : float
        Long range part of the potential.

    acc_f : array
        Long range part of particles' accelerations.

    """

    N = pos.shape[0]

    Lx = Lv[0]
    Ly = Lv[1]
    Lz = Lv[2]

    Mx = MGrid[0]
    My = MGrid[1]
    Mz = MGrid[2]

    hx = Lx / float(Mx)
    hy = Ly / float(My)
    hz = Lz / float(Mz)

    V = Lx * Ly * Lz
    M_V = Mx * My * Mz

    rho_r = calc_charge_dens(pos, Z, N, cao, Mx, My, Mz, hx, hy, hz)

    fftw_n = pyfftw.builders.fftn(rho_r)
    rho_k_fft = fftw_n()
    rho_k = np.fft.fftshift(rho_k_fft)

    phi_k = G_k * rho_k

    rho_k_real = np.real(rho_k)
    rho_k_imag = np.imag(rho_k)
    rho_k_sq = rho_k_real * rho_k_real + rho_k_imag * rho_k_imag

    U_f = 0.5 * np.sum(rho_k_sq * G_k) / V

    E_kx, E_ky, E_kz = calc_field(phi_k, kx_v, ky_v, kz_v)

    E_kx_unsh = np.fft.ifftshift(E_kx)
    E_ky_unsh = np.fft.ifftshift(E_ky)
    E_kz_unsh = np.fft.ifftshift(E_kz)

    ifftw_n = pyfftw.builders.ifftn(E_kx_unsh)
    E_x = ifftw_n()
    ifftw_n = pyfftw.builders.ifftn(E_ky_unsh)
    E_y = ifftw_n()
    ifftw_n = pyfftw.builders.ifftn(E_kz_unsh)
    E_z = ifftw_n()

    E_x = M_V * E_x / V
    E_y = M_V * E_y / V
    E_z = M_V * E_z / V

    E_x_r = np.real(E_x)
    E_y_r = np.real(E_y)
    E_z_r = np.real(E_z)

    acc_f = calc_acc_pm(E_x_r, E_y_r, E_z_r, pos, Z, N, cao, Mass, Mx, My, Mz, hx, hy, hz)

    return U_f, acc_f
