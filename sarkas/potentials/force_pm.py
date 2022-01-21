"""
Module for handling the Particle-Mesh part of the force and potential calculation.
"""

import numpy as np
from numba import jit, njit
import pyfftw

# These "ignore" are needed because numba does not support pyfftw yet
from numba.core.errors import NumbaWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


@njit
def force_optimized_green_function(box_lengths, mesh_sizes, aliases, p, constants):
    """
    Calculate the Optimized Green Function given by eq.(22) of Ref. [Stern2008].

    Parameters
    ----------

    mesh_sizes : numpy.ndarray
        number of mesh points in x,y,z

    aliases : numpy.ndarray
        number of aliases in each direction

    box_lengths : numpy.ndarray
        Length of simulation's box in each direction

    p : int
        Charge assignment order (CAO)

    constants : numpy.ndarray
        Screening parameter, Ewald parameter, 4 pi eps0.

    Returns
    -------
    G_k : numpy.ndarray
        optimal Green Function

    kx_v : numpy.ndarray
       array of reciprocal space vectors along the x-axis

    ky_v : numpy.ndarray
       array of reciprocal space vectors along the y-axis

    kz_v : numpy.ndarray
       array of reciprocal space vectors along the z-axis

    PM_err : float
        Error in the force calculation due to the optimized Green's function. eq.(28) of :cite:`Dharuman2017` .

    PP_err : float
        Error in the force calculation due to the distance cutoff. eq.(30) of :cite:`Stern2008` .

    """
    kappa = constants[0]
    Gew = constants[1]
    fourpie0 = constants[2]

    h_array = box_lengths / mesh_sizes

    kappa_sq = kappa * kappa
    Gew_sq = Gew * Gew

    G_k = np.zeros((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]))

    nz_mid = mesh_sizes[2] / 2 if np.mod(mesh_sizes[2], 2) == 0 else (mesh_sizes[2] - 1) / 2
    ny_mid = mesh_sizes[1] / 2 if np.mod(mesh_sizes[1], 2) == 0 else (mesh_sizes[1] - 1) / 2
    nx_mid = mesh_sizes[0] / 2 if np.mod(mesh_sizes[0], 2) == 0 else (mesh_sizes[0] - 1) / 2

    # nx_v = np.arange(mesh_sizes[0]).reshape((1, mesh_sizes[0]))
    # ny_v = np.arange(mesh_sizes[1]).reshape((mesh_sizes[1], 1))
    # nz_v = np.arange(mesh_sizes[2]).reshape((mesh_sizes[2], 1, 1))
    # Dev Note:
    # The above three lines where giving a problem with Numba in Windows only.
    # I replaced them with the ones below. I don't know why it was giving a problem.
    nx_v = np.zeros((1, mesh_sizes[0]), dtype=np.int64)
    nx_v[0, :] = np.arange(mesh_sizes[0])

    ny_v = np.zeros((mesh_sizes[1], 1), dtype=np.int64)
    ny_v[:, 0] = np.arange(mesh_sizes[1])

    nz_v = np.zeros((mesh_sizes[2], 1, 1), dtype=np.int64)
    nz_v[:, 0, 0] = np.arange(mesh_sizes[2])

    kx_v = 2.0 * np.pi * (nx_v - nx_mid) / box_lengths[0]
    ky_v = 2.0 * np.pi * (ny_v - ny_mid) / box_lengths[1]
    kz_v = 2.0 * np.pi * (nz_v - nz_mid) / box_lengths[2]

    PM_err = 0.0

    four_pi = 4.0 * np.pi if fourpie0 == 1.0 else 4.0 * np.pi / fourpie0
    two_pi = 2.0 * np.pi

    for nz in range(mesh_sizes[2]):
        nz_sh = nz - nz_mid
        kz = two_pi * nz_sh / box_lengths[2]

        for ny in range(mesh_sizes[1]):
            ny_sh = ny - ny_mid
            ky = two_pi * ny_sh / box_lengths[1]

            for nx in range(mesh_sizes[0]):
                nx_sh = nx - nx_mid
                kx = two_pi * nx_sh / box_lengths[0]

                k_sq = kx * kx + ky * ky + kz * kz

                if k_sq != 0.0:

                    U_k_sq = 0.0
                    U_G_k = 0.0

                    # Sum over the aliases
                    for mz in range(-aliases[2], aliases[2] + 1):
                        kz_M = two_pi * (nz_sh + mz * mesh_sizes[2]) / box_lengths[2]
                        U_kz_M = np.sin(0.5 * kz_M * h_array[2]) / (0.5 * kz_M * h_array[2]) if kz_M != 0.0 else 1.0

                        for my in range(-aliases[1], aliases[1] + 1):
                            ky_M = two_pi * (ny_sh + my * mesh_sizes[1]) / box_lengths[1]
                            U_ky_M = np.sin(0.5 * ky_M * h_array[1]) / (0.5 * ky_M * h_array[1]) if ky_M != 0.0 else 1.0

                            for mx in range(-aliases[0], aliases[0] + 1):
                                kx_M = two_pi * (nx_sh + mx * mesh_sizes[0]) / box_lengths[0]
                                U_kx_M = (
                                    np.sin(0.5 * kx_M * h_array[0]) / (0.5 * kx_M * h_array[0]) if kx_M != 0.0 else 1.0
                                )

                                k_M_sq = kx_M * kx_M + ky_M * ky_M + kz_M * kz_M

                                U_k_M = (U_kx_M * U_ky_M * U_kz_M) ** p
                                U_k_M_sq = U_k_M * U_k_M

                                G_k_M = four_pi * np.exp(-0.25 * (kappa_sq + k_M_sq) / Gew_sq) / (kappa_sq + k_M_sq)

                                k_dot_k_M = kx * kx_M + ky * ky_M + kz * kz_M

                                U_G_k += U_k_M_sq * G_k_M * k_dot_k_M
                                U_k_sq += U_k_M_sq

                    # eq.(22) of Ref.[Dharuman2017]_
                    G_k[nz, ny, nx] = U_G_k / ((U_k_sq ** 2) * k_sq)
                    Gk_hat = four_pi * np.exp(-0.25 * (kappa_sq + k_sq) / Gew_sq) / (kappa_sq + k_sq)

                    # eq.(28) of Ref.[Dharuman2017]_
                    PM_err += Gk_hat * Gk_hat * k_sq - U_G_k ** 2 / ((U_k_sq ** 2) * k_sq)

    PM_err = np.sqrt(PM_err) / np.prod(box_lengths) ** (1.0 / 3.0)

    return G_k, kx_v, ky_v, kz_v, PM_err


@njit
def assgnmnt_func(cao, x):
    """ 
    Calculate the charge assignment function as given in Ref. [Deserno1998].
    
    Parameters
    ----------
    cao : int
        Charge assignment order.

    x : float
        Distance to closest mesh point if cao is even.
    
    Returns
    ------
    W : numpy.ndarray
        Charge Assignment Function. 

    """
    W = np.zeros(cao)

    if cao == 1:

        W[0] = 1

    elif cao == 2:

        W[0] = 0.5 * (1.0 - 2.0 * x)
        W[1] = 0.5 * (1.0 + 2.0 * x)

    elif cao == 3:

        W[0] = (1.0 - 4.0 * x + 4.0 * x ** 2) / 8.0
        W[1] = (3.0 - 4.0 * x ** 2) / 4.0
        W[2] = (1.0 + 4.0 * x + 4.0 * x ** 2) / 8.0

    elif cao == 4:

        W[0] = (1.0 - 6.0 * x + 12.0 * x ** 2 - 8.0 * x ** 3) / 48.0
        W[1] = (23.0 - 30.0 * x - 12.0 * x ** 2 + 24.0 * x ** 3) / 48.0
        W[2] = (23.0 + 30.0 * x - 12.0 * x ** 2 - 24.0 * x ** 3) / 48.0
        W[3] = (1.0 + 6.0 * x + 12.0 * x ** 2 + 8.0 * x ** 3) / 48.0

    elif cao == 5:

        W[0] = (1.0 - 8.0 * x + 24.0 * x ** 2 - 32.0 * x ** 3 + 16.0 * x ** 4) / 384.0
        W[1] = (19.0 - 44.0 * x + 24.0 * x ** 2 + 16.0 * x ** 3 - 16.0 * x ** 4) / 96.0
        W[2] = (115.0 - 120.0 * x ** 2 + 48.0 * x ** 4) / 192.0
        W[3] = (19.0 + 44.0 * x + 24.0 * x ** 2 - 16.0 * x ** 3 - 16.0 * x ** 4) / 96.0
        W[4] = (1.0 + 8.0 * x + 24.0 * x ** 2 + 32.0 * x ** 3 + 16.0 * x ** 4) / 384.0

    elif cao == 6:
        W[0] = (1.0 - 10.0 * x + 40.0 * x ** 2 - 80.0 * x ** 3 + 80.0 * x ** 4 - 32.0 * x ** 5) / 3840.0
        W[1] = (237.0 - 750.0 * x + 840.0 * x ** 2 - 240.0 * x ** 3 - 240.0 * x ** 4 + 160.0 * x ** 5) / 3840.0
        W[2] = (841.0 - 770.0 * x - 440.0 * x ** 2 + 560.0 * x ** 3 + 80.0 * x ** 4 - 160.0 * x ** 5) / 1920.0
        W[3] = (841.0 + 770.0 * x - 440.0 * x ** 2 - 560.0 * x ** 3 + 80.0 * x ** 4 + 160.0 * x ** 5) / 1920.0
        W[4] = (237.0 + 750.0 * x + 840.0 * x ** 2 + 240.0 * x ** 3 - 240.0 * x ** 4 - 160.0 * x ** 5) / 3840.0
        W[5] = (1.0 + 10.0 * x + 40.0 * x ** 2 + 80.0 * x ** 3 + 80.0 * x ** 4 + 32.0 * x ** 5) / 3840.0

    elif cao == 7:

        W[0] = (
            1.0 - 12.0 * x + 60.0 * x * 2 - 160.0 * x ** 3 + 240.0 * x ** 4 - 192.0 * x ** 5 + 64.0 * x ** 6
        ) / 46080.0

        W[1] = (
            361.0 - 1416.0 * x + 2220.0 * x ** 2 - 1600.0 * x ** 3 + 240.0 * x ** 4 + 384.0 * x ** 5 - 192.0 * x ** 6
        ) / 23040.0

        W[2] = (
            10543.0 - 17340.0 * x + 4740.0 * x ** 2 + 6880.0 * x ** 3 - 4080.0 * x ** 4 - 960.0 * x ** 5 + 960.0 * x ** 6
        ) / 46080.0

        W[3] = (5887.0 - 4620.0 * x ** 2 + 1680.0 * x ** 4 - 320.0 * x ** 6) / 11520.0

        W[4] = (
            10543.0 + 17340.0 * x + 4740.0 * x ** 2 - 6880.0 * x ** 3 - 4080.0 * x ** 4 + 960.0 * x ** 5 + 960.0 * x ** 6
        ) / 46080.0

        W[5] = (
            361.0 + 1416.0 * x + 2220.0 * x ** 2 + 1600.0 * x ** 3 + 240.0 * x ** 4 - 384.0 * x ** 5 - 192.0 * x ** 6
        ) / 23040.0

        W[6] = (
            1.0 + 12.0 * x + 60.0 * x ** 2 + 160.0 * x ** 3 + 240.0 * x ** 4 + 192.0 * x ** 5 + 64.0 * x ** 6
        ) / 46080.0

    return W


@njit
def calc_charge_dens(pos, charges, N, cao, mesh_sz, h_array):
    """ 
    Assigns Charges to Mesh Points.

    Parameters
    ----------
    h_array: numpy.ndarray
        Distances between mesh points per dimension.

    mesh_sz: numpy.ndarray
        Mesh points per direction.

    pos: numpy.ndarray
        Particles' positions.
    
    charges: numpy.ndarray
        Particles' charges.
    
    N: int
        Number of particles.

    cao: int
        Charge assignment order.

    Returns
    -------
    rho_r: numpy.ndarray
        Charge density distributed on mesh.

    """

    rho_r = np.zeros((mesh_sz[2], mesh_sz[1], mesh_sz[0]))

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
        # (ix + 0.5)*h_array[0] = midpoint between the two mesh points closest to the particle
        # x = the difference between the particle's position and the midpoint
        # Rescale

        ix = int(pos[ipart, 0] / h_array[0])
        x = pos[ipart, 0] / h_array[0] - (ix + mid)

        iy = int(pos[ipart, 1] / h_array[1])
        y = pos[ipart, 1] / h_array[1] - (iy + mid)

        iz = int(pos[ipart, 2] / h_array[2])
        z = pos[ipart, 2] / h_array[2] - (iz + mid)

        wx = assgnmnt_func(cao, x)
        wy = assgnmnt_func(cao, y)
        wz = assgnmnt_func(cao, z)

        izn = iz - pshift  # min. index along z-axis

        for g in range(cao):

            # if izn < 0:
            r_g = izn + mesh_sz[2] * (izn < 0) - mesh_sz[2] * (izn > (mesh_sz[2] - 1))
            # elif izn > (mesh_sz[2] - 1):
            #     r_g = izn - mesh_sz[2]
            # else:
            #     r_g = izn

            iyn = iy - pshift  # min. index along y-axis

            for i in range(cao):

                r_i = iyn + mesh_sz[1] * (iyn < 0) - mesh_sz[1] * (iyn > (mesh_sz[1] - 1))

                # if iyn < 0:
                #     r_i = iyn + mesh_sz[1]
                # elif iyn > (mesh_sz[1] - 1):
                #     r_i = iyn - mesh_sz[1]
                # else:
                #     r_i = iyn

                ixn = ix - pshift  # min. index along x-axis

                for j in range(cao):

                    r_j = ixn + mesh_sz[0] * (ixn < 0) - mesh_sz[0] * (ixn > (mesh_sz[0] - 1))

                    # if ixn < 0:
                    #     r_j = ixn + mesh_sz[0]
                    # elif ixn > (mesh_sz[0] - 1):
                    #     r_j = ixn - mesh_sz[0]
                    # else:
                    #     r_j = ixn

                    rho_r[r_g, r_i, r_j] += charges[ipart] * wz[g] * wy[i] * wx[j]

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
    phi_k : numpy.ndarray
        3D array of the Potential.

    kx_v : numpy.ndarray
        3D array containing the values of kx.

    ky_v : numpy.ndarray
        3D array containing the values of ky.

    kz_v : numpy.ndarray
        3D array containing the values of kz.
    
    Returns
    -------
    E_kx : numpy.ndarray
       Electric Field along kx-axis.

    E_ky : numpy.ndarray
       Electric Field along ky-axis.

    E_kz : numpy.ndarray
       Electric Field along kz-axis.
    
    """

    E_kx = -1j * kx_v * phi_k
    E_ky = -1j * ky_v * phi_k
    E_kz = -1j * kz_v * phi_k

    return E_kx, E_ky, E_kz


@njit
def calc_acc_pm(E_x_r, E_y_r, E_z_r, pos, charges, N, cao, masses, mesh_sz, h_array):
    """ 
    Calculates the long range part of particles' accelerations. 
    
    Parameters
    ----------
    E_x_r : numpy.ndarray
        Electric field along x-axis.

    E_y_r : numpy.ndarray
        Electric field along y-axis.
    
    E_z_r : numpy.ndarray
        Electric field along z-axis.
    
    pos : numpy.ndarray
        Particles' positions.
    
    charges : numpy.ndarray
        Particles' charges.
    
    N : int
        Number of particles.

    cao : int
        Charge assignment order.
    
    masses : numpy.ndarray
        Particles' masses.


    Returns
    -------

    acc : numpy.ndarray
          Acceleration from Electric Field.

    """
    E_x_p = np.zeros(N)
    E_y_p = np.zeros(N)
    E_z_p = np.zeros(N)

    acc = np.zeros_like(pos)

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

        ix = int(pos[ipart, 0] / h_array[0])
        x = pos[ipart, 0] / h_array[0] - (ix + mid)

        iy = int(pos[ipart, 1] / h_array[1])
        y = pos[ipart, 1] / h_array[1] - (iy + mid)

        iz = int(pos[ipart, 2] / h_array[2])
        z = pos[ipart, 2] / h_array[2] - (iz + mid)

        wx = assgnmnt_func(cao, x)
        wy = assgnmnt_func(cao, y)
        wz = assgnmnt_func(cao, z)

        izn = iz - pshift  # min. index along z-axis

        for g in range(cao):
            #
            # if izn < 0:
            #     r_g = izn + mesh_sz[2]
            # elif izn > (mesh_sz[2] - 1):
            #     r_g = izn - mesh_sz[2]
            # else:
            #     r_g = izn

            r_g = izn + mesh_sz[2] * (izn < 0) - mesh_sz[2] * (izn > (mesh_sz[2] - 1))

            iyn = iy - pshift  # min. index along y-axis

            for i in range(cao):

                # if iyn < 0:
                #     r_i = iyn + mesh_sz[1]
                # elif iyn > (mesh_sz[1] - 1):
                #     r_i = iyn - mesh_sz[1]
                # else:
                #     r_i = iyn
                r_i = iyn + mesh_sz[1] * (iyn < 0) - mesh_sz[1] * (iyn > (mesh_sz[1] - 1))

                ixn = ix - pshift  # min. index along x-axis

                for j in range(cao):

                    r_j = ixn + mesh_sz[0] * (ixn < 0) - mesh_sz[0] * (ixn > (mesh_sz[0] - 1))

                    # if ixn < 0:
                    #     r_j = ixn + mesh_sz[0]
                    # elif ixn > (mesh_sz[0] - 1):
                    #     r_j = ixn - mesh_sz[0]
                    # else:
                    #     r_j = ixn

                    q_over_m = charges[ipart] / masses[ipart]
                    E_x_p[ipart] += q_over_m * E_x_r[r_g, r_i, r_j] * wz[g] * wy[i] * wx[j]
                    E_y_p[ipart] += q_over_m * E_y_r[r_g, r_i, r_j] * wz[g] * wy[i] * wx[j]
                    E_z_p[ipart] += q_over_m * E_z_r[r_g, r_i, r_j] * wz[g] * wy[i] * wx[j]

                    ixn += 1

                iyn += 1

            izn += 1

    acc[:, 0] = E_x_p
    acc[:, 1] = E_y_p
    acc[:, 2] = E_z_p

    return acc


# FFTW version
@jit  # Numba does not support pyfftw yet, however, this decorator still speeds up the function.
def update(pos, charges, masses, mesh_sizes, box_lengths, G_k, kx_v, ky_v, kz_v, cao):
    """ 
    Calculate the long range part of particles' accelerations.

    Parameters
    ----------
    box_lengths: numpy.ndarray
        Box length in each direction.

    mesh_sizes: numpy.ndarray
        Mesh points per direction.

    pos: numpy.ndarray
        Particles' positions.

    charges: numpy.ndarray
        Particles' charges.

    masses: numpy.ndarray
        Particles' masses.

    G_k : numpy.ndarray
        Optimized Green's function.

    kx_v : numpy.ndarray
        Array of kx values.

    ky_v : numpy.ndarray
        Array of ky values.

    kz_v : numpy.ndarray
        Array of kz values.
    
    cao : int
        Charge order parameter.

    Returns
    -------
    U_f : float
        Long range part of the potential.

    acc_f : numpy.ndarray
        Long range part of particles' accelerations.

    """
    # number of particles
    N = pos.shape[0]
    # Mesh spacings = h_x, h_y, h_z
    mesh_spacings = box_lengths / mesh_sizes
    # Calculate charge density on mesh
    rho_r = calc_charge_dens(pos, charges, N, cao, mesh_sizes, mesh_spacings)
    # Prepare for fft
    fftw_n = pyfftw.builders.fftn(rho_r)
    # Calculate fft
    rho_k_fft = fftw_n()

    # Shift the DC value at the center of the ndarray
    rho_k = np.fft.fftshift(rho_k_fft)

    # Potential from Poisson eq.
    phi_k = G_k * rho_k

    # Charge density
    rho_k_real = np.real(rho_k)
    rho_k_imag = np.imag(rho_k)
    rho_k_sq = rho_k_real * rho_k_real + rho_k_imag * rho_k_imag

    # Long range part of the potential
    U_f = 0.5 * np.sum(rho_k_sq * G_k) / np.prod(box_lengths)

    # Calculate the Electric field's component on the mesh
    E_kx, E_ky, E_kz = calc_field(phi_k, kx_v, ky_v, kz_v)

    # Prepare for fft. Shift the DC value back to its original position that is [0, 0, 0]
    E_kx_unsh = np.fft.ifftshift(E_kx)
    E_ky_unsh = np.fft.ifftshift(E_ky)
    E_kz_unsh = np.fft.ifftshift(E_kz)

    # Prepare and compute IFFT
    ifftw_n = pyfftw.builders.ifftn(E_kx_unsh)
    E_x = ifftw_n()
    ifftw_n = pyfftw.builders.ifftn(E_ky_unsh)
    E_y = ifftw_n()
    ifftw_n = pyfftw.builders.ifftn(E_kz_unsh)
    E_z = ifftw_n()

    # I am worried that this normalization is not needed
    E_x_r = np.real(E_x) / np.prod(mesh_spacings)
    E_y_r = np.real(E_y) / np.prod(mesh_spacings)
    E_z_r = np.real(E_z) / np.prod(mesh_spacings)

    acc_f = calc_acc_pm(E_x_r, E_y_r, E_z_r, pos, charges, N, cao, masses, mesh_sizes, mesh_spacings)

    return U_f, acc_f
