"""
Module for handling the Particle-Mesh part of the force and potential calculation.
"""

from numba import jit
from numba.core.types import complex128, float64, int64, Tuple, UniTuple
from numpy import arange, array, exp, imag, mod, pi, real, sin, sqrt, zeros, zeros_like
from numpy.fft import fftshift, ifftshift
from pyfftw.builders import fftn, ifftn


@jit(Tuple((float64[:, :], float64[:, :], float64[:, :, :]))(int64[:], float64[:]), nopython=True)
def create_k_arrays(mesh_sizes, non_zero_box_lengths):
    """Calculate the reciprocal space arrays.

    Parameters
    ----------
    non_zero_box_lengths : numpy.ndarray
        Length of simulation's box in each direction. Note that no element should be equal to 0.0.
        If the dimensionality of the problem is lower than 3, then use 1.0 as the box length for those dimensions.
        Example: 2D non_zero_box_lengths = [Lx, Ly, 1.0].

    mesh_sizes : numpy.ndarray
        Number of mesh points in x,y,z.

    Returns
    -------
    kx_v : numpy.ndarray
       Array of reciprocal space vectors along the x-axis

    ky_v : numpy.ndarray
       Array of reciprocal space vectors along the y-axis

    kz_v : numpy.ndarray
       Array of reciprocal space vectors along the z-axis

    """
    nz_mid = mesh_sizes[2] / 2 if mod(mesh_sizes[2], 2) == 0 else (mesh_sizes[2] - 1) / 2
    ny_mid = mesh_sizes[1] / 2 if mod(mesh_sizes[1], 2) == 0 else (mesh_sizes[1] - 1) / 2
    nx_mid = mesh_sizes[0] / 2 if mod(mesh_sizes[0], 2) == 0 else (mesh_sizes[0] - 1) / 2

    # nx_v = arange(mesh_sizes[0]).reshape((1, mesh_sizes[0]))
    # ny_v = arange(mesh_sizes[1]).reshape((mesh_sizes[1], 1))
    # nz_v = arange(mesh_sizes[2]).reshape((mesh_sizes[2], 1, 1))
    # Dev Note:
    # The above three lines where giving a problem with Numba in Windows only.
    # I replaced them with the ones below. I don't know why it was giving a problem.
    nx_v = zeros((1, mesh_sizes[0]), dtype=int64)
    nx_v[0, :] = arange(mesh_sizes[0])

    ny_v = zeros((mesh_sizes[1], 1), dtype=int64)
    ny_v[:, 0] = arange(mesh_sizes[1])

    nz_v = zeros((mesh_sizes[2], 1, 1), dtype=int64)
    nz_v[:, 0, 0] = arange(mesh_sizes[2])

    two_pi = 2.0 * pi
    kx_v = two_pi * (nx_v - nx_mid) / non_zero_box_lengths[0]
    ky_v = two_pi * (ny_v - ny_mid) / non_zero_box_lengths[1]
    kz_v = two_pi * (nz_v - nz_mid) / non_zero_box_lengths[2]

    return kx_v, ky_v, kz_v


@jit(UniTuple(float64[:, :], 3)(int64[:], int64[:], float64[:]), nopython=True)
def create_k_aliases(aliases, mesh_sizes, non_zero_box_lengths):
    """Calculate the alias arrays of the reciprocal space arrays for anti-aliasing.

    Parameters
    ----------
    aliases : numpy.ndarray, numba.int64
        Number of aliases per dimension.

    mesh_sizes : numpy.ndarray, numba.int64
        Number of mesh points in x,y,z.

    non_zero_box_lengths : numpy.ndarray, numba.float64
        Length of simulation's box in each direction. Note that no element should be equal to 0.0.
        If the dimensionality of the problem is lower than 3, then use 1.0 as the box length for those dimensions.
        Example: 2D non_zero_box_lengths = [Lx, Ly, 1.0].

    Returns
    -------
    kx_M : numpy.ndarray
       Array of aliases for each kx value. Shape=( mesh_size[0], 2 * aliases[0] + 1)

    ky_M : numpy.ndarray
       Array of aliases for each ky value. Shape=( mesh_size[1], 2 * aliases[1] + 1)

    kz_M : numpy.ndarray
       Array of aliases for each kz value. Shape=( mesh_size[2], 2 * aliases[2] + 1)

    """

    nz_mid = mesh_sizes[2] / 2 if mod(mesh_sizes[2], 2) == 0 else (mesh_sizes[2] - 1) / 2
    ny_mid = mesh_sizes[1] / 2 if mod(mesh_sizes[1], 2) == 0 else (mesh_sizes[1] - 1) / 2
    nx_mid = mesh_sizes[0] / 2 if mod(mesh_sizes[0], 2) == 0 else (mesh_sizes[0] - 1) / 2

    two_pi = 2.0 * pi

    kx_M = zeros((mesh_sizes[0], 2 * aliases[0] + 1), dtype=float64)
    ky_M = zeros((mesh_sizes[1], 2 * aliases[1] + 1), dtype=float64)
    kz_M = zeros((mesh_sizes[2], 2 * aliases[2] + 1), dtype=float64)

    for nz in range(mesh_sizes[2]):
        nz_sh = nz - nz_mid
        for mz in range(-aliases[2], aliases[2] + 1):
            kz_M[nz, mz + aliases[2]] = two_pi * (nz_sh + mz * mesh_sizes[2]) / non_zero_box_lengths[2]

    for ny in range(mesh_sizes[1]):
        ny_sh = ny - ny_mid
        for my in range(-aliases[1], aliases[1] + 1):
            ky_M[ny, my + aliases[1]] = two_pi * (ny_sh + my * mesh_sizes[1]) / non_zero_box_lengths[1]

    for nx in range(mesh_sizes[0]):
        nx_sh = nx - nx_mid
        for mx in range(-aliases[0], aliases[0] + 1):
            kx_M[nx, mx + aliases[0]] = two_pi * (nx_sh + mx * mesh_sizes[0]) / non_zero_box_lengths[0]

    return kx_M, ky_M, kz_M


@jit(
    UniTuple(float64, 2)(
        float64, float64, float64, float64[:], float64[:], float64[:], float64[:], int64[:], float64, float64, float64
    ),
    nopython=True,
)
def sum_over_aliases(kx, ky, kz, kx_M, ky_M, kz_M, h_array, p, four_pi, alpha_sq, kappa_sq):

    U_k_sq = 0.0
    U_G_k = 0.0

    # Sum over the aliases
    for mz, kzm in enumerate(kz_M):
        kz_M_arg = 0.5 * kzm * h_array[2]
        U_kz_M = (sin(kz_M_arg) / kz_M_arg) ** p[2] if kz_M_arg != 0.0 else 1.0

        for my, kym in enumerate(ky_M):
            ky_M_arg = 0.5 * kym * h_array[1]
            U_ky_M = (sin(ky_M_arg) / ky_M_arg) ** p[1] if ky_M_arg != 0.0 else 1.0

            for mx, kxm in enumerate(kx_M):
                kx_M_arg = 0.5 * kxm * h_array[0]
                U_kx_M = (sin(kx_M_arg) / kx_M_arg) ** p[0] if kx_M_arg != 0.0 else 1.0

                k_M_sq = kxm * kxm + kym * kym + kzm * kzm

                U_k_M = U_kx_M * U_ky_M * U_kz_M
                U_k_M_sq = U_k_M * U_k_M

                G_k_M = four_pi * exp(-0.25 * (kappa_sq + k_M_sq) / alpha_sq) / (kappa_sq + k_M_sq)

                k_dot_k_M = kx * kxm + ky * kym + kz * kzm

                U_G_k += U_k_M_sq * G_k_M * k_dot_k_M
                U_k_sq += U_k_M_sq

    return U_G_k, U_k_sq


@jit(
    Tuple((float64[:, :, :], float64[:, :], float64[:, :], float64[:, :, :], float64))(
        float64[:], float64[:], int64[:], int64[:], int64[:], float64[:]
    ),
    nopython=True,
)
def force_optimized_green_function(box_lengths, h_array, mesh_sizes, aliases, p, constants):
    """
    Numba'd function to calculate the Optimized Green Function given by eq.(22) of Ref.:cite:`Stern2008`.

    Parameters
    ----------
    box_lengths : numpy.ndarray
        Length of simulation's box in each direction.

    h_array : numpy.ndarray
        Mesh spacings.

    mesh_sizes : numpy.ndarray
        Number of mesh points in x,y,z.

    aliases : numpy.ndarray
        Number of aliases in each direction.

    p : numpy.ndarray
        Array of charge assignment order (cao) for each dimension.

    constants : numpy.ndarray
        Screening parameter, Ewald parameter, :math:`4 \\pi \\eplison_0`.

    Returns
    -------
    G_k : numpy.ndarray
        Optimal Green Function

    PM_err : float
        Error in the force calculation due to the optimized Green's function. eq.(28) of :cite:`Dharuman2017` .

    """
    kappa = constants[0]
    Gew = constants[1]
    fourpie0 = constants[2]

    four_pi = 4.0 * pi if fourpie0 == 1.0 else 4.0 * pi / fourpie0
    two_pi = 2.0 * pi

    mask = box_lengths.nonzero()
    non_zero_box_lengths = array([1.0, 1.0, 1.0], dtype=float64)
    non_zero_box_lengths[mask] = box_lengths[mask].copy()

    kappa_sq = kappa * kappa
    Gew_sq = Gew * Gew

    G_k = zeros((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]))

    PM_err = 0.0

    kx_v, ky_v, kz_v = create_k_arrays(mesh_sizes, non_zero_box_lengths)

    kx_M, ky_M, kz_M = create_k_aliases(aliases, mesh_sizes, non_zero_box_lengths)

    for nz, kz in enumerate(kz_v[:, 0, 0]):
        for ny, ky in enumerate(ky_v[:, 0]):
            for nx, kx in enumerate(kx_v[0, :]):
                k_sq = kx * kx + ky * ky + kz * kz
                if k_sq != 0.0:
                    #
                    # U_k_sq = 0.0
                    # U_G_k = 0.0

                    # Sum over the aliases
                    # for mz in range(-aliases[2], aliases[2] + 1):
                    #     kz_M = two_pi * (nz_sh + mz * mesh_sizes[2]) / non_zero_box_lengths[2]
                    #     kz_M_arg = 0.5 * kz_M * h_array[2]
                    #     U_kz_M = (sin(kz_M_arg) / kz_M_arg) ** p[2] if kz_M_arg != 0.0 else 1.0
                    #
                    #     for my in range(-aliases[1], aliases[1] + 1):
                    #         ky_M = two_pi * (ny_sh + my * mesh_sizes[1]) / non_zero_box_lengths[1]
                    #         ky_M_arg = 0.5 * ky_M * h_array[1]
                    #         U_ky_M = (sin(ky_M_arg) / ky_M_arg) ** p[1] if ky_M_arg != 0.0 else 1.0
                    #
                    #         for mx in range(-aliases[0], aliases[0] + 1):
                    #             kx_M = two_pi * (nx_sh + mx * mesh_sizes[0]) / non_zero_box_lengths[0]
                    #             kx_M_arg = 0.5 * kx_M * h_array[0]
                    #             U_kx_M = (sin(kx_M_arg) / kx_M_arg) ** p[0] if kx_M_arg != 0.0 else 1.0
                    #
                    #             # print(mx, my, mz, kx_M, ky_M, kz_M)
                    #             k_M_sq = kx_M * kx_M + ky_M * ky_M + kz_M * kz_M
                    #
                    #             U_k_M = U_kx_M * U_ky_M * U_kz_M
                    #             U_k_M_sq = U_k_M * U_k_M
                    #
                    #             G_k_M = four_pi * exp(-0.25 * (kappa_sq + k_M_sq) / Gew_sq) / (kappa_sq + k_M_sq)
                    #
                    #             k_dot_k_M = kx * kx_M + ky * ky_M + kz * kz_M
                    #
                    #             U_G_k += U_k_M_sq * G_k_M * k_dot_k_M
                    #             U_k_sq += U_k_M_sq

                    # eq.(22) of Ref.[Dharuman2017]_
                    U_G_k, U_k_sq = sum_over_aliases(
                        kx, ky, kz, kx_M[nx], ky_M[ny], kz_M[nz], h_array, p, four_pi, Gew_sq, kappa_sq
                    )

                    G_k[nz, ny, nx] = U_G_k / ((U_k_sq**2) * k_sq)

                    Gk_hat = four_pi * exp(-0.25 * (kappa_sq + k_sq) / Gew_sq) / (kappa_sq + k_sq)

                    # eq.(28) of Ref.[Dharuman2017]_
                    PM_err += Gk_hat * Gk_hat * k_sq - U_G_k**2 / ((U_k_sq**2) * k_sq)

    PM_err = sqrt(abs(PM_err)) / non_zero_box_lengths.prod() ** (1.0 / len(box_lengths.nonzero()[0]))

    return G_k, kx_v, ky_v, kz_v, PM_err


@jit(float64[:](int64, float64), nopython=True)
def assgnmnt_func(cao, x):
    """
    Calculate the charge assignment function as given in Ref.:cite:`Deserno1998`

    Parameters
    ----------
    cao : int
        Charge assignment order.

    x : float
        Distance to the closest mesh point.

    Returns
    ------
    W : numpy.ndarray
        Charge Assignment Function. Each element is the fraction of the charge on each of the `cao` mesh points
        starting from the far left.

    """
    W = zeros(cao)

    if cao == 1:

        W[0] = 1.0

    elif cao == 2:

        W[0] = 0.5 * (1.0 - 2.0 * x)
        W[1] = 0.5 * (1.0 + 2.0 * x)

    elif cao == 3:

        W[0] = (1.0 - 4.0 * x + 4.0 * x**2) / 8.0
        W[1] = (3.0 - 4.0 * x**2) / 4.0
        W[2] = (1.0 + 4.0 * x + 4.0 * x**2) / 8.0

    elif cao == 4:

        W[0] = (1.0 - 6.0 * x + 12.0 * x**2 - 8.0 * x**3) / 48.0
        W[1] = (23.0 - 30.0 * x - 12.0 * x**2 + 24.0 * x**3) / 48.0
        W[2] = (23.0 + 30.0 * x - 12.0 * x**2 - 24.0 * x**3) / 48.0
        W[3] = (1.0 + 6.0 * x + 12.0 * x**2 + 8.0 * x**3) / 48.0

    elif cao == 5:

        W[0] = (1.0 - 8.0 * x + 24.0 * x**2 - 32.0 * x**3 + 16.0 * x**4) / 384.0
        W[1] = (19.0 - 44.0 * x + 24.0 * x**2 + 16.0 * x**3 - 16.0 * x**4) / 96.0
        W[2] = (115.0 - 120.0 * x**2 + 48.0 * x**4) / 192.0
        W[3] = (19.0 + 44.0 * x + 24.0 * x**2 - 16.0 * x**3 - 16.0 * x**4) / 96.0
        W[4] = (1.0 + 8.0 * x + 24.0 * x**2 + 32.0 * x**3 + 16.0 * x**4) / 384.0

    elif cao == 6:
        W[0] = (1.0 - 10.0 * x + 40.0 * x**2 - 80.0 * x**3 + 80.0 * x**4 - 32.0 * x**5) / 3840.0
        W[1] = (237.0 - 750.0 * x + 840.0 * x**2 - 240.0 * x**3 - 240.0 * x**4 + 160.0 * x**5) / 3840.0
        W[2] = (841.0 - 770.0 * x - 440.0 * x**2 + 560.0 * x**3 + 80.0 * x**4 - 160.0 * x**5) / 1920.0
        W[3] = (841.0 + 770.0 * x - 440.0 * x**2 - 560.0 * x**3 + 80.0 * x**4 + 160.0 * x**5) / 1920.0
        W[4] = (237.0 + 750.0 * x + 840.0 * x**2 + 240.0 * x**3 - 240.0 * x**4 - 160.0 * x**5) / 3840.0
        W[5] = (1.0 + 10.0 * x + 40.0 * x**2 + 80.0 * x**3 + 80.0 * x**4 + 32.0 * x**5) / 3840.0

    elif cao == 7:

        W[0] = (
            1.0 - 12.0 * x + 60.0 * x * 2 - 160.0 * x**3 + 240.0 * x**4 - 192.0 * x**5 + 64.0 * x**6
        ) / 46080.0

        W[1] = (
            361.0 - 1416.0 * x + 2220.0 * x**2 - 1600.0 * x**3 + 240.0 * x**4 + 384.0 * x**5 - 192.0 * x**6
        ) / 23040.0

        W[2] = (
            10543.0 - 17340.0 * x + 4740.0 * x**2 + 6880.0 * x**3 - 4080.0 * x**4 - 960.0 * x**5 + 960.0 * x**6
        ) / 46080.0

        W[3] = (5887.0 - 4620.0 * x**2 + 1680.0 * x**4 - 320.0 * x**6) / 11520.0

        W[4] = (
            10543.0 + 17340.0 * x + 4740.0 * x**2 - 6880.0 * x**3 - 4080.0 * x**4 + 960.0 * x**5 + 960.0 * x**6
        ) / 46080.0

        W[5] = (
            361.0 + 1416.0 * x + 2220.0 * x**2 + 1600.0 * x**3 + 240.0 * x**4 - 384.0 * x**5 - 192.0 * x**6
        ) / 23040.0

        W[6] = (
            1.0 + 12.0 * x + 60.0 * x**2 + 160.0 * x**3 + 240.0 * x**4 + 192.0 * x**5 + 64.0 * x**6
        ) / 46080.0

    return W


@jit(float64[:, :, :](float64[:, :], float64[:], int64[:], int64[:], float64[:]), nopython=True)
def calc_charge_dens(pos, charges, cao, mesh_sz, h_array):
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

    cao: numpy.ndarray
        Charge assignment order.

    Returns
    -------
    rho_r: numpy.ndarray
        Charge density distributed on mesh.

    """

    rho_r = zeros((mesh_sz[2], mesh_sz[1], mesh_sz[0]), dtype=float64)
    pshift = zeros(len(cao), dtype=int64)
    mid = zeros(len(cao), dtype=float64)

    for ic, p in enumerate(cao):
        # Mid point calculation
        if p % 2 == 0:
            # Choose the midpoint between the two closest mesh point to the particle's position
            mid[ic] = 0.5
            pshift[ic] = int(float(p) / 2.0 - 1)
        else:
            # Choose the mesh point closes to the particle
            mid[ic] = 0.0
            pshift[ic] = int(float(p) / 2.0)

    for ipart in range(len(charges)):

        # ix = x-coord of the (left) closest mesh point
        # (ix + 0.5)*h_array[0] = midpoint between the two mesh points closest to the particle
        # x = the difference between the particle's position and the midpoint
        # Rescale

        ix = round(pos[ipart, 0] / h_array[0]) * (cao[0] % 2 != 0) + int(pos[ipart, 0] / h_array[0]) * (cao[0] % 2 == 0)
        x = pos[ipart, 0] / h_array[0] - (ix + mid[0])

        iy = round(pos[ipart, 1] / h_array[1]) * (cao[1] % 2 != 0) + int(pos[ipart, 1] / h_array[1]) * (cao[1] % 2 == 0)
        y = pos[ipart, 1] / h_array[1] - (iy + mid[1])

        iz = round(pos[ipart, 2] / h_array[2]) * (cao[2] % 2 != 0) + int(pos[ipart, 2] / h_array[2]) * (cao[2] % 2 == 0)
        z = pos[ipart, 2] / h_array[2] - (iz + mid[2])
        # The above branchless programming is because int gives the integer value and NOT the closest integer

        wx = assgnmnt_func(cao[0], x)
        wy = assgnmnt_func(cao[1], y)
        wz = assgnmnt_func(cao[2], z)

        izn = iz - pshift[2]  # min. index along z-axis

        for g in range(cao[2]):

            # if izn < 0:
            #   r_g = izn + mesh_sz[2]
            # elif izn > (mesh_sz[2] - 1):
            #     r_g = izn - mesh_sz[2]
            # else:
            #     r_g = izn

            r_g = izn + mesh_sz[2] * (izn < 0) - mesh_sz[2] * (izn > (mesh_sz[2] - 1))
            iyn = iy - pshift[1]  # min. index along y-axis

            for i in range(cao[1]):

                r_i = iyn + mesh_sz[1] * (iyn < 0) - mesh_sz[1] * (iyn > (mesh_sz[1] - 1))

                # if iyn < 0:
                #     r_i = iyn + mesh_sz[1]
                # elif iyn > (mesh_sz[1] - 1):
                #     r_i = iyn - mesh_sz[1]
                # else:
                #     r_i = iyn

                ixn = ix - pshift[0]  # min. index along x-axis

                for j in range(cao[0]):
                    r_j = ixn + mesh_sz[0] * (ixn < 0) - mesh_sz[0] * (ixn > (mesh_sz[0] - 1))

                    # if ixn < 0:
                    #     r_j = ixn + mesh_sz[0]
                    # elif ixn > (mesh_sz[0] - 1):
                    #     r_j = ixn - mesh_sz[0]
                    # else:
                    #     r_j = ixn

                    rho_r[r_g, r_i, r_j] += charges[ipart] * wz[g] * wy[i] * wx[j]

                    ixn += 1 * (mesh_sz[0] > 1)  # Do not increase the index if there is only 1 point mesh

                iyn += 1 * (mesh_sz[1] > 1)  # Do not increase the index if there is only 1 point mesh

            izn += 1 * (mesh_sz[2] > 1)  # Do not increase the index if there is only 1 point mesh

    return rho_r


@jit(
    UniTuple(complex128[:, :, :], 3)(complex128[:, :, :], float64[:, :], float64[:, :], float64[:, :, :]),
    nopython=True,
)
def calc_field(phi_k, kx_v, ky_v, kz_v):
    """
    Numba'd function that calculates the Electric field in Fourier space.

    Parameters
    ----------
    phi_k : numpy.ndarray, numba.complex128
        3D array of the Potential.

    kx_v : numpy.ndarray, numba.float64
        2D array containing the values of kx.

    ky_v : numpy.ndarray, numba.float64
        2D array containing the values of ky.

    kz_v : numpy.ndarray, numba.float64
        3D array containing the values of kz.

    Returns
    -------
    E_kx : numpy.ndarray, numba.complex128
       Electric Field along kx-axis.

    E_ky : numpy.ndarray, numba.complex128
       Electric Field along ky-axis.

    E_kz : numpy.ndarray, numba.complex128
       Electric Field along kz-axis.

    """

    E_kx = -1j * kx_v * phi_k
    E_ky = -1j * ky_v * phi_k
    E_kz = -1j * kz_v * phi_k

    return E_kx, E_ky, E_kz


@jit(
    float64[:, :](
        float64[:, :, :],
        float64[:, :, :],
        float64[:, :, :],
        float64[:, :],
        float64[:],
        int64[:],
        float64[:],
        int64[:],
        float64[:],
    ),
    nopython=True,
)
def calc_acc_pm(E_x_r, E_y_r, E_z_r, pos, charges, cao, masses, mesh_sz, h_array):
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

    cao : int
        Charge assignment order.

    masses : numpy.ndarray
        Particles' masses.


    Returns
    -------

    acc : numpy.ndarray
          Acceleration from Electric Field.

    """
    E_x_p = zeros_like(charges)
    E_y_p = zeros_like(charges)
    E_z_p = zeros_like(charges)

    acc = zeros_like(pos)

    pshift = zeros(len(cao), dtype=int64)
    mid = zeros(len(cao), dtype=float64)

    for ic, p in enumerate(cao):
        # Mid point calculation
        if p % 2 == 0:
            # Choose the midpoint between the two closest mesh point to the particle's position
            mid[ic] = 0.5
            # Number of points to the left of the chosen one
            pshift[ic] = int(float(p) / 2.0 - 1)
        else:
            # Choose the mesh point closes to the particle
            mid[ic] = 0.0
            # Number of points to the left of the chosen one
            pshift[ic] = int(float(p) / 2.0)

    for ipart in range(len(charges)):

        ix = round(pos[ipart, 0] / h_array[0]) * (cao[0] % 2 != 0) + int(pos[ipart, 0] / h_array[0]) * (cao[0] % 2 == 0)
        x = pos[ipart, 0] / h_array[0] - (ix + mid[0])

        iy = round(pos[ipart, 1] / h_array[1]) * (cao[1] % 2 != 0) + int(pos[ipart, 1] / h_array[1]) * (cao[1] % 2 == 0)
        y = pos[ipart, 1] / h_array[1] - (iy + mid[1])

        iz = round(pos[ipart, 2] / h_array[2]) * (cao[2] % 2 != 0) + int(pos[ipart, 2] / h_array[2]) * (cao[2] % 2 == 0)
        z = pos[ipart, 2] / h_array[2] - (iz + mid[2])
        # The above branchless programming is because int gives the integer value and NOT the closest integer

        wx = assgnmnt_func(cao[0], x)
        wy = assgnmnt_func(cao[1], y)
        wz = assgnmnt_func(cao[2], z)

        izn = iz - pshift[2]  # min. index along z-axis

        for g in range(cao[2]):
            #
            # if izn < 0:
            #     r_g = izn + mesh_sz[2]
            # elif izn > (mesh_sz[2] - 1):
            #     r_g = izn - mesh_sz[2]
            # else:
            #     r_g = izn

            r_g = izn + mesh_sz[2] * (izn < 0) - mesh_sz[2] * (izn > (mesh_sz[2] - 1))

            iyn = iy - pshift[1]  # min. index along y-axis

            for i in range(cao[1]):

                # if iyn < 0:
                #     r_i = iyn + mesh_sz[1]
                # elif iyn > (mesh_sz[1] - 1):
                #     r_i = iyn - mesh_sz[1]
                # else:
                #     r_i = iyn
                r_i = iyn + mesh_sz[1] * (iyn < 0) - mesh_sz[1] * (iyn > (mesh_sz[1] - 1))

                ixn = ix - pshift[0]  # min. index along x-axis

                for j in range(cao[0]):
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
@jit(
    Tuple((float64, float64[:, :]))(
        float64[:, :],
        float64[:],
        float64[:],
        int64[:],
        float64[:],
        float64,
        float64,
        float64[:, :, :],
        float64[:, :],
        float64[:, :],
        float64[:, :, :],
        int64[:],
    ),
    nopython=False,
    forceobj=True,  # This is needed so that it doesn't throw an error nor warning
)
def update(pos, charges, masses, mesh_sizes, mesh_spacings, mesh_volume, box_volume, G_k, kx_v, ky_v, kz_v, cao):
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

    # Mesh spacings = h_x, h_y, h_z
    # mesh_spacings = box_lengths / mesh_sizes
    # Calculate charge density on mesh
    rho_r = calc_charge_dens(pos, charges, cao, mesh_sizes, mesh_spacings)
    # Prepare for fft
    fftw_n = fftn(rho_r)
    # Calculate fft
    rho_k_fft = fftw_n()

    # Shift the DC value at the center of the ndarray
    rho_k = fftshift(rho_k_fft)

    # Potential from Poisson eq.
    phi_k = G_k * rho_k

    # Charge density
    rho_k_real = real(rho_k)
    rho_k_imag = imag(rho_k)
    rho_k_sq = rho_k_real * rho_k_real + rho_k_imag * rho_k_imag

    # Long range part of the potential
    U_f = 0.5 * (rho_k_sq * G_k).sum() / box_volume

    # Calculate the Electric field's component on the mesh
    E_kx, E_ky, E_kz = calc_field(phi_k, kx_v, ky_v, kz_v)

    # Prepare for fft. Shift the DC value back to its original position that is [0, 0, 0]
    E_kx_unsh = ifftshift(E_kx)
    E_ky_unsh = ifftshift(E_ky)
    E_kz_unsh = ifftshift(E_kz)

    # Prepare and compute IFFT
    ifftw_n = ifftn(E_kx_unsh)
    E_x = ifftw_n()
    ifftw_n = ifftn(E_ky_unsh)
    E_y = ifftw_n()
    ifftw_n = ifftn(E_kz_unsh)
    E_z = ifftw_n()

    # I am worried that this normalization is not needed
    E_x_r = real(E_x) / mesh_volume
    E_y_r = real(E_y) / mesh_volume
    E_z_r = real(E_z) / mesh_volume

    acc_f = calc_acc_pm(E_x_r, E_y_r, E_z_r, pos, charges, cao, masses, mesh_sizes, mesh_spacings)

    return U_f, acc_f
