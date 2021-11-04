"""
Module for handling Particle-Particle interaction.
"""

import numpy as np
from numba import njit


@njit
def update_0D(pos, id_ij, mass_ij, Lv, rc, potential_matrix, force, measure, rdf_hist):
    """
    Updates particles' accelerations when the cutoff radius :math: `r_c` is half the box's length, :math: `r_c = L/2`
    For no sub-cell. All ptcls within :math: `r_c = L/2` participate for force calculation. Cost ~ O(N^2)

    Parameters
    ----------
    force: func
        Force function.

    potential_matrix: array
        Potential parameters.

    rc: float
        Cut-off radius.

    Lv: array
        Array of box sides' lentgh.

    mass_ij: array
        Mass of each particle.

    id_ij: array
        Id of each particle

    pos: array
        Particles' positions.

    measure : bool
        Boolean for rdf calculation.

    rdf_hist : array
        Radial Distribution function array.

    Returns
    -------
    U_s_r : array
        Potential.

    acc_s_r : array
        Particles' accelerations.

    """
    L = Lv[0]
    Lh = L / 2.0
    N = pos.shape[0]  # Number of particles

    U_s_r = 0.0  # Short-ranges potential energy accumulator
    acc_s_r = np.zeros_like(pos)  # Vector of accelerations

    rdf_nbins = rdf_hist.shape[0]
    dr_rdf = L / float(2.0 * rdf_nbins)

    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]

            if dx >= Lh:
                dx = L - dx
            elif dx <= -Lh:
                dx = L + dx

            if dy >= Lh:
                dy = L - dy
            elif dy <= -Lh:

                dy = L + dy

            if dz >= Lh:
                dz = L - dz
            elif dz <= -Lh:
                dz = L + dz

            # Compute distance between particles i and j
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            if measure and int(r / dr_rdf) < rdf_nbins:
                rdf_hist[int(r / dr_rdf), id_ij[i], id_ij[j]] += 1

            if 0 < r < rc:
                id_i = id_ij[i]
                id_j = id_ij[j]
                mass_i = mass_ij[i]
                mass_j = mass_ij[j]

                p_matrix = potential_matrix[:, id_i, id_j]
                # Compute the short-ranged force
                pot, fr = force(r, p_matrix)
                U_s_r += pot

                # Update the acceleration for i particles in each dimension

                acc_ix = dx * fr / mass_i
                acc_iy = dy * fr / mass_i
                acc_iz = dz * fr / mass_i

                acc_jx = dx * fr / mass_j
                acc_jy = dy * fr / mass_j
                acc_jz = dz * fr / mass_j

                acc_s_r[i, 0] += acc_ix
                acc_s_r[i, 1] += acc_iy
                acc_s_r[i, 2] += acc_iz

                # Apply Newton's 3rd law to update acceleration on j particles
                acc_s_r[j, 0] -= acc_jx
                acc_s_r[j, 1] -= acc_jy
                acc_s_r[j, 2] -= acc_jz

    return U_s_r, acc_s_r


@njit
def update(pos, p_id, p_mass, box_lengths, rc, potential_matrix, force, measure, rdf_hist):
    """
    Update the force on the particles based on a linked cell-list (LCL) algorithm.

    Parameters
    ----------
    force: func
        Potential and force values.

    potential_matrix: numpy.ndarray
        Potential parameters.

    rc: float
        Cut-off radius.

    box_lengths: numpy.ndarray
        Array of box sides' length.

    p_mass: numpy.ndarray
        Mass of each particle.

    p_id: numpy.ndarray
        Id of each particle

    pos: numpy.ndarray
        Particles' positions.

    measure : bool
        Boolean for rdf calculation.

    rdf_hist : numpy.ndarray
        Radial Distribution function array.

    Returns
    -------
    U_s_r : float
        Short-ranged component of the potential energy of the system.

    acc_s_r : numpy.ndarray
        Short-ranged component of the acceleration for the particles.

    virial : numpy.ndarray
        Virial term of each particle. \n
        Shape = (pos.shape[0], pos.shape[0], pos.shape[1])

    Notes
    -----
    Here the "short-ranged component" refers to the Ewald decomposition of the
    short and long ranged interactions. See the wikipedia article:
    https://en.wikipedia.org/wiki/Ewald_summation or
    "Computer Simulation of Liquids by Allen and Tildesley" for more information.

    """

    # Declare parameters
    N = pos.shape[0]  # Number of particles
    d = pos.shape[1]  # Number of dimensions
    rshift = np.zeros(d)  # Shifts for array flattening

    acc_s_r = np.zeros_like(pos)
    # Virial term for the viscosity calculation
    virial = np.zeros((d, d, N))

    # Initialize
    U_s_r = 0.0  # Short-ranges potential energy accumulator
    ls = np.arange(N)  # List of particle indices in a given cell

    # The number of cells in each dimension
    cells_per_dim = (box_lengths / rc).astype(np.int64)
    cell_length_per_dim = box_lengths / cells_per_dim

    # Total number of cells in volume
    Ncell = cells_per_dim.prod()
    head = np.arange(Ncell)  # List of head particles
    empty = -50  # value for empty list and head arrays
    head.fill(empty)  # Make head list empty until population

    rdf_nbins = rdf_hist.shape[0]
    dr_rdf = rc / float(rdf_nbins)

    # Loop over all particles and place them in cells
    for i in range(N):
        # Determine what cell, in each direction, the i-th particle is in
        cx = int(pos[i, 0] / cell_length_per_dim[0])  # X cell
        cy = int(pos[i, 1] / cell_length_per_dim[1])  # Y cell
        cz = int(pos[i, 2] / cell_length_per_dim[2])  # Z cell

        # Determine cell in 3D volume for i-th particle
        c = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]
        # List of particle indices occupying a given cell
        ls[i] = head[c]

        # The last particle found to lie in cell c (head particle)
        head[c] = i

    # Loop over all cells in x, y, and z direction
    for cx in range(cells_per_dim[0]):
        for cy in range(cells_per_dim[1]):
            for cz in range(cells_per_dim[2]):

                # Compute the cell in 3D volume
                c = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]

                # Loop over all cell pairs (N-1 and N+1)
                for cz_N in range(cz - 1, cz + 2):
                    # z cells
                    # Check periodicity: needed for 0th cell
                    # if cz_N < 0:
                    #     cz_shift = cells_per_dim[2]
                    #     rshift[2] = -box_lengths[2]
                    # # Check periodicity: needed for Nth cell
                    # elif cz_N >= cells_per_dim[2]:
                    #     cz_shift = -cells_per_dim[2]
                    #     rshift[2] = box_lengths[2]
                    # else:
                    #     cz_shift = 0
                    #     rshift[2] = 0.0
                    cz_shift = 0 + cells_per_dim[2] * (cz_N < 0) - cells_per_dim[2] * (cz_N >= cells_per_dim[2])
                    rshift[2] = 0.0 - box_lengths[2] * (cz_N < 0) + box_lengths[2] * (cz_N >= cells_per_dim[2])

                    for cy_N in range(cy - 1, cy + 2):
                        # y cells
                        # Check periodicity
                        # if cy_N < 0:
                        #     cy_shift = cells_per_dim[1]
                        #     rshift[1] = -box_lengths[1]
                        # elif cy_N >= cells_per_dim[1]:
                        #     cy_shift = -cells_per_dim[1]
                        #     rshift[1] = box_lengths[1]
                        # else:
                        #     cy_shift = 0
                        #     rshift[1] = 0.0

                        cy_shift = 0 + cells_per_dim[1] * (cy_N < 0) - cells_per_dim[1] * (cy_N >= cells_per_dim[1])
                        rshift[1] = 0.0 - box_lengths[1] * (cy_N < 0) + box_lengths[1] * (cy_N >= cells_per_dim[1])

                        for cx_N in range(cx - 1, cx + 2):
                            # x cells
                            # Check periodicity
                            # if cx_N < 0:
                            #     cx_shift = cells_per_dim[0]
                            #     rshift[0] = -box_lengths[0]
                            # elif cx_N >= cells_per_dim[0]:
                            #     cx_shift = -cells_per_dim[0]
                            #     rshift[0] = box_lengths[0]
                            # else:
                            #     cx_shift = 0
                            #     rshift[0] = 0.0

                            cx_shift = 0 + cells_per_dim[0] * (cx_N < 0) - cells_per_dim[0] * (cx_N >= cells_per_dim[0])
                            rshift[0] = 0.0 - box_lengths[0] * (cx_N < 0) + box_lengths[0] * (cx_N >= cells_per_dim[0])

                            # Compute the location of the N-th cell based on shifts
                            c_N = (
                                (cx_N + cx_shift)
                                + (cy_N + cy_shift) * cells_per_dim[0]
                                + (cz_N + cz_shift) * cells_per_dim[0] * cells_per_dim[1]
                            )

                            i = head[c]
                            # First compute interaction of head particle with neighboring cell head particles
                            # Then compute interactions of head particle within a specific cell
                            while i != empty:

                                # Check neighboring head particle interactions
                                j = head[c_N]

                                while j != empty:

                                    # Only compute particles beyond i-th particle (Newton's 3rd Law)
                                    if i < j:

                                        # Compute the difference in positions for the i-th and j-th particles
                                        dx = pos[i, 0] - (pos[j, 0] + rshift[0])
                                        dy = pos[i, 1] - (pos[j, 1] + rshift[1])
                                        dz = pos[i, 2] - (pos[j, 2] + rshift[2])

                                        # Compute distance between particles i and j
                                        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                                        if measure and int(r / dr_rdf) < rdf_nbins:
                                            rdf_hist[int(r / dr_rdf), p_id[i], p_id[j]] += 1

                                        # If below the cutoff radius, compute the force
                                        if r < rc:
                                            p_matrix = potential_matrix[:, p_id[i], p_id[j]]

                                            # Compute the short-ranged force
                                            pot, fr = force(r, p_matrix)
                                            fr /= r
                                            U_s_r += pot

                                            # Update the acceleration for i particles in each dimension

                                            acc_s_r[i, 0] += dx * fr / p_mass[i]
                                            acc_s_r[i, 1] += dy * fr / p_mass[i]
                                            acc_s_r[i, 2] += dz * fr / p_mass[i]

                                            # Apply Newton's 3rd law to update acceleration on j particles
                                            acc_s_r[j, 0] -= dx * fr / p_mass[j]
                                            acc_s_r[j, 1] -= dy * fr / p_mass[j]
                                            acc_s_r[j, 2] -= dz * fr / p_mass[j]

                                            # Since we have the info already calculate the virial
                                            virial[0, 0, i] += dx * dx * fr
                                            virial[0, 1, i] += dx * dy * fr
                                            virial[0, 2, i] += dx * dz * fr
                                            virial[1, 0, i] += dy * dx * fr
                                            virial[1, 1, i] += dy * dy * fr
                                            virial[1, 2, i] += dy * dz * fr
                                            virial[2, 0, i] += dz * dx * fr
                                            virial[2, 1, i] += dz * dy * fr
                                            virial[2, 2, i] += dz * dz * fr

                                    # Move down list (ls) of particles for cell interactions with a head particle
                                    j = ls[j]

                                # Check if head particle interacts with other cells
                                i = ls[i]
    return U_s_r, acc_s_r, virial


@njit
def calculate_virial(pos, p_id, box_lengths, rc, potential_matrix, force):
    """
    Update the force on the particles based on a linked cell-list (LCL) algorithm.

    Parameters
    ----------
    force: float, float
        Potential and force values.

    potential_matrix: array
        Potential parameters.

    rc: float
        Cut-off radius.

    box_lengths: array
        Array of box sides' length.

    p_id: array
        Id of each particle

    pos: array
        Particles' positions.

    Returns
    -------

    """
    # Declare parameters
    N = pos.shape[0]  # Number of particles
    d = pos.shape[1]  # Number of dimensions
    rshift = np.zeros(d)  # Shifts for array flattening
    # Virial term for the viscosity calculation
    virial = np.zeros((3, 3, N))
    # Initialize
    U_s_r = 0.0  # Short-ranges potential energy accumulator
    ls = np.arange(N)  # List of particle indices in a given cell

    # The number of cells in each dimension
    cells_per_dim = (box_lengths / rc).astype(np.int64)
    cell_length_per_dim = box_lengths / cells_per_dim

    # Total number of cells in volume
    Ncell = cells_per_dim.prod()
    head = np.arange(Ncell)  # List of head particles
    empty = -50  # value for empty list and head arrays
    head.fill(empty)  # Make head list empty until population

    # Loop over all particles and place them in cells
    for i in range(N):
        # Determine what cell, in each direction, the i-th particle is in
        cx = int(pos[i, 0] / cell_length_per_dim[0])  # X cell
        cy = int(pos[i, 1] / cell_length_per_dim[1])  # Y cell
        cz = int(pos[i, 2] / cell_length_per_dim[2])  # Z cell

        # Determine cell in 3D volume for i-th particle
        c = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]
        # List of particle indices occupying a given cell
        ls[i] = head[c]

        # The last particle found to lie in cell c (head particle)
        head[c] = i

    # Loop over all cells in x, y, and z direction
    for cx in range(cells_per_dim[0]):
        for cy in range(cells_per_dim[1]):
            for cz in range(cells_per_dim[2]):

                # Compute the cell in 3D volume
                c = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]

                # Loop over all cell pairs (N-1 and N+1)
                for cz_N in range(cz - 1, cz + 2):
                    # z cells
                    # Check periodicity: needed for 0th cell
                    # if cz_N < 0:
                    #     cz_shift = cells_per_dim[2]
                    #     rshift[2] = -box_lengths[2]
                    # # Check periodicity: needed for Nth cell
                    # elif cz_N >= cells_per_dim[2]:
                    #     cz_shift = -cells_per_dim[2]
                    #     rshift[2] = box_lengths[2]
                    # else:
                    #     cz_shift = 0
                    #     rshift[2] = 0.0
                    cz_shift = 0 + cells_per_dim[2] * (cz_N < 0) - cells_per_dim[2] * (cz_N >= cells_per_dim[2])
                    rshift[2] = 0.0 - box_lengths[2] * (cz_N < 0) + box_lengths[2] * (cz_N >= cells_per_dim[2])

                    for cy_N in range(cy - 1, cy + 2):
                        # y cells
                        # Check periodicity
                        # if cy_N < 0:
                        #     cy_shift = cells_per_dim[1]
                        #     rshift[1] = -box_lengths[1]
                        # elif cy_N >= cells_per_dim[1]:
                        #     cy_shift = -cells_per_dim[1]
                        #     rshift[1] = box_lengths[1]
                        # else:
                        #     cy_shift = 0
                        #     rshift[1] = 0.0

                        cy_shift = 0 + cells_per_dim[1] * (cy_N < 0) - cells_per_dim[1] * (cy_N >= cells_per_dim[1])
                        rshift[1] = 0.0 - box_lengths[1] * (cy_N < 0) + box_lengths[1] * (cy_N >= cells_per_dim[1])

                        for cx_N in range(cx - 1, cx + 2):
                            # x cells
                            # Check periodicity
                            # if cx_N < 0:
                            #     cx_shift = cells_per_dim[0]
                            #     rshift[0] = -box_lengths[0]
                            # elif cx_N >= cells_per_dim[0]:
                            #     cx_shift = -cells_per_dim[0]
                            #     rshift[0] = box_lengths[0]
                            # else:
                            #     cx_shift = 0
                            #     rshift[0] = 0.0

                            cx_shift = 0 + cells_per_dim[0] * (cx_N < 0) - cells_per_dim[0] * (cx_N >= cells_per_dim[0])
                            rshift[0] = 0.0 - box_lengths[0] * (cx_N < 0) + box_lengths[0] * (cx_N >= cells_per_dim[0])

                            # Compute the location of the N-th cell based on shifts
                            c_N = (
                                (cx_N + cx_shift)
                                + (cy_N + cy_shift) * cells_per_dim[0]
                                + (cz_N + cz_shift) * cells_per_dim[0] * cells_per_dim[1]
                            )

                            i = head[c]
                            # First compute interaction of head particle with neighboring cell head particles
                            # Then compute interactions of head particle within a specific cell
                            while i != empty:

                                # Check neighboring head particle interactions
                                j = head[c_N]

                                while j != empty:

                                    # Only compute particles beyond i-th particle (Newton's 3rd Law)
                                    if i < j:

                                        # Compute the difference in positions for the i-th and j-th particles
                                        dx = pos[i, 0] - (pos[j, 0] + rshift[0])
                                        dy = pos[i, 1] - (pos[j, 1] + rshift[1])
                                        dz = pos[i, 2] - (pos[j, 2] + rshift[2])

                                        # Compute distance between particles i and j
                                        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                                        # If below the cutoff radius, compute the force
                                        if r < rc:
                                            p_matrix = potential_matrix[:, p_id[i], p_id[j]]

                                            # Compute the short-ranged force
                                            pot, fr = force(r, p_matrix)
                                            fr /= r
                                            U_s_r += pot

                                            virial[0, 0, i] += dx * dx * fr
                                            virial[0, 1, i] += dx * dy * fr
                                            virial[0, 2, i] += dx * dz * fr
                                            virial[1, 0, i] += dy * dx * fr
                                            virial[1, 1, i] += dy * dy * fr
                                            virial[1, 2, i] += dy * dz * fr
                                            virial[2, 0, i] += dz * dx * fr
                                            virial[2, 1, i] += dz * dy * fr
                                            virial[2, 2, i] += dz * dz * fr

                                    # Move down list (ls) of particles for cell interactions with a head particle
                                    j = ls[j]

                                # Check if head particle interacts with other cells
                                i = ls[i]
    return virial
