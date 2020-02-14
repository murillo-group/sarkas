"""
S_calc_force_pp.py

Module for handling the Particle-Particle interaction.

"""

import numpy as np
import numba as nb
import math as mt
import sys
import time

from numba.errors import NumbaWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

# These "ignore" should be only temporary until we figure out a way to speed up the update functions
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

@nb.jit
def update_0D(ptcls, params):
    """
    Updates particles' accelerations when rc = L/2
    For no sub-cell. All ptcls within rc (= L/2) participate for force calculation. Cost ~ O(N^2)

    Parameters
    ----------
    ptcls : class
            Particles' class. See S_Particles for more info

    params : class
             Simulation Parameters. See S_Params for more info

    Returns
    -------
    U_s_r : array
            Potential

    acc_s_r : array
              Particles' accelerations

    """
    pos = ptcls.pos
    rc = params.Potential.rc
    L = params.L
    Lh = L/2.
    N = params.N # Number of particles

    potential_matrix = params.Potential.matrix
    id_ij = ptcls.species_id
    mass_ij = ptcls.mass
    force = params.force
    
    U_s_r = 0.0 # Short-ranges potential energy accumulator
    acc_s_r = np.zeros( (params.N, params.d) ) # Vector of accelerations

    for i in range(N):
        for j in range(i+1, N):
            dx = (pos[i,0] - pos[j,0])
            dy = (pos[i,1] - pos[j,1])
            dz = (pos[i,2] - pos[j,2])
            if (1):
                if(dx >= Lh):
                    #dx -= Lh 
                    dx = L - dx
                elif(dx <= -Lh):
                    #dx += Lh
                    dx = L + dx

                if(dy >= Lh):
                    #dy -= Lh 
                    dy = L - dy
                elif(dy <= -Lh):
                    #dy += Lh
                    dy = L + dy

                if(dz >= Lh):
                    #dz -= Lh 
                    dz = L - dz
                elif(dz <= -Lh):
                    #dz += Lh
                    dz = L + dz

            # Compute distance between particles i and j
            r = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if (r < rc):
                id_i = id_ij[i]
                id_j = id_ij[j]
                mass_i = mass_ij[i]
                mass_j = mass_ij[j]
                
                p_matrix = potential_matrix[:, id_i, id_j]
                # Compute the short-ranged force
                pot, fr = force(r, p_matrix)
                U_s_r += pot

                # Update the acceleration for i particles in each dimension

                acc_ix = dx*fr/mass_i
                acc_iy = dy*fr/mass_i
                acc_iz = dz*fr/mass_i

                acc_jx = dx*fr/mass_j
                acc_jy = dy*fr/mass_j
                acc_jz = dz*fr/mass_j

                acc_s_r[i,0] = acc_s_r[i,0] + acc_ix 
                acc_s_r[i,1] = acc_s_r[i,1] + acc_iy
                acc_s_r[i,2] = acc_s_r[i,2] + acc_iz
                
                # Apply Newton's 3rd law to update acceleration on j particles
                acc_s_r[j,0] = acc_s_r[j,0] - acc_jx
                acc_s_r[j,1] = acc_s_r[j,1] - acc_jy
                acc_s_r[j,2] = acc_s_r[j,2] - acc_jz

    return U_s_r, acc_s_r


@nb.jit # This will give a warning, but it is still faster than without it or in forceobj=True mode.
def update(ptcls,params):
    """ 
    Update the force on the particles based on a linked cell-list (LCL) algorithm.
  
    Parameters
    ----------
    ptcls : class 
        Particles's data. See S_particles.py for more info

    params : class
            Simulation's parameters. See S_params.py for more info

    Returns
    -------
    U_s_r : float
        Short-ranged component of the potential energy of the system

    acc_s_r : array_like
        Short-ranged component of the acceleration for the particles

    Notes
    -----
    Here the "short-ranged component" refers to the Ewald decomposition of the
    short and long ranged interactions. See the wikipedia article:
    https://en.wikipedia.org/wiki/Ewald_summation or
    "Computer Simulation of Liquids by Allen and Tildesley" for more information.
    """
    pos = ptcls.pos
    acc_s_r = np.zeros_like(pos)

    # Declare parameters 
    rc = params.Potential.rc # Cutoff-radius
    N = params.N # Number of particles
    d = params.d # Number of dimensions
    rshift = np.zeros(d) # Shifts for array flattening
    Lx = params.Lv[0] # X length of box
    Ly = params.Lv[1] # Y length of box
    Lz = params.Lv[2] # Z length of box
    potential_matrix = params.Potential.matrix
    id_ij = ptcls.species_id
    mass_ij = ptcls.mass
    force = params.force

    # Initialize
    U_s_r = 0.0 # Short-ranges potential energy accumulator
    ls = np.arange(N) # List of particle indices in a given cell

    # The number of cells in each dimension
    Lxd = int(Lx/rc)
    Lyd = int(Ly/rc)
    Lzd = int(Lz/rc)

    # Width of each cell
    rc_x = Lx/Lxd
    rc_y = Ly/Lyd
    rc_z = Lz/Lzd

    # Total number of cells in volume
    Ncell = Lxd*Lyd*Lzd
    head = np.arange(Ncell) # List of head particles
    empty = -50 # value for empty list and head arrays
    head.fill(empty) # Make head list empty until population

    # Loop over all particles and place them in cells
    for i in range(N):
    
        # Determine what cell, in each direction, the i-th particle is in
        cx = int(pos[i,0]/rc_x) # X cell
        cy = int(pos[i,1]/rc_y) # Y cell
        cz = int(pos[i,2]/rc_z) # Z cell
        # Determine cell in 3D volume for i-th particle
        c = cx + cy*Lxd + cz*Lxd*Lyd
        # List of particle indices occupying a given cell
        ls[i] = head[c]

        # The last particle found to lie in cell c (head particle)
        head[c] = i
    
    # Loop over all cells in x, y, and z direction
    for cx in range(Lxd):
        for cy in range(Lyd):
            for cz in range(Lzd):

                # Compute the cell in 3D volume
                c = cx + cy*Lxd + cz*Lxd*Lyd

                # Loop over all cell pairs (N-1 and N+1)
                for cz_N in range(cz-1,cz+2):
                    for cy_N in range(cy-1,cy+2):
                        for cx_N in range(cx-1,cx+2):

                            ## x cells ##
                            # Check if periodicity is needed for 0th cell
                            if (cx_N < 0): 
                                cx_shift = Lxd
                                rshift[0] = -Lx
                            # Check if periodicity is needed for Nth cell
                            elif (cx_N >= Lxd): 
                                cx_shift = -Lxd
                                rshift[0] = Lx
                            else:
                                cx_shift = 0
                                rshift[0] = 0.0
                
                            ## y cells ##
                            # Check periodicity
                            if (cy_N < 0): 
                                cy_shift = Lyd
                                rshift[1] = -Ly
                            # Check periodicity
                            elif (cy_N >= Lyd): 
                                cy_shift = -Lyd
                                rshift[1] = Ly
                            else:
                                cy_shift = 0
                                rshift[1] = 0.0
                
                            ## z cells ##
                            # Check periodicity
                            if (cz_N < 0): 
                                cz_shift = Lzd
                                rshift[2] = -Lz
                            # Check periodicity
                            elif (cz_N >= Lzd): 
                                cz_shift = -Lzd
                                rshift[2] = Lz
                            else:
                                cz_shift = 0
                                rshift[2] = 0.0
                
                            # Compute the location of the N-th cell based on shifts
                            c_N = (cx_N+cx_shift) + (cy_N+cy_shift)*Lxd + (cz_N+cz_shift)*Lxd*Lyd
            
                            i = head[c]
            
                            # First compute interaction of head particle with neighboring cell head particles
                            # Then compute interactions of head particle within a specific cell
                            while(i != empty):
                                
                                # Check neighboring head particle interactions
                                j = head[c_N]

                                while(j != empty):
                    
                                    # Only compute particles beyond i-th particle (Newton's 3rd Law)
                                    if i < j:

                                        # Compute the difference in positions for the i-th and j-th particles
                                        dx = pos[i,0] - (pos[j,0] + rshift[0])
                                        dy = pos[i,1] - (pos[j,1] + rshift[1])
                                        dz = pos[i,2] - (pos[j,2] + rshift[2])

                                        # Compute distance between particles i and j
                                        r = np.sqrt(dx**2 + dy**2 + dz**2)
                                        # If below the cutoff radius, compute the force
                                        if r < rc:
                                            id_i = id_ij[i]
                                            id_j = id_ij[j]
                                            mass_i = mass_ij[i]
                                            mass_j = mass_ij[j]
                                            p_matrix = potential_matrix[:, id_i, id_j]

                                            # Compute the short-ranged force
                                            pot, fr = force(r, p_matrix)
                                            U_s_r += pot

                                            # Update the acceleration for i particles in each dimension

                                            acc_ix = dx*fr/mass_i
                                            acc_iy = dy*fr/mass_i
                                            acc_iz = dz*fr/mass_i

                                            acc_jx = dx*fr/mass_j
                                            acc_jy = dy*fr/mass_j
                                            acc_jz = dz*fr/mass_j

                                            acc_s_r[i,0] = acc_s_r[i,0] + acc_ix
                                            acc_s_r[i,1] = acc_s_r[i,1] + acc_iy
                                            acc_s_r[i,2] = acc_s_r[i,2] + acc_iz
                                            
                                            # Apply Newton's 3rd law to update acceleration on j particles
                                            acc_s_r[j,0] = acc_s_r[j,0] - acc_jx
                                            acc_s_r[j,1] = acc_s_r[j,1] - acc_jy
                                            acc_s_r[j,2] = acc_s_r[j,2] - acc_jz
                                    
                                    # Move down list (ls) of particles for cell interactions with a head particle
                                    j = ls[j]

                                # Check if head particle interacts with other cells
                                i = ls[i]
    return U_s_r, acc_s_r

@nb.jit
def update_brute(ptcls,params):
    """ 
    Update particles' accelerations via brute force calculation. Cost O(N^2)

    Parameters
    ----------
    ptcls : class 
        Particles's data. See S_particles.py for more info

    params : class
            Simulation's parameters. See S_params.py for more info

    Returns
    -------
    U_s_r : float
        Potential energy

    acc_s_r : array_like
        Particles' accelerations

    """
    pos = ptcls.pos
    acc_s_r = np.zeros_like(pos)

    # Declare parameters 
    rc = params.Potential.rc # Cutoff-radius
    N = params.N # Number of particles
    d = params.d # Number of dimensions
    rshift = np.zeros(d) # Shifts for array flattening
    Lx = params.Lv[0] # X length of box
    Ly = params.Lv[1] # Y length of box
    Lz = params.Lv[2] # Z length of box
    potential_matrix = params.Potential.matrix
    id_ij = ptcls.species_id
    mass_ij = ptcls.mass
    force = params.force

    # Initialize
    U_s_r = 0.0 # Short-ranges potential energy accumulator
  
    # Only compute particles beyond i-th particle (Newton's 3rd Law)
    for i in range( N):
        for j in range(i + 1, N):

            # Compute the difference in positions for the i-th and j-th particles
            dx = pos[i,0] - (pos[j,0] )
            dy = pos[i,1] - (pos[j,1] )
            dz = pos[i,2] - (pos[j,2] )

            # Compute distance between particles i and j
            r = np.sqrt(dx*dx + dy*dy + dz*dz)
            # If below the cutoff radius, compute the force

            id_i = id_ij[i]
            id_j = id_ij[j]
            mass_i = mass_ij[i]
            mass_j = mass_ij[j]
            p_matrix = potential_matrix[:, id_i, id_j]

            # Compute the short-ranged force
            pot, fr = force(r, p_matrix)
            U_s_r += pot

            # Update the acceleration for i particles in each dimension

            acc_ix = dx*fr/mass_i
            acc_iy = dy*fr/mass_i
            acc_iz = dz*fr/mass_i

            acc_jx = dx*fr/mass_j
            acc_jy = dy*fr/mass_j
            acc_jz = dz*fr/mass_j

            acc_s_r[i,0] = acc_s_r[i,0] + acc_ix
            acc_s_r[i,1] = acc_s_r[i,1] + acc_iy
            acc_s_r[i,2] = acc_s_r[i,2] + acc_iz
            
            # Apply Newton's 3rd law to update acceleration on j particles
            acc_s_r[j,0] = acc_s_r[j,0] - acc_jx
            acc_s_r[j,1] = acc_s_r[j,1] - acc_jy
            acc_s_r[j,2] = acc_s_r[j,2] - acc_jz

    return U_s_r, acc_s_r

    
