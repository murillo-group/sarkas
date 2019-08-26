
import numpy as np
import numba as nb
import math as mt
import sys
import time
import S_global_names as glb

@nb.jit()
def update(pos, acc_s_r):
    ''' Updates the force on the particles based on a linked cell-list (LCL) algorithm.

    
    Parameters
    ----------
    pos : array_like
        Positions of the particles in x, y, and z direction

    acc_s_r : array_like
        Short-ranged acceleration of the particles in the x, y, and z direction

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
    '''

    # Declare parameters 
    rc = glb.rc # Cutoff-radius
    N = len(pos[:,0]) # Number of particles
    d = len(pos[0,:]) # Number of dimensions
    rshift = np.zeros(d) # Shifts for array flattening
    Lx = glb.Lv[0] # X length of box
    Ly = glb.Lv[1] # Y length of box
    Lz = glb.Lv[2] # Z length of box

    # Initialize
    U_s_r = 0.0 # Short-ranges potential energy accumulator
    acc_s_r.fill(0.0) # Vector of accelerations
    head = np.arange(Ncell) # List of head particles
    empty = -50 # value for empty list and head arrays
    head.fill(empty) # Make head list empty until population
    ls = np.arange(N) # List of particle indices in a given cell

    # The number of cells in each dimension
    Lxd = int(np.floor(Lx/rc))
    Lyd = int(np.floor(Ly/rc))
    Lzd = int(np.floor(Lz/rc))

    # Width of each cell
    rc_x = Lx/Lxd
    rc_y = Ly/Lyd
    rc_z = Lz/Lzd

    # Total number of cells in volume
    Ncell = Lxd*Lyd*Lzd

    # Loop over all particles and place them in cells
    for i in range(N):
    
        # Determine what cell, in each direction, the i-th particle is in
        cx = int(np.floor(pos[i,0]/rc_x)) # X cell
        cy = int(np.floor(pos[i,1]/rc_y)) # Y cell
        cz = int(np.floor(pos[i,2]/rc_z)) # Z cell

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

                                            # Compute the short-ranged force
                                            U_s_r, fr = glb.force(r, U_s_r)

                                            # Update the acceleration for i particles in each dimension
                                            acc_s_r[i,0] = acc_s_r[i,0] + fr*dx/r
                                            acc_s_r[i,1] = acc_s_r[i,1] + fr*dy/r
                                            acc_s_r[i,2] = acc_s_r[i,2] + fr*dz/r
                                            
                                            # Apply Newton's 3rd law to update acceleration on j particles
                                            acc_s_r[j,0] = acc_s_r[j,0] - fr*dx/r
                                            acc_s_r[j,1] = acc_s_r[j,1] - fr*dy/r
                                            acc_s_r[j,2] = acc_s_r[j,2] - fr*dz/r
                                    
                                    # Move down list (ls) of particles for cell interactions with a head particle
                                    j = ls[j]

                                # Check if head particle interacts with other cells
                                i = ls[i]
    return U_s_r, acc_s_r