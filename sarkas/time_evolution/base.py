"""
Module of various types of time_evolution


"""

import numpy as np
from numba import njit
from tqdm import tqdm
from . import integrators, thermostats




@njit
def enforce_pbc(pos, cntr, BoxVector):
    """
    Enforce Periodic Boundary conditions.

    Parameters
    ----------
    pos : array
        particles' positions. See ``S_particles.py`` for more info.

    cntr: array
        Counter for the number of times each particle get folded back into the main simulation box

    BoxVector : array
        Box Dimensions.

    """

    # Loop over all particles
    for p in np.arange(pos.shape[0]):
        for d in np.arange(pos.shape[1]):

            # If particle is outside of box in positive direction, wrap to negative side
            if pos[p, d] > BoxVector[d]:
                pos[p, d] -= BoxVector[d]
                cntr[p, d] += 1
            # If particle is outside of box in negative direction, wrap to positive side
            if pos[p, d] < 0.0:
                pos[p, d] += BoxVector[d]
                cntr[p, d] -= 1
    return
