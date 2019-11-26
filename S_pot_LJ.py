'''
S_LJ.py

LJ potential and force calculation
'''

import numpy as np
import numba as nb
import sys

# MD modules
import S_global_names as glb


@nb.jit
def potential_and_force(r, pot_matrix_ij):
    epsilon = pot_matrix_ij[0]
    sigma = pot_matrix_ij[1]
    s_over_r = sigma/r
    s_over_r_6 = s_over_r**6
    s_over_r_12 = s_over_r**12

    U = 4*epsilon*(s_over_r_12 - s_over_r_6)
    force = 48*epsilon/r*(s_over_r_12 - 0.5*s_over_r_6)
    return U, force
