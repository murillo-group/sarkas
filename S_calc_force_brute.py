'''
S_calc_force_brute.py

* Calculate force and potential using a brute-force technique. 

* This is only for test potentials and integrators, not a real simulation purpose!
* No periodic bc is assumed.

* glb.force is defined in S_force.py
'''

import numpy as np
import numba as nb
import math as mt
import sys
import time

import S_global_names as glb

@nb.jit
def update(pos, acc_s_r):

    U_s_r = 0.0
    acc_s_r.fill(0.0)

    for i in range(N):
        for j in range(i, N):
            dx = pos[i,0] - (pos[j,0]
            dy = pos[i,1] - (pos[j,1]
            dz = pos[i,2] - (pos[j,2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)

            U_s_r, fr = glb.force(r, U_s_r)

            acc_s_r[i,0] = acc_s_r[i,0] + fr*dx/r
            acc_s_r[i,1] = acc_s_r[i,1] + fr*dy/r
            acc_s_r[i,2] = acc_s_r[i,2] + fr*dz/r
            
            acc_s_r[j,0] = acc_s_r[j,0] - fr*dx/r
            acc_s_r[j,1] = acc_s_r[j,1] - fr*dy/r
            acc_s_r[j,2] = acc_s_r[j,2] - fr*dz/r

    return U_s_r, acc_s_r
