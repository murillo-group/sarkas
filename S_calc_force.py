'''
S_calc_force.py

* Call an appropriate force calculation module based on the algorithm

* input: particles
* output: potential

'''
import numpy as np
import sys
import S_calc_force_pp as force_pp
import S_global_names as glb
import S_constants as const

def force_pot(ptcls):
    N = glb.N

    acc_s_r = np.zeros((glb.N, glb.d))
    acc_fft = np.zeros_like(acc_s_r)

    acc_s_r.fill(0.0)
    if(glb.LL_on):
        U_short, acc_s_r = force_pp.update(ptcls, acc_s_r)

    else:
        U_short, acc_s_r = force_pp.update_0D(ptcls, acc_s_r)
    ptcls.acc = (acc_s_r)
    U_fft = 0
    U_self = 0.
    U = U_short
    return U
