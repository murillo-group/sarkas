'''
S_calc_force.py

* Call an appropriate force calculation module based on the algorithm

* input: particles
* output: potential

'''
import numpy as np
import sys
import S_calc_force_pp as force_pp
import S_constants as const

def force_pot(ptcls,params):
    N = ptcls.pos.shape[0]
    d = ptcls.pos.shape[1]
    acc_s_r = np.zeros( (N,  d) )
    acc_fft = np.zeros_like(acc_s_r)

    acc_s_r.fill(0.0)
    if(params.Potential.LL_on):
        U_short, acc_s_r = force_pp.update(ptcls, acc_s_r, params)
    else:
        U_short, acc_s_r = force_pp.update_0D(ptcls, acc_s_r, params)
    ptcls.acc = (acc_s_r)
    U_fft = 0
    U_self = 0.
    U = U_short
    return U
