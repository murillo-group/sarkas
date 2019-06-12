import numpy as np
import sys

import S_velocity_verlet as velocity_verlet
import S_global_names as glb
import S_constants as const

def vscale(pos, vel, acc, T_desired, it, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p,mpiComm):
    kf = glb.kf
    dt = glb.dt
    N = glb.N
    ai = glb.ai
    mi = const.pMass
    q1 = glb.q1
    q2 = glb.q2

    pos, vel, acc, U = velocity_verlet.update(pos, vel, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p,mpiComm)


    if(glb.units == "Yukawa"):
        kf = 1.5
        K = kf*np.ndarray.sum(vel**2)
        T = K/kf/float(N)
        #K *= 3
        #T *= 3
    else:
        K = 0.5*mi*np.ndarray.sum(vel**2)
        T = (2/3)*K/float(N)/const.kb


#    print dt*it, T
    
    if it <= 1999:
         
        fact = np.sqrt(T_desired/T)
        vel = vel*fact
        
    else:
        
        fact = np.sqrt((20.0*T_desired/T-1.0)/20.0)
        vel = vel*fact
    return pos, vel, acc, U
