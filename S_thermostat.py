import numpy as np
import sys
from mpi4py import MPI

import S_velocity_verlet as velocity_verlet
import S_global_names as glb
import S_constants as const

def vscale(pos, vel, acc, T_desired, it, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p,mpiComm,eq):
    kf = glb.kf
    dt = glb.dt
    N = glb.N
    ai = glb.ai
    mi = const.pMass
    q1 = glb.q1
    q2 = glb.q2

    U, acc, vel, pos = velocity_verlet.update(pos, vel, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p,mpiComm,eq)

    K = 0.0
    T = 0.0
    if(glb.units == "Yukawa"):
        kf = 1.5
        K = kf*np.ndarray.sum(vel**2)
        totalK = np.array([0.0])
        mpiComm.comm.Allreduce(K, totalK, op=MPI.SUM)
        K = totalK[0]
        T = K/kf/float(N)
        #K *= 3
        #T *= 3
    else:
        K = 0.5*mi*np.ndarray.sum(vel**2)
        T = (2/3)*K/float(N)/const.kb


    if it <= 1999:
        fact = np.sqrt(T_desired/T)
        vel = vel*fact

    else:
        fact = np.sqrt((21.0*T_desired/T-1.0)/20.0)
        vel = vel*fact


    return  U, acc, vel, pos
