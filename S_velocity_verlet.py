import numpy as np
import numba as nb
import sys
import S_p3m as p3m
import S_global_names as glb
import S_constants as const

#@nb.autojit
def update(pos, vel, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p,mpiComm):
    dt = glb.dt
    N = glb.N
    d = glb.d
    Lv = glb.Lv
    PBC = glb.PBC
    Lmax_v = glb.Lmax_v
    Lmin_v = glb.Lmin_v
    vel = vel + 0.5*acc*dt
    
    pos = pos + vel*dt
    
    # periodic boundary condition
    if PBC == 1:
        for i in np.arange(len(pos[:,0])):
            for p in np.arange(d):
        
                if pos[i,p] > Lmax_v[p]:
                    pos[i,p] = pos[i,p] - Lv[p]
                if pos[i,p] < Lmin_v[p]:
                    pos[i,p] = pos[i,p] + Lv[p]
        
    U, acc, vel, pos = p3m.force_pot(pos, vel, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p,mpiComm)
    vel = vel + 0.5*acc*dt

    return pos, vel, acc, U

#@nb.autojit
def update_Langevin(pos, vel, acc, Z,  G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p, mpiComm):
    dt = glb.dt
    g = glb.g_0
    Gamma = glb.Gamma
    Lmax_v = glb.Lmax_v
    Lmin_v = glb.Lmin_v
    Lv = glb.Lv
    PBC = glb.PBC
    N = len(pos[:,0]) #local Particle Num.
    d = glb.d

    rtdt = np.sqrt(dt)

    sig = np.sqrt(2. * g*const.kb*glb.T_desired/const.pMass)
    if(glb.units == "Yukawa"):
        sig = np.sqrt(2. * g/(3*Gamma))

    c1 = (1. - 0.5*g*dt) 
    c2 = 1./(1. + 0.5*g*dt)
    beta = np.random.normal(0., 1., 3*N).reshape(N, 3)

    #if mpiComm.rank == 0:
    #    print("acc = ", acc.shape)
    #    print("vel = ", vel.shape)
    #    print("pos = ", pos.shape)

    pos = pos + c1*dt*vel + 0.5*dt**2*acc + 0.5*sig* dt**1.5*beta
    
    # periodic boundary condition
    if PBC == 1:
        for i in np.arange(N):
            for p in np.arange(d):
        
                if pos[i,p] > Lmax_v[p]:
                    pos[i,p] = pos[i,p] - Lv[p]
                if pos[i,p] < Lmin_v[p]:
                    pos[i,p] = pos[i,p] + Lv[p]
        
    
    U, acc_new, vel, pos = p3m.force_pot(pos, vel, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p,mpiComm)
    
    vel = c1*c2*vel + 0.5*dt*(acc_new + acc)*c2 + c2*sig*rtdt*beta
    acc = acc_new
    
    return pos, vel, acc, U
