import numpy as np
import sys

import S_velocity_verlet as velocity_verlet
import S_global_names as glb
import S_constants as const
from S_integrator import integrator 

class thermostat:
    def __init__(self, params, glb):
        self.integrator = integrator(params, glb)

        self.params = params
        self.glb_vars = glb

        if(params.Integrator[0].type == "Verlet"):
            print("therm verlet")
            self.integrator = self.integrator.Verlet

        if not (params.Integrator[0].type == "Verlet"):
            print("Only Verlet integrator is supported. Check your input file, integrator part.")
            sys.exit()


    def update(self, pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
        
        T_desired = glb.T_desired
        kf = glb.kf
        dt = glb.dt
        N = glb.N
        ai = glb.ai
        mi = const.proton_mass
        q1 = glb.q1
        q2 = glb.q2
        G_k = glb.G_k
        kx_v = glb.kx_v
        ky_v = glb.ky_v
        kz_v = glb.kz_v




        pos, vel, acc, U =  self.integrator(pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)

        
        if(glb.units == "Yukawa"):
            kf = 1.5
            K = kf*np.ndarray.sum(vel**2)
            T = K/kf/float(glb.N)
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


def vscale(pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
    T_desired = glb.T_desired
    kf = glb.kf
    dt = glb.dt
    N = glb.N
    ai = glb.ai
    mi = const.proton_mass
    q1 = glb.q1
    q2 = glb.q2
    G_k = glb.G_k
    kx_v = glb.kx_v
    ky_v = glb.ky_v
    kz_v = glb.kz_v

    pos, vel, acc, U = velocity_verlet.update(pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)


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
