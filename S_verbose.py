'''
S_verbose.py
printout Version info & simulation progess
'''
import numpy as np
from inspect import currentframe, getframeinfo
import time


class verbose:
    def __init__(self, params, glb):
        print("Sarkas Ver. 1.0")
        self.params = params
        self.glb = glb

    def sim_setting_summary(self):
        glb = self.glb
        params = self.params
        print('\n\n----------- Molecular Dynamics Simulation of Yukawa System ----------------------')
        print("units: ", glb.units)
        if(glb.potential_type == glb.Yukawa_PP or glb.potential_type == glb.Yukawa_P3M):
            print('Gamma = ', glb.Gamma)
            print('kappa = ', glb.kappa)
            print('grid_size * Ewald_parameter (h * alpha) = ', glb.hx*glb.G_ew)
        print('Temperature = ', glb.T_desired)
        print('No. of particles = ', glb.N)
        print('Box length along x axis = ', glb.Lv[0])
        print('Box length along y axis = ', glb.Lv[1])
        print('Box length along z axis = ', glb.Lv[2])
        print('No. of non-zero box dimensions = ', glb.d)
        print('time step = ', glb.dt)
        print('No. of equilibration steps = ', glb.Neq)
        print('No. of post-equilibration steps = ', glb.Nt)
        print('snapshot interval = ', glb.snap_int)
        print('Periodic boundary condition{1=yes, 0=no} =', glb.PBC)
        print("Langevin model = ", glb.Langevin_model)
        if(glb.units != "Yukawa"):
            print("plasma frequency, wi = ", glb.wp)
            print("number density, ni = ", glb.ni)
        print('smallest interval in Fourier space for S(q,w): dq = ', 2*np.pi/glb.Lx)

    def time_stamp(self, time_stamp):
        t = time_stamp
        print('Time for computing converged Greens function = ', t[1]-t[0])
        print('Time for initialization = ', t[2]-t[1])
        print('Time for equilibration = ', t[3]-t[2])
        print('Time for production = ', t[4]-t[3])
        print('Total elapsed time = ', t[4]-t[0])
