'''
S_particle.py

particle loading, adding, and removing

species_name
px,py,pz: position components
vx, vy, vz: velocity components
ax, ay, az: acc. components
charge
mass
'''
import numpy as np
from inspect import currentframe, getframeinfo
import time

class verbose:
    def __init__(self, params, glb):
        print("Sarkas Ver. 0.1")
        self.params = params
        self.glb = glb

    def output(self):
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
        print('time step = ',glb.dt)
        print('No. of equilibration steps = ', glb.Neq)
        print('No. of post-equilibration steps = ', glb.Nt)
        print('snapshot interval = ', glb.snap_int)
        print('Periodic boundary condition{1=yes, 0=no} =', glb.PBC)
        print("Langevin model = ", glb.Langevin_model)
        if(glb.units != "Yukawa"):
            print("plasma frequency, wi = ", glb.wp)
            print("number density, ni = ", glb.ni)
        print('smallest interval in Fourier space for S(q,w): dq = ', 2*np.pi/glb.Lx)


