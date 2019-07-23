'''
S_thermostat.py

Berendsen only.
'''
import numpy as np
import sys

import S_constants as const
from S_integrator import Integrator


class Thermostat:
    def __init__(self, params, glb):
        self.integrator = Integrator(params, glb)

        self.params = params
        self.glb_vars = glb
        self.kf = 0.5
        if(glb.units == "Yukawa"):
            self.kf = 1.5

        mi = const.proton_mass
        self.K_factor = self.kf*mi
        self.T_factor = 1/self.kf/float(glb.N)/const.kb

        if(params.thermostat[0].type == "Berendsen"):
            self.type = self.Berendsen
        else:
            print("Only Berendsen thermostat is supported. Check your input file, thermostat part.")
            sys.exit()

        if(params.Integrator[0].type == "Verlet"):
            self.integrator = self.integrator.Verlet
        else:
            print("Only Verlet integrator is supported. Check your input file, integrator part.")
            sys.exit()

    def update(self, ptcls, it):
        U = self.type(ptcls, it)
        return U

    def Berendsen(self, ptcls, it):
        ''' Update particle velocity based on Berendsen thermostat.
    
        Parameters
        ----------
        ptlcs: particles data. See S_particles.py for the detailed information
        it: timestep

        Returns
        -------
        U : float
            Total potential energy
        '''
        T_desired = self.glb_vars.T_desired

        U = self.integrator(ptcls)
        mi = const.proton_mass
        K = self.kf*mi*np.ndarray.sum(ptcls.vel**2)
        T = K/self.kf/float(self.glb_vars.N)/const.kb

        # K = self.K_factor*np.ndarray.sum(vel**2)
        # T = self.T_factor*K

        if it <= 1999:
            fact = np.sqrt(T_desired/T)
            ptcls.vel = ptcls.vel*fact

        else:
            fact = np.sqrt((21.0*T_desired/T-1.0)/20.0)
            ptcls.vel = ptcls.vel*fact

        return U
