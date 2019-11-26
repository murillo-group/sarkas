'''
S_thermostat.py

Berendsen only.
N: strength of temperature relaxation after t > 2000th steps.
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

        if(params.Thermostat.type == "Berendsen"):
            self.type = self.Berendsen
        else:
            print("Only Berendsen thermostat is supported. Check your input file, thermostat part.")
            sys.exit()

        if(params.Integrator.type == "Verlet"):
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

        K = 0
        species_start = 0
        for i in range(self.params.num_species):
            species_end = species_start + self.params.species[i].num
            K += 0.5*self.params.species[i].mass*np.ndarray.sum(ptcls.vel[species_start:species_end, :]**2)
            species_start = species_end

        T = (2/3)*K/float(self.params.total_num_ptcls)/const.kb

        N = 20.0 # hardcode
        if it <= 1999: #hardcode
            fact = np.sqrt(T_desired/T)
            ptcls.vel = ptcls.vel*fact

        else:
            fact = np.sqrt((T_desired/T + N - 1.0)/N)
            ptcls.vel = ptcls.vel*fact

        return U
