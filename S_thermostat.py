'''
S_thermostat.py

Berendsen only.
'''
import numpy as np
import sys

import S_constants as const
from S_integrator import integrator


class thermostat:
    def __init__(self, params, glb):
        self.integrator = integrator(params, glb)

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

    def update(self, pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
        pos, vel, acc, U = self.type(pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)
        return pos, vel, acc, U

    def Berendsen(self, pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
        T_desired = self.glb_vars.T_desired

        pos, vel, acc, U = self.integrator(pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)
        mi = const.proton_mass
        K = self.kf*mi*np.ndarray.sum(vel**2)
        T = K/self.kf/float(self.glb_vars.N)/const.kb

        # K = self.K_factor*np.ndarray.sum(vel**2)
        # T = self.T_factor*K

        if it <= 1999:
            fact = np.sqrt(T_desired/T)
            vel = vel*fact

        else:
            fact = np.sqrt((20.0*T_desired/T-1.0)/20.0)
            vel = vel*fact

        return pos, vel, acc, U
