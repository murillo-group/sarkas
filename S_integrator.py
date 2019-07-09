'''
S_integrator.py

velocity integrator

Verlet: velocity Verlet
Verlet_with_Langevin: Verlet with Langevin damping
RK45: N/A
RK45_with_Langevin: N/A
'''

import numpy as np
import sys
import S_p3m as p3m
import S_constants as const
import S_yukawa_gf_opt as yukawa_gf_opt


class integrator:
    def __init__(self, params, glb):
        self.params = params
        self.glb_vars = glb
        if(self.params.potential[0].type == "Yukawa"):
            # need one more condition, P3M
            self.glb_vars.G_k, self.glb_vars.kx_v, self.glb_vars.ky_v, self.glb_vars.kz_v, self.glb_vars.A_pm = yukawa_gf_opt.gf_opt()

        if(params.Integrator[0].type == "Verlet"):
            if(params.Langevin):
                self.update = self.Verlet_with_Langevin
            else:
                self.update = self.Verlet
        else:
            print("Only Verlet integrator is supported. Check your input file, integrator part.")
            sys.exit()

    def Verlet(self, pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
        G_k = self.glb_vars.G_k
        kx_v = self.glb_vars.kx_v
        ky_v = self.glb_vars.ky_v
        kz_v = self.glb_vars.kz_v

        dt = self.glb_vars.dt
        N = self.glb_vars.N
        d = self.glb_vars.d
        Lv = self.glb_vars.Lv
        PBC = self.glb_vars.PBC
        Lmax_v = self.glb_vars.Lmax_v
        Lmin_v = self.glb_vars.Lmin_v

        vel = vel + 0.5*acc*dt
        pos = pos + vel*dt

        # periodic boundary condition
        if PBC == 1:
            for i in np.arange(N):
                for p in np.arange(d):
                    if pos[i, p] > Lmax_v[p]:
                        pos[i, p] = pos[i, p] - Lv[p]
                    if pos[i, p] < Lmin_v[p]:
                        pos[i, p] = pos[i, p] + Lv[p]

        U, acc = p3m.force_pot(pos, acc, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)
        vel = vel + 0.5*acc*dt

        return pos, vel, acc, U

    def Verlet_with_Langevin(self, pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
        dt = self.glb_vars.dt
        g = self.glb_vars.g_0
        Gamma = self.glb_vars.Gamma
        Lmax_v = self.glb_vars.Lmax_v
        Lmin_v = self.glb_vars.Lmin_v
        Lv = self.glb_vars.Lv
        PBC = self.glb_vars.PBC
        N = self.glb_vars.N
        d = self.glb_vars.d

        rtdt = np.sqrt(dt)

        sig = np.sqrt(2. * g*const.kb*self.glb_vars.T_desired/const.proton_mass)
        if(self.glb_vars.units == "Yukawa"):
            sig = np.sqrt(2. * g/(3*Gamma))

        c1 = (1. - 0.5*g*dt)
        c2 = 1./(1. + 0.5*g*dt)
        beta = np.random.normal(0., 1., 3*N).reshape(N, 3)

        pos = pos + c1*dt*vel + 0.5*dt**2*acc + 0.5*sig*dt**1.5*beta

        # periodic boundary condition
        if PBC == 1:
            for i in np.arange(N):
                for p in np.arange(d):
                    if pos[i, p] > Lmax_v[p]:
                        pos[i, p] = pos[i, p] - Lv[p]
                    if pos[i, p] < Lmin_v[p]:
                        pos[i, p] = pos[i, p] + Lv[p]

        U, acc_new = p3m.force_pot(pos, acc, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)
        vel = c1*c2*vel + 0.5*dt*(acc_new + acc)*c2 + c2*sig*rtdt*beta
        acc = acc_new
        return pos, vel, acc, U

    def RK45(self):
        pass

    def RK45_with_Langevin(self):
        pass
