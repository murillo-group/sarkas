import numpy as np
import sys
import S_p3m as p3m
import S_global_names as glb
import S_constants as const
import S_yukawa_gf_opt as yukawa_gf_opt


class integrator:
    def __init__(self, params, glb):
        self.params = params
        self.glb_vars = glb
        if(self.params.potential[0].type == "Yukawa"):
            glb.G_k, glb.kx_v, glb.ky_v, glb.kz_v, glb.A_pm = yukawa_gf_opt.gf_opt()

        if(params.Integrator[0].type == "Verlet"):
            if(params.Langevin):
                self.update = self.Verlet_with_Langevin
            else:
                self.update = self.Verlet
        else:
            print("Only Verlet integrator is supported. Check your input file, integrator part.")
            sys.exit()

    def Verlet(self, pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
        G_k = glb.G_k
        kx_v = glb.kx_v
        ky_v = glb.ky_v
        kz_v = glb.kz_v

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
            for i in np.arange(N):
                for p in np.arange(d):
                    if pos[i, p] > Lmax_v[p]:
                        pos[i, p] = pos[i, p] - Lv[p]
                    if pos[i, p] < Lmin_v[p]:
                        pos[i, p] = pos[i, p] + Lv[p]

        U, acc = p3m.force_pot(pos, acc, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)
        vel = vel + 0.5*acc*dt

        return pos, vel, acc, U

    def Verlet_with_Langevin(pos, vel, acc, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
        dt = glb.dt
        g = glb.g_0
        Gamma = glb.Gamma
        Lmax_v = glb.Lmax_v
        Lmin_v = glb.Lmin_v
        Lv = glb.Lv
        PBC = glb.PBC
        N = glb.N
        d = glb.d

        rtdt = np.sqrt(dt)

        sig = np.sqrt(2. * g*const.kb*glb.T_desired/const.proton_mass)
        if(glb.units == "Yukawa"):
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
