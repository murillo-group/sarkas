import numpy as np
import sys
import S_calc_force_pp as force_pp
import S_pp_EGS as pp_EGS
import S_global_names as glb
import S_constants as const

def force_pot(ptcls):
    N = glb.N
    af = glb.af
    uf = glb.uf
    Zi = glb.Zi
    q1 = glb.q1
    q2 = glb.q2
    mi = const.proton_mass

    G = glb.G
    if(glb.p3m_flag == 1):
        G_k = glb.G_k
        kx_v = glb.kx_v
        ky_v = glb.ky_v
        kz_v = glb.kz_v
        p3m = glb.p3m_flag

    Z = np.ones(glb.N)
    acc_s_r = np.zeros((glb.N, glb.d))
    acc_fft = np.zeros_like(acc_s_r)

    if(glb.p3m_flag == 1):
        rho_r = np.zeros((glb.Mz, glb.My, glb.Mx))
        E_x_p = np.zeros(glb.N)
        E_y_p = np.zeros(glb.N)
        E_z_p = np.zeros(glb.N)

        rho_r.fill(0.0)
        E_x_p.fill(0.0)
        E_y_p.fill(0.0)
        E_z_p.fill(0.0)

    acc_s_r.fill(0.0)
    acc_fft.fill(0.0)
    

    U_fft = 0.
    U_self = 0.
    U_short, acc_s_r = force_pp.update(ptcls.pos, acc_s_r)
    ptcls.acc = af*(acc_s_r)
            #acc *= af
    U_fft = 0
    U_self = 0.
    U = uf*(U_short)
    return U
