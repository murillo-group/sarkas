import numpy as np
import sys
import S_pm as pm
import S_pp as pp
import S_pp_yukawa as pp_yukawa
import S_pp_EGS as pp_EGS
#import S_pp_EGS as pp_EGS
import S_global_names as glb
import S_constants as const

def force_pot(ptcls, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
    N = glb.N
    G = glb.G
    af = glb.af
    uf = glb.uf
    Zi = glb.Zi
    q1 = glb.q1
    q2 = glb.q2
    mi = const.proton_mass
    G_k = glb.G_k
    kx_v = glb.kx_v
    ky_v = glb.ky_v
    kz_v = glb.kz_v
    p3m = glb.p3m_flag

    acc_s_r.fill(0.0)
    acc_fft.fill(0.0)
    
    rho_r.fill(0.0)
    E_x_p.fill(0.0)
    E_y_p.fill(0.0)
    E_z_p.fill(0.0)

    U_fft = 0.
    U_self = 0.
#    if(glb.potential_type == glb.EGS):
#        U_short, acc_s_r = pp_EGS.particle_particle(pos,acc_s_r)
#    else: 
    if(glb.potential_type == glb.EGS):
        U_short, acc_s_r = pp_EGS.particle_particle(ptcls.pos, acc_s_r)
    else:
        U_short, acc_s_r = pp_yukawa.particle_particle(ptcls.pos, acc_s_r)

    if(p3m == 1):  # PM
        U_fft, acc_fft = pm.particle_mesh_fft_r(ptcls.pos, Z, G_k, kx_v, ky_v, kz_v, rho_r, acc_fft, E_x_p,E_y_p, E_z_p)
        ptcls.acc = af*(acc_s_r + acc_fft)
        U_self = -glb.N*glb.G/np.sqrt(np.pi)
        U = uf*(U_short + U_fft + U_self)
#        print('U_short, U_long, U_self =', [U_short, U_fft, U_self])

    else:
        if(glb.units == "Yukawa"):
            ptcls.acc = af*(acc_s_r)
            #acc *= af
        else:
            ptcls.acc = af*(acc_s_r)*(3*q1*q2/mi)

        if(glb.units == "mks"):
            ptcls.acc /= (4*np.pi*const.eps_0)

        U_fft = 0
        U_self = 0.
        #U = uf*(U_short)*(q1*q2)/(4*np.pi*const.eps_0)
        U = uf*(U_short)*(q1*q2)
        if(glb.units =="mks"):
          U /= (4*np.pi*const.eps_0)
    #    print('U_short, U_long, U_self =', [U_short, U_fft, U_self])
    return U
