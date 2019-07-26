'''
S_global_names.py

setting global variables and physical constants
'''

import numpy as np
import sys
import S_global_names as glb # self import. I know it is weird, but it works.
import S_constants as const
import S_yukawa_gf_opt as yukawa_gf_opt
import S_force as force

# read input data from yukawa_MD_p3m.in
def init(params):

    units = params.control[0].units
    potential_type = params.potential[0].type

    # Setup units-indep. vars.
    Aux(params)

    # Setup units-dep. vars.
    if(units == "Yukawa" or "yukawa"):
        Yukawa_units(params)

    elif(units == "CGS" or "cgs"):
        CGS_units_CGS(params)

    elif(units == "MKS" or "mks"):
        MKS_units(params)
    else:
        print("No such units are available")
        sys.exit()


    return


def Aux(params):

    glb.verbose = 0
    glb.rc = 1.8

    glb.Yukawa_P3M = 1
    glb.Yukawa_PP = 2
    glb.EGS = 3

    potential_type = params.potential[0].type
    glb.pot_calc_algrthm = params.potential[0].algorithm

    glb.N = 0
    for i, load in enumerate(params.load):
        glb.N += params.load[i].Num  # currently same as glb.N

    glb.dt = params.control[0].dt
    glb.Neq = params.control[0].Neq

    glb.Nt = params.control[0].Nstep 
    if( params.control[0].BC == "periodic"):
        glb.PBC = 1

    if(params.Langevin):
        glb.Langevin_model = params.Langevin[0].type
        glb.g_0 = params.Langevin[0].gamma

    glb.snap_int = params.control[0].dump_step
    glb.write_xyz = params.control[0].writexyz
    glb.seed_int = params.load[0].rand_seed
    glb.Zi = params.species[0].charge
    glb.ni = params.load[0].np

    glb.rc = params.potential[0].rc
    glb.Te =        params.species[0].T0
    glb.T_desired = params.species[0].T0
    glb.verbose = params.control[0].verbose


# pre-factors as a result of using 'reduced' units
    glb.af = 1.0/3.0                          # acceleration factor for Yukawa units
    glb.uf = 1.0                              # potential energy factor for Yukawa units
    glb.kf = 1.5                              # kinetic energy factor for Yukawa units

    return


def Yukawa_units(params):
    if not (params.potential[0].type == "Yukawa" or  params.potential[0].type == "yukawa"):
        print("Yukawa units are only for Yukawa potential.")
        sys.exit()

    glb.Gamma = params.potential[0].Gamma
    glb.kappa = params.potential[0].kappa

    glb.units = "Yukawa"
    const.elec_charge = 1
    const.elec_mass   = 1
    const.proton_mass   = 1
    const.kb      = 1
    const.e_0     = 1.

    glb.ni = 1
    glb.wp = 1
    glb.ai = 1
    glb.Zi = 1
    glb.q1 = 1
    glb.q2 = 1
    glb.mi = const.proton_mass

    if(params.potential[0].algorithm == "PP"):
        glb.force = force.Yukawa_force_PP
        glb.potential_type = glb.Yukawa_PP
        glb.p3m_flag = 0

    elif(params.potential[0].algorithm == "P3M"):
        glb.force = force.Yukawa_force_P3M
        glb.potential_type = glb.Yukawa_P3M
        glb.p3m_flag = 1  

    else:
        print("Wrong potential algorithm.")
        sys.exit()


    glb.T_desired = 1/(glb.Gamma)                # desired temperature



    '''
    Below is temporary until the paramtere optimization is done.
    '''

# Other MD parameters
    T_desired = glb.T_desired
    Nt = glb.Nt
    Neq = glb.Neq
    glb.L = glb.ai*(4.0*np.pi*glb.N/3.0)**(1.0/3.0)      # box length
    L = glb.L
    glb.Lx = L
    glb.Ly = L
    glb.Lz = L
    glb.Lv = np.array([L, L, L])              # box length vector
    glb.d = np.count_nonzero(glb.Lv)              # no. of dimensions
    glb.Lmax_v = np.array([L, L, L])
    glb.Lmin_v = np.array([0.0, 0.0, 0.0])

    glb.dq = 2.*np.pi/glb.L
    glb.q_max = 30/glb.ai
    glb.Nq = 3*int(glb.q_max/glb.dq)
    

    # Ewald parameters
    glb.G = 0.46/glb.ai
    glb.G_ew = glb.G
    glb.rc *= glb.ai

    if(params.potential[0].algorithm == "P3M"):
    # P3M parameters
        glb.Mx = 64
        glb.My = 64
        glb.Mz = 64
        glb.hx = glb.Lx/glb.Mx
        glb.hy = glb.Ly/glb.My
        glb.hz = glb.Lz/glb.Mz
        glb.p = 6
        glb.mx_max = 3
        glb.my_max = 3
        glb.mz_max = 3


        # Optimized Green's Function
        glb.G_k, glb.kx_v, glb.ky_v, glb.kz_v, glb_vars.A_pm = yukawa_gf_opt.gf_opt()

    return

def CGS_units(params):
    glb.kappa /= glb.ai
    pass

def MKS_units(params):
    glb.kappa /= glb.ai
    pass
