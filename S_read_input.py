import numpy as np
import sys
import S_global_names as glb
import S_constants as const
from S_params import Params

# read input data from yukawa_MD_p3m.in
def parameters(input_file):

    units = "Yukawa"
    glb.Gamma = -1.
    glb.kappa = -1.
    write_output = 1
    write_xyz = 0
    PBC = 1
    snap_int = 1000
    init = 1
    seed_int = 1

    glb.verbose = 0
    glb.rc = 1.8
    glb.T_disired = -1.
    glb.Te = -1

    glb.Yukawa_P3M = 1
    glb.Yukawa_PP = 2
    glb.EGS = 3

    glb.pot_type = glb.Yukawa_P3M

    params = Params()
    params.setup(input_file)
    potential_type = params.potential[0].type
    units = params.control[0].units
    glb.pot_calc_algrthm = params.potential[0].algorithm
    glb.Gamma = params.potential[0].Gamma
    glb.kappa = params.potential[0].kappa

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
    #glb.init = params.control[0].init
    glb.write_xyz = params.control[0].writexyz
    glb.seed_int = params.load[0].rand_seed
    glb.Zi = params.species[0].charge
    glb.ni = params.load[0].np

    glb.rc = params.potential[0].rc
    glb.Te =        params.species[0].T0
    glb.T_desired = params.species[0].T0
    glb.verbose = params.control[0].verbose
            
    if(units == "yukawa" or units =="Yukawa"):
        glb.units     = "Yukawa"
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

    if(units == "cgs" or units == "CGS"):
        glb.units     = "cgs"
        const.elec_charge = 4.80320425e-10
        const.elec_mass   = 9.10938356e-28
        const.proton_mass   = 1.672621898e-24
        const.kb      = 1.38064852e-16
        const.hbar    = 1.05e-27
        const.eps_0     = 1.

        glb.q1 = const.elec_charge*glb.Zi
        glb.q2 = const.elec_charge*glb.Zi
        glb.ai = (3/(4*np.pi*glb.ni))**(1./3.)
        glb.wp = np.sqrt(4*np.pi*glb.q1*glb.q2*glb.ni/const.proton_mass)
        #print("wp: {0:15E}, ai: {1:15E}".format(glb.dt, glb.ai))
        #print("g0: {0:17E}".format(glb.wp*0.25))

    if(units == "mks" or units =="MKS"):
        glb.units     = "mks"
        const.elec_charge = 1.602176634e-19
        const.elec_mass   = 9.10938356e-31
        const.proton_mass   = 1.672621898e-27
        const.kb      = 1.38064852e-23
        const.eps_0   = 8.854187817e-12
        const.hbar    = 1.05e-34

        glb.q1 = const.elec_charge*glb.Zi
        glb.q2 = const.elec_charge*glb.Zi
        glb.ai = (3/(4*np.pi*glb.ni))**(1./3.)
        glb.wp = np.sqrt(glb.q1*glb.q2*glb.ni/const.proton_mass/const.eps_0)
        #print("wp: {0:15E}, ai: {1:15E}".format(glb.dt, glb.ai))

    #if(units == "cgs-eV"):
    #if(units == "mks-eV"):
    if(glb.pot_calc_algrthm == "PP" and potential_type == "Yukawa"):
        glb.potential_type = glb.Yukawa_PP

    if(glb.pot_calc_algrthm == "P3M" and potential_type == "Yukawa"):
        glb.potential_type = glb.Yukawa_P3M

    if(potential_type == "EGS"):
        glb.potential_type = glb.EGS
        glb.gamma_m = 0
        glb.gamma_p = 0
        glb.alpha = 0
        glb.nu = 0
        if(glb.T_desired < 0):
            print("You must set Ti for EGS.")
            sys.exit()
        if(glb.Te < 0):
            print("You must set Te for EGS.")
            sys.exit()

    if(glb.potential_type == "Yukawa"):
      if (glb.Gamma < 0. or glb.kappa < 0.):
        print("Check your input file. Gamma and/or kappa is wrong in the Yukawa potential.")
        sys.exit(0)

    glb.mi = const.proton_mass
    glb.kappa /= glb.ai

# pre-factors as a result of using 'reduced' units
    glb.af = 1.0/3.0                          # acceleration factor for Yukawa units
    glb.uf = 1.0                              # potential energy factor for Yukawa units
    glb.kf = 1.5                              # kinetic energy factor for Yukawa units
    glb.p3m_flag = 1    # default is P3M
    if(glb.pot_calc_algrthm == "PP"):
        glb.p3m_flag = 0


    '''
    Below is temporary until the paramtere optimization is done.
    '''

# Other MD parameters
    if(glb.potential_type == glb.Yukawa_PP or glb.potential_type == glb.Yukawa_P3M):
        if(glb.units == "Yukawa"):
            glb.T_desired = 1/(glb.Gamma)                # desired temperature

        if(glb.units == "cgs"):
            glb.T_desired = q1*q2/ai/(const.kb*glb.Gamma)                # desired temperature

        if(glb.units == "mks"):
            glb.T_desired = q1*q2/ai/(const.kb*glb.Gamma*4*np.pi*const.eps_0)                # desired temperature

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

    return
