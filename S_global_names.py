'''
S_global_names.py

setting global variables and physical constants
'''
import pdb
import numpy as np
import sys
import fdint

import S_global_names as glb # self import. I know it is weird, but it works.
import S_constants as const
import S_yukawa_gf_opt as yukawa_gf_opt
import S_force as force


# read input data from yukawa_MD_p3m.in
def init(params):

    units = params.control[0].units

    # Setup units-indep. vars.
    Aux(params)

    # Setup units-dep. vars.
    if(units == "Yukawa" or units == "yukawa"):
        Yukawa_units(params)

    elif(units == "CGS" or units == "cgs"):
        CGS_units(params)

    elif(units == "MKS" or units == "mks"):
        MKS_units(params)

    else:
        print("No such units are available")
        sys.exit()

    # Setup for Yukawa potential related vars.
    if (params.potential[0].type == "Yukawa" or  params.potential[0].type == "yukawa"):
        Yukawa_aux(params)

    # Setup for a brute-force technique. This is only for testing new integrators. 
    # Do not use this part any other purpose unless you understand what you are doing!
    if (params.potential[0].algorithm == "Brute"):
        Brute_aux(params)

    return

def Aux(params):

    glb.verbose = 0
    glb.rc = 1.8

    glb.Yukawa_P3M = 1
    glb.Yukawa_PP = 2
    glb.EGS = 3
    glb.Coulomb = 4

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
    glb.ni = params.species[0].ni

    glb.rc = params.potential[0].rc
#    glb.Te =        params.species[0].Te
    glb.T_desired = params.species[0].Ti
    glb.verbose = params.control[0].verbose

    return

def Brute_aux(params):
        glb.force = force.Coulomb_force
        glb.potential_type = glb.Coulomb


def Yukawa_aux(params):
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
    const.epsilon_0     = 1.

    glb.ni = 1
    glb.wp = 1
    glb.ai = 1
    glb.Zi = 1
    glb.q1 = 1
    glb.q2 = 1
    glb.mi = const.proton_mass

    glb.T_desired = 1/(glb.Gamma)                # desired temperature

# pre-factors as a result of using 'reduced' units
    glb.af = 1.0/3.0                          # acceleration factor for Yukawa units
    glb.uf = 1.0                              # potential energy factor for Yukawa units
    glb.kf = 1.5                              # kinetic energy factor for Yukawa units

    return

def CGS_units(params):

    const.elec_charge = 4.80320425e-10
    const.elec_mass = 9.10938356e-28
    const.proton_mass = 1.672621898e-24
    const.kb = 1.38064852e-16
    const.epsilon_0 = 1.
    const.hbar = 1.05e-27

    glb.units = "cgs"
    glb.q1 = const.elec_charge*glb.Zi
    glb.q2 = const.elec_charge*glb.Zi
    glb.ni = params.species[0].ni
    glb.ai = (3/(4*np.pi*glb.ni))**(1./3.)
    glb.wp = np.sqrt(glb.q1*glb.q2*glb.ni/const.proton_mass/const.epsilon_0)
    glb.Zi = params.species[0].charge
    glb.mi = params.species[0].mass

    if (params.potential[0].type == "Yukawa" or  params.potential[0].type == "yukawa"):
        ne = params.species[0].charge*params.species[0].ni # when ne is not defined 
        Te = params.species[0].Ti

        #kF = (3*np.pi**2*ne)**(1./3.)
        #Ef = const.hbar**2*kF**2/(2*const.elec_mass)  # Fermi Energy
        #lambda_TF = np.sqrt( np.sqrt((const.kb*Te)**2 + 4./9.*Ef**2 )/(4*np.pi*ne*const.elec_charge**2))
        # Using MKS relation to obtain kappa and Gamma
        k = 1.38064852e-23
        e = 1.602176634e-19
        hbar = 1.05e-34
        m_e = 9.10938356e-31
        e_0 = 8.854187817e-12
        T  = Te
        Zi = glb.Zi
        n = ne*1.e6
        fdint_fdk_vec = np.vectorize(fdint.fdk)
        fdint_dfdk_vec = np.vectorize(fdint.dfdk)
        fdint_ifd1h_vec = np.vectorize(fdint.ifd1h)
        beta = 1/(k*T)
        eta = fdint_ifd1h_vec(np.pi**2*(beta*hbar**2/(m_e))**(3/2)/np.sqrt(2)*n) #eq 4 inverted

        #kF = (3*np.pi**2*ne)**(1./3.)
        #E_F = (hbar**2/(2*m_e))*(3*np.pi**2*n)**(2/3)
        lambda_TF = np.sqrt((4*np.pi**2*e_0*hbar**2)/(m_e*e**2)*np.sqrt(2*beta*hbar**2/m_e)/(4*fdint_fdk_vec(k=-0.5, phi=eta))) #eq 10
        glb.ai = (3./(4*np.pi*ni))**(1./3)
        glb.kappa = (glb.ai*1e-2)/lambda_TF
        glb.Gamma = (glb.Zi*const.elec_charge)**2/(glb.ai*const.kb*Te)
        glb.af = (glb.q1*glb.q2/glb.mi) # acceleration factor for cgs units
        glb.uf = glb.q1*glb.q2  # potential factor for cgs units

    return

def MKS_units(params):

    const.elec_charge = 1.602176634e-19
    const.elec_mass = 9.10938356e-31
    const.proton_mass = 1.672621898e-27
    const.kb = 1.38064852e-23
    const.epsilon_0 = 8.854187817e-12
    const.hbar= 1.05e-34

    glb.units = "mks"
    glb.q1 = const.elec_charge*glb.Zi
    glb.q2 = const.elec_charge*glb.Zi
    glb.ni = params.species[0].ni
    glb.ai = (3/(4*np.pi*glb.ni))**(1./3.)
    glb.wp = np.sqrt(glb.q1*glb.q2*glb.ni/const.proton_mass/const.epsilon_0)
    glb.Zi = params.species[0].charge
    glb.mi = params.species[0].mass

    if (params.potential[0].type == "Yukawa" or  params.potential[0].type == "yukawa"):
        ne = params.species[0].charge*params.species[0].ni # when ne is not defined 
        Te = params.species[0].Ti

        k = const.kb
        e = const.elec_charge
        hbar = const.hbar
        m_e = const.elec_mass
        e_0 = const.epsilon_0
        T  = Te
        Zi = glb.Zi
        n = ne
        fdint_fdk_vec = np.vectorize(fdint.fdk)
        fdint_dfdk_vec = np.vectorize(fdint.dfdk)
        fdint_ifd1h_vec = np.vectorize(fdint.ifd1h)
        beta = 1/(k*T)
        eta = fdint_ifd1h_vec(np.pi**2*(beta*hbar**2/(m_e))**(3/2)/np.sqrt(2)*n) #eq 4 inverted

        #kF = (3*np.pi**2*ne)**(1./3.)
        #E_F = (hbar**2/(2*m_e))*(3*np.pi**2*n)**(2/3)
        lambda_TF = np.sqrt((4*np.pi**2*e_0*hbar**2)/(m_e*e**2)*np.sqrt(2*beta*hbar**2/m_e)/ \
                (4*fdint_fdk_vec(k=-0.5, phi=eta))) #eq 10
        glb.ai = (3./(4*np.pi*ni))**(1./3)
        glb.kappa = glb.ai/lambda_TF
        glb.Gamma = (glb.Zi*const.elec_charge)**2/(4*np.pi*const.epsilon_0*glb.ai*const.kb*Te)
        glb.af = (glb.q1*glb.q2/glb.mi/(4*np.pi*const.epsilon_0)) # acceleration factor for mks units
        glb.uf = glb.q1*glb.q2/(4*np.pi*const.epsilon_0)  # potential factor for mks units

    return
