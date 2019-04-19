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




    if(1):
        params = Params()
        params.setup(input_file)
        potential_type = params.potential[0].type
        units = params.control[0].units
        glb.pot_calc_algrthm = params.potential[0].algorithm
        glb.Gamma = params.potential[0].Gamma
        glb.kappa = params.potential[0].kappa

        glb.N = params.load[0].Num
        glb.dt = params.control[0].dt
        glb.Neq = params.control[0].Neq

        glb.Nt = params.control[0].Nstep 
        if( params.control[0].BC == "periodic"):
            glb.PBC = 1

        glb.Langevin_model = params.Langevin[0].type
        glb.g_0 = params.Langevin[0].gamma
        glb.snap_int = params.control[0].dump_step
        glb.init = params.control[0].init
        glb.write_output = params.control[0].writeout
        glb.write_xyz = params.control[0].writexyz
        glb.seed_int = params.control[0].seed
        glb.Zi = params.species[0].charge
        glb.ni = params.load[0].np

        glb.rc = params.potential[0].rc
        glb.Te =        params.species[0].T0
        glb.T_desired = params.species[0].T0
        glb.verbose = params.control[0].verbose
            
        
        
#  c = [np.array(map(None, line.split())) for line in open('yukawa_MD_p3m.in')]
#  print(len(c))
    if(0):
     for line in open(input_file):  
        c = line.split()
        try:
            c[0]
# check the potential system
            if(c[0] == "potential"):
              if(c[1] == "Yukawa"): 
                potential_type = "Yukawa"

              elif(c[1] == "EGS"): 
                potential_type = "EGS"
                units = "MKS"

              else :
                print("Check your input file. Only Yukawa or EGS potential is supported!")

                sys.exit(0)

            if(c[0] == "units"):
                units = str(c[1])
            if(c[0] == "pot_calc_algrthm"):
              glb.pot_calc_algrthm = c[1]

            if(c[0] == "Gamma"):
              glb.Gamma = float(c[1])

            if(c[0] == "kappa"):
              glb.kappa = float(c[1])

            if(c[0] == "Np"):
              glb.N = int(c[1])

            if(c[0] == "dt"):
              glb.dt = float(c[1])

            if(c[0] == "N_eq"):
              glb.Neq = int(c[1])

            if(c[0] == "N_post_eq"):
              glb.Nt = int(c[1])

            if(c[0] == "PBC"):
              glb.PBC = int(c[1])

            if(c[0] == "Langevin"):
              glb.Langevin_model = str(c[1])
              if(c[1] == "BBK"):
                glb.g_0 = float(c[2])

            if(c[0] == "snap_int"):
              glb.snap_int = int(c[1])

            if(c[0] == "init"):
              glb.init = int(c[1])

            if(c[0] == "write_output"):
              glb.write_output = int(c[1])

            if(c[0] == "write_xyz"):
              glb.write_xyz = int(c[1])

            if(c[0] == "seed"):
              glb.seed_int = int(c[1])

            if(c[0] == "Zi"):
              glb.Zi = float(c[1])

            if(c[0] == "ni"):
              glb.ni = float(c[1])

#            if(c[0] == "Nq"):
#              glb.Nq = int(c[1])

            if(c[0] == "rc"):
              glb.rc = float(c[1])

            if(c[0] == "Ti"):
              glb.T_desired = float(c[1])

            if(c[0] == "Te"):
              glb.Te = float(c[1])

            if(c[0] == "verbose"):
              glb.verbose = int(c[1])


        except IndexError:
           break;




    if(units == "yukawa" or units =="Yukawa"):
        glb.units     = "Yukawa"
        const.eCharge = 1
        const.eMass   = 1
        const.pMass   = 1
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
        const.eCharge = 4.80320425e-10
        const.eMass   = 9.10938356e-28
        const.pMass   = 1.672621898e-24
        const.kb      = 1.38064852e-16
        const.hbar    = 1.05e-27
        const.eps_0     = 1.

        glb.q1 = const.eCharge*glb.Zi
        glb.q2 = const.eCharge*glb.Zi
        glb.ai = (3/(4*np.pi*glb.ni))**(1./3.)
        glb.wp = np.sqrt(4*np.pi*glb.q1*glb.q2*glb.ni/const.pMass)
        #print("wp: {0:15E}, ai: {1:15E}".format(glb.dt, glb.ai))
        #print("g0: {0:17E}".format(glb.wp*0.25))

    if(units == "mks" or units =="MKS"):
        glb.units     = "mks"
        const.eCharge = 1.602176634e-19
        const.eMass   = 9.10938356e-31
        const.pMass   = 1.672621898e-27
        const.kb      = 1.38064852e-23
        const.eps_0   = 8.854187817e-12
        const.hbar    = 1.05e-34

        glb.q1 = const.eCharge*glb.Zi
        glb.q2 = const.eCharge*glb.Zi
        glb.ai = (3/(4*np.pi*glb.ni))**(1./3.)
        glb.wp = np.sqrt(glb.q1*glb.q2*glb.ni/const.pMass/const.eps_0)
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
    return

#  return Gamma, kappa, N, dt, Neq, Nt, PBC, Langevin_model, g_0, snap_int, init, \
#         write_output, write_xyz, seed_int
