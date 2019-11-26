'''
S_global_names.py

setting global variables
'''
import numpy as np
import sys

import S_global_names as glb # self import. I know it is weird, but it works.


def init(params):

    glb.verbose = 0
    glb.rc = 1.8

    potential_type = params.Potential.type
    glb.pot_calc_method = params.Potential.method

    glb.N = params.total_num_ptcls 
#    glb.d = params.d

    glb.dt = params.Control.dt
    glb.Neq = params.Control.Neq

    glb.Nt = params.Control.Nstep 
    if( params.Control.BC == "periodic"):
        glb.PBC = 1

    if(params.Langevin.on):
        glb.Langevin_model = params.Langevin.type
        glb.g_0 = params.Langevin.gamma

    glb.snap_int = params.Control.dump_step
    glb.write_xyz = params.Control.writexyz
    glb.seed_int = params.load_rand_seed
    glb.Z = np.zeros(params.num_species)
    glb.n = np.zeros(params.num_species)

    for ic in range(params.num_species):
        if hasattr(params.species[ic], "Z"):
            glb.Z[ic] = params.species[ic].Z
            glb.n[ic] = params.species[ic].num_density

    glb.rc = params.Potential.rc
    glb.LL_on = params.Potential.LL_on
    glb.potential_matrix = params.Potential.matrix
#    glb.Te =        params.species.Te
    glb.T_desired = params.Ti
    glb.verbose = params.Control.verbose
    glb.force = params.force

# Other MD parameters
    T_desired = params.Ti
    Nt = params.Control.Nstep
    Neq = params.Control.Neq
    glb.ai = params.ai
    glb.L = params.L
    L = params.L
    glb.Lx = L
    glb.Ly = L
    glb.Lz = L
    glb.Lv = params.Lv              # box length vector
    glb.d = params.d
    glb.Lmax_v = params.Lmax_v
    glb.Lmin_v = params.Lmin_v

    glb.dq = params.dq
    glb.q_max = params.q_max
    glb.Nq = params.Nq

    glb.potential_method = params.Potential.method
    if(params.Potential.type == "Yukawa"):
        if(params.Potential.method == "P3M"):
        # P3M parameters for Yukawa. hardcode
            glb.G = params.P3M.G
            glb.G_ew = params.P3M.G_ew
            glb.Mx = params.P3M.Mx
            glb.My = params.P3M.My
            glb.Mz = params.P3M.Mz
            glb.hx = params.Lx/params.P3M.Mx
            glb.hy = params.Ly/params.P3M.My
            glb.hz = params.Lz/params.P3M.Mz
            glb.p = params.P3M.p
            glb.mx_max = params.P3M.mx_max
            glb.my_max = params.P3M.my_max
            glb.mz_max = params.P3M.mz_max

            # Optimized Green's Function
            glb.G_k = params.P3M.G_k
            glb.kx_v = params.kx_v
            glb.ky_v = params.ky_v
            glb.kz_v = params.kz_v
            glb.A_pm = params.A_pm

    return
