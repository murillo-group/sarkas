#############################################################################################
#                       Sarkas                                                              #
#                                                                                           #
# A code that executes molecular dynamics simulation for a Yukawa plasma                    #
# using the efficient Particle-Particle-Particle-Mesh algorithm for force computation.      #
# The code constitutes a number of functions that are in separate files.                    #
#                                                                                           #
# Developed by the research group of:                                                       #
#  Professor Michael S. Murillo                                                             #
#  murillom@msu.edu                                                                         #
# Dept. of Computational Mathematics, Science, and Engineering,                             #
# Michigan State University                                                                 #
#############################################################################################

# python modules
import numpy as np
import time
import sys
import os

time_stamp = np.zeros(10)
its = 0
time_stamp[its] = time.time(); its += 1

# Importing MD modules
import S_initialize_pos_vel as initialize_pos_vel
import S_velocity_verlet as velocity_verlet
import S_thermostat as thermostat
import S_yukawa_gf_opt as yukawa_gf_opt
import S_EGS as EGS
import S_p3m as p3m
import S_read_input as read_input
import S_global_names as glb
import S_constants as const
from S_particles import particles
from S_verbose import verbose
from S_params import Params
from S_checkpoint import checkpoint

input_file = sys.argv[1]
# Reading MD conditions from input file
read_input.parameters(input_file)
####
# 
params = Params()
params.setup(input_file)
verbose = verbose(params, glb)
checkpoint = checkpoint(params)
###
#glb.Zi = 1

Zi = glb.Zi
q1 = glb.q1
q2 = glb.q2
ni = glb.ni
wp = glb.wp
ai = glb.ai
mi = const.pMass

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
L = ai*(4.0*np.pi*glb.N/3.0)**(1.0/3.0)      # box length
glb.Lx = L
glb.Ly = L
glb.Lz = L
glb.Lv = np.array([L, L, L])              # box length vector
glb.d = np.count_nonzero(glb.Lv)              # no. of dimensions
glb.Lmax_v = np.array([L, L, L]) 
glb.Lmin_v = np.array([0.0, 0.0, 0.0])

#Ewald parameters
glb.G = 0.46/ai
glb.G_ew = glb.G
glb.rc *= ai

#P3M parameters
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

time_stamp[its] = time.time(); its += 1

G_k, kx_v, ky_v, kz_v, A_pm = yukawa_gf_opt.gf_opt()
if(glb.potential_type == glb.EGS):
  EGS.init_parameters()

time_stamp[its] = time.time(); its += 1
glb.kappa /=glb.ai

# pre-factors as a result of using 'reduced' units
glb.af = 1.0/3.0                          # acceleration factor for Yukawa units
glb.uf = 1.0                              # potential energy factor for Yukawa units
glb.kf = 1.5                              # kinetic energy factor for Yukawa units
af = glb.af
uf = glb.uf
kf = glb.kf
N = glb.N
dt = glb.dt
glb.p3m_flag = 1 # default is P3M
if(glb.pot_calc_algrthm == "PP"):
  glb.p3m_flag = 0

# Particle positions and velocities array
pos = np.zeros((glb.N, glb.d))
vel = np.zeros_like(pos)
acc = np.zeros_like(pos)
Z = np.ones(glb.N)

acc_s_r = np.zeros_like(pos)
acc_fft = np.zeros_like(pos)

rho_r = np.zeros((glb.Mz, glb.My, glb.Mx))
E_x_p = np.zeros(glb.N)
E_y_p = np.zeros(glb.N)
E_z_p = np.zeros(glb.N)

# F(k,t): Spatial Fourier transform of density fluctutations
if(glb.verbose):
    verbose.sim_setting_summary() # simulation setting summary

dq = 2.*np.pi/L
q_max = 30/ai
glb.Nq = 3*int(q_max/dq)
Nq = glb.Nq   # 3 is for x, y, and z commponent

n_q_t = np.zeros((Nt, Nq, 3),dtype='complex128') #

# initializing the wave vector vector qv
qv = np.zeros(Nq)

for iqv in range(0, Nq, 3):
    iq = iqv/3.
    qv[iqv] = (iq+1.)*dq
    qv[iqv+1] = (iq+1.)*np.sqrt(2.)*dq
    qv[iqv+2] = (iq+1.)*np.sqrt(3.)*dq

#array for temperature, total energy, kinetic energy, potential energy
t_Tp_E_K_U2 = np.zeros((1,5))


total_num_ptcls = 0
for i, load in enumerate(params.load):
    total_num_ptcls += params.load[i].Num  # currently same as glb.N

# Initializing particle positions and velocities
ptcls = particles(params, total_num_ptcls)
pos, vel = ptcls.load(glb, total_num_ptcls)

time_stamp[its] = time.time(); its += 1

# Calculating initial forces and potential energy
U, acc = p3m.force_pot(pos, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)


K = 0.5*mi*np.ndarray.sum(vel**2)
Tp = (2/3)*K/float(N)/const.kb
if(glb.units == "Yukawa"):
    K *= 3
    Tp *= 3
E = K + U
print("=====T, E, K, U = ", Tp, E, K, U)

if not (params.load[0].method == "restart"):
    print('\n------------- Equilibration -------------')
    for it in range(Neq):
        pos, vel, acc, U = thermostat.vscale(pos, vel, acc, T_desired, it, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)
#---------------
        K = 0.5*mi*np.ndarray.sum(vel**2)
        Tp = (2/3)*K/float(N)/const.kb
        if(glb.units == "Yukawa"):
            K *= 3
            Tp *= 3

        E = K + U
        if(it%glb.snap_int == 0 and glb.verbose):
            print("Equilibration: timestep, T, E, K, U = ", it, Tp, E, K, U)

time_stamp[its] = time.time(); its += 1

print('\n------------- Production -------------')
# Opening files for writing particle positions, velcoities and forces
f_output_E = open('t_T_totalE_kinE_potE.out','w')
f_xyz = open('p_v_a.xyz','w')

#print('time - total energy - kinetic energy - potential energy')
if (params.load[0].method == "restart"):
    it_start = params.load[0].restart_step+1
else:
    it_start = 0

for it in range(it_start, Nt):

    pos, vel, acc, U = velocity_verlet.update(pos, vel, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)

    K = 0.5*mi*np.ndarray.sum(vel**2)
    Tp = (2/3)*K/float(N)/const.kb
    if(glb.units == "Yukawa"):
        K *= 3.
        Tp *= 3.

    E = K + U

    if(it%glb.snap_int == 0 and glb.verbose):
        print("Production: timestep, T, E, K, U = ", it, Tp, E, K, U)
    
    t_Tp_E_K_U = np.array([dt*it, Tp, E, K, U])
    t_Tp_E_K_U2[:] = t_Tp_E_K_U


    # writing particle positions and velocities to file
    if(it%params.control[0].dump_step == 0):
        checkpoint.dump(pos, vel, acc, it)
    
    # Spatial Fourier transform
    for iqv in range(Nq):
        q_p = qv[iqv]
        n_q_t[it,iqv,0] = np.sum(np.exp(-1j*q_p*pos[:,0]))
        n_q_t[it,iqv,1] = np.sum(np.exp(-1j*q_p*pos[:,1]))
        n_q_t[it,iqv,2] = np.sum(np.exp(-1j*q_p*pos[:,2]))
    
    np.savetxt(f_output_E, t_Tp_E_K_U2)

    if glb.write_xyz == 1:
        f_xyz.writelines('{0:d}\n'.format(N))
        f_xyz.writelines('x y z vx vy vz ax ay az\n')
        np.savetxt(f_xyz,irp)

np.save('n_qt',n_q_t)

# closing output files        
f_output_E.close()
f_xyz.close()
# saving last positions, velocities and accelerations
checkpoint.dump(pos, vel, acc, Nt)

time_stamp[its] = time.time(); its += 1

if(glb.verbose):
    verbose.time_stamp(time_stamp)

# end of the code
