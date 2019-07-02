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

DEBUG = 0
t1 = time.time()

# Importing MD modules
import S_read as read
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


input_file = sys.argv[1]
# Reading MD conditions from input file
read_input.parameters(input_file)
####
# 
params = Params()
params.setup(input_file)
verbose = verbose(params, glb)
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

t2 = time.time()
#if(glb.potential_type == glb.Yukawa_P3M):
G_k, kx_v, ky_v, kz_v, A_pm = yukawa_gf_opt.gf_opt()
if(glb.potential_type == glb.EGS):
  EGS.init_parameters()

t3 = time.time()
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
    verbose.output() # simulation setting

dq = 2.*np.pi/L
q_max = 30/ai
glb.Nq = 3*int(q_max/dq)
Nq = glb.Nq   # 3 is for x, y, and z commponent

n_q_t = np.zeros((Nt, Nq, 3),dtype='complex128') #

# initializing the q vector
qv = np.zeros(Nq)

for iqv in range(0, Nq, 3):
    iq = iqv/3.
    qv[iqv] = (iq+1.)*dq
    qv[iqv+1] = (iq+1.)*np.sqrt(2.)*dq
    qv[iqv+2] = (iq+1.)*np.sqrt(3.)*dq

#array for temperature, total energy, kinetic energy, potential energy
t_Tp_E_K_U2 = np.zeros((1,5))

restartDir = "Restart"
if not (os.path.exists(restartDir)):
    os.mkdir(restartDir)

total_num_ptcls = 0
for i, load in enumerate(params.load):
    total_num_ptcls += params.load[i].Num  # currently same as glb.N

# Initializing particle positions and velocities
ptcls = particles(params, total_num_ptcls)
pos, vel = ptcls.load(glb, total_num_ptcls)

t4 = time.time()
# Calculating initial forces and potential energy
U, acc = p3m.force_pot(pos, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)

K = 0.5*mi*np.ndarray.sum(vel**2)
Tp = (2/3)*K/float(N)/const.kb
if(glb.units == "Yukawa"):
    K *= 3
    Tp *= 3
E = K + U
print("=====T, E, K, U = ", Tp, E, K, U)

print('\n------------- Equilibration -------------')
#print('time - temperature')
for it in range(Neq):
#    print("it = ", it)
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
    
t5 = time.time()

print('\n------------- Production -------------')
# Opening files for writing particle positions, velcoities and forces
f_output = open('p_v_a.out','w')
f_output_E = open('t_T_totalE_kinE_potE.out','w')
f_xyz = open('p_v_a.xyz','w')

#print('time - total energy - kinetic energy - potential energy')

for it in range(Nt):
    
    pos, vel, acc, U = velocity_verlet.update_Langevin(pos, vel, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)

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
    
    # Spatial Fourier transform
    for iqv in range(Nq):
        q_p = qv[iqv]
        n_q_t[it,iqv,0] = np.sum(np.exp(-1j*q_p*pos[:,0]))
        n_q_t[it,iqv,1] = np.sum(np.exp(-1j*q_p*pos[:,1]))
        n_q_t[it,iqv,2] = np.sum(np.exp(-1j*q_p*pos[:,2]))
    
    # writing particle positions and velocities to file
    if glb.write_output == 1:
        if np.mod(it+1, glb.snap_int) == 0:
            irp = np.hstack((pos, vel, acc))
            np.savetxt(f_output, irp)
            np.savetxt(f_output_E, t_Tp_E_K_U2)
            
            if glb.write_xyz == 1:
                f_xyz.writelines('{0:d}\n'.format(N))
                f_xyz.writelines('x y z vx vy vz ax ay az\n')
                np.savetxt(f_xyz,irp)

np.save('n_qt',n_q_t)

# closing output files        
f_output.close()
f_output_E.close()
f_xyz.close()
# saving last positions, velocities and accelerations
irp2 = np.hstack((pos,vel,acc))
np.savetxt('p_v_a_final.out',irp2)

t6 = time.time()

if(glb.verbose):
    print('Time for importing required libraries = ', t2-t1)
    print('Time for computing converged Greens function = ', t3-t2)
    print('Time for initialization = ', t4-t3)
    print('Time for equilibration = ', t5-t4)
    print('Time for production = ', t6-t5)
    print('Total elapsed time = ', t6-t1)

# end of the code
