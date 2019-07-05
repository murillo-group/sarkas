'''
                       Sarkas

A code that executes molecular dynamics simulation for a Yukawa plasma using
the efficient Particle-Particle-Particle-Mesh algorithm for force computation.
The code constitutes a number of functions that are in separate files.

Developed by the research group of:
Professor Michael S. Murillo
murillom@msu.edu
Dept. of Computational Mathematics, Science, and Engineering,
Michigan State University
'''

# python modules
import numpy as np
import time
import sys
import os


# Importing MD modules
import S_initialize_pos_vel as initialize_pos_vel
import S_velocity_verlet as velocity_verlet
import S_yukawa_gf_opt as yukawa_gf_opt
import S_thermostat as thermostat
import S_EGS as EGS
import S_p3m as p3m
import S_read_input as read_input
import S_global_names as glb
import S_constants as const

#from S_thermostat import thermostat
from S_particles import particles
from S_verbose import verbose
from S_params import Params
from S_checkpoint import checkpoint

time_stamp = np.zeros(10)
its = 0

input_file = sys.argv[1]
# Reading MD conditions from input file
read_input.parameters(input_file)

params = Params()
params.setup(input_file)
verbose = verbose(params, glb)
checkpoint = checkpoint(params)  # For restart and pva backups.
###
Nt = params.control[0].Nstep    # number of time steps
time_stamp[its] = time.time(); its += 1

G_k, kx_v, ky_v, kz_v, A_pm = yukawa_gf_opt.gf_opt()
if(glb.potential_type == glb.EGS):
    EGS.init_parameters()

time_stamp[its] = time.time(); its += 1

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
    verbose.sim_setting_summary()   # simulation setting summary


n_q_t = np.zeros((glb.Nt, glb.Nq, 3), dtype="complex128")

# initializing the wave vector vector qv
qv = np.zeros(glb.Nq)

for iqv in range(0, glb.Nq, 3):
    iq = iqv/3.
    qv[iqv] = (iq+1.)*glb.dq
    qv[iqv+1] = (iq+1.)*np.sqrt(2.)*glb.dq
    qv[iqv+2] = (iq+1.)*np.sqrt(3.)*glb.dq

# array for temperature, total energy, kinetic energy, potential energy
t_Tp_E_K_U2 = np.zeros((1, 5))

total_num_ptcls = 0
for i, load in enumerate(params.load):
    total_num_ptcls += params.load[i].Num  # currently same as glb.N
N = total_num_ptcls

# Initializing particle positions and velocities
ptcls = particles(params, total_num_ptcls)
pos, vel = ptcls.load(glb, total_num_ptcls)

time_stamp[its] = time.time(); its += 1

# Calculating initial forces and potential energy
U, acc = p3m.force_pot(pos, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)

K = 0.5*glb.mi*np.ndarray.sum(vel**2)
Tp = (2/3)*K/float(N)/const.kb
if(glb.units == "Yukawa"):
    K *= 3
    Tp *= 3
E = K + U
print("=====T, E, K, U = ", Tp, E, K, U)

if not (params.load[0].method == "restart"):
    print("\n------------- Equilibration -------------")
    for it in range(glb.Neq):
        
        #thermostat.update(pos, vel, acc, params, it)

        pos, vel, acc, U = thermostat.vscale(pos, vel, acc, glb.T_desired, it, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)





        K = 0.5*glb.mi*np.ndarray.sum(vel**2)
        Tp = (2/3)*K/float(N)/const.kb
        if(glb.units == "Yukawa"):
            K *= 3
            Tp *= 3

        E = K + U
        if(it % glb.snap_int == 0 and glb.verbose):
            print("Equilibration: timestep, T, E, K, U = ", it, Tp, E, K, U)

time_stamp[its] = time.time(); its += 1

print("\n------------- Production -------------")
# Opening files for writing particle positions, velcoities and forces
f_output_E = open("t_T_totalE_kinE_potE.out", "w")
f_xyz = open("p_v_a.xyz", "w")

if (params.load[0].method == "restart"):
    it_start = params.load[0].restart_step+1
else:
    it_start = 0

for it in range(it_start, Nt):

    pos, vel, acc, U = velocity_verlet.update(pos, vel, acc, Z, G_k, kx_v, ky_v, kz_v, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)

    K = 0.5*glb.mi*np.ndarray.sum(vel**2)
    Tp = (2/3)*K/float(N)/const.kb
    if(glb.units == "Yukawa"):
        K *= 3.
        Tp *= 3.

    E = K + U

    if(it % glb.snap_int == 0 and glb.verbose):
        print("Production: timestep, T, E, K, U = ", it, Tp, E, K, U)

    t_Tp_E_K_U = np.array([glb.dt*it, Tp, E, K, U])
    t_Tp_E_K_U2[:] = t_Tp_E_K_U

    # writing particle positions and velocities to file
    if(it % params.control[0].dump_step == 0):
        checkpoint.dump(pos, vel, acc, it)

    # Spatial Fourier transform
    for iqv in range(glb.Nq):
        q_p = qv[iqv]
        n_q_t[it, iqv, 0] = np.sum(np.exp(-1j*q_p*pos[:, 0]))
        n_q_t[it, iqv, 1] = np.sum(np.exp(-1j*q_p*pos[:, 1]))
        n_q_t[it, iqv, 2] = np.sum(np.exp(-1j*q_p*pos[:, 2]))

    np.savetxt(f_output_E, t_Tp_E_K_U2)

    if glb.write_xyz == 1:
        f_xyz.writelines("{0:d}\n".format(N))
        f_xyz.writelines("x y z vx vy vz ax ay az\n")
        np.savetxt(f_xyz, irp)

np.save("n_qt", n_q_t)

# closing output files
f_output_E.close()
f_xyz.close()
# saving last positions, velocities and accelerations
checkpoint.dump(pos, vel, acc, Nt)

time_stamp[its] = time.time(); its += 1

if(glb.verbose):
    verbose.time_stamp(time_stamp)

# end of the code
