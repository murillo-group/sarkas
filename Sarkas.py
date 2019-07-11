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
import S_EGS as EGS
import S_p3m as p3m
import S_read_input as read_input
import S_global_names as glb
import S_constants as const

from S_thermostat import Thermostat
from S_integrator import Integrator
from S_particles import Particles
from S_verbose import Verbose
from S_params import Params
from S_checkpoint import Checkpoint

time_stamp = np.zeros(10)
its = 0

input_file = sys.argv[1]
# Reading MD conditions from input file
read_input.parameters(input_file)

params = Params()
params.setup(input_file)
verbose = Verbose(params, glb)
checkpoint = Checkpoint(params)  # For restart and pva backups.
integrator = Integrator(params, glb)
thermostat = Thermostat(params, glb)

###
Nt = params.control[0].Nstep    # number of time steps
time_stamp[its] = time.time(); its += 1

if(glb.potential_type == glb.EGS):
    EGS.init_parameters()

time_stamp[its] = time.time(); its += 1

#######################
# this variable will be moved to observable class
n_q_t = np.zeros((glb.Nt, glb.Nq, 3), dtype="complex128")
########################

if(glb.verbose):
    verbose.sim_setting_summary()   # simulation setting summary

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
ptcls = Particles(params, total_num_ptcls)
ptcls.load(glb, total_num_ptcls)

time_stamp[its] = time.time(); its += 1

# Calculating initial forces and potential energy
U = p3m.force_pot(ptcls)

K = 0.5*glb.mi*np.ndarray.sum(ptcls.vel**2)
Tp = (2/3)*K/float(N)/const.kb
if(glb.units == "Yukawa"):
    K *= 3
    Tp *= 3
E = K + U
print("=====T, E, K, U = ", Tp, E, K, U)

if not (params.load[0].method == "restart"):
    print("\n------------- Equilibration -------------")
    for it in range(glb.Neq):
        U = thermostat.update(ptcls, it)
        K = 0.5*glb.mi*np.ndarray.sum(ptcls.vel**2)
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

    U = integrator.update(ptcls)

    K = 0.5*glb.mi*np.ndarray.sum(ptcls.vel**2)
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
        checkpoint.dump(ptcls.pos, ptcls.vel, ptcls.acc, it)

    # Spatial Fourier transform
    # will be move to observable class
    if(1):
        for iqv in range(glb.Nq):
            q_p = qv[iqv]
            n_q_t[it, iqv, 0] = np.sum(np.exp(-1j*q_p*ptcls.pos[:, 0]))
            n_q_t[it, iqv, 1] = np.sum(np.exp(-1j*q_p*ptcls.pos[:, 1]))
            n_q_t[it, iqv, 2] = np.sum(np.exp(-1j*q_p*ptcls.pos[:, 2]))

    np.savetxt(f_output_E, t_Tp_E_K_U2)

    if glb.write_xyz == 1:
        f_xyz.writelines("{0:d}\n".format(N))
        f_xyz.writelines("x y z vx vy vz ax ay az\n")
        np.savetxt(f_xyz, irp)

# will be moved to observable class
np.save("n_qt", n_q_t)

# closing output files
f_output_E.close()
f_xyz.close()

# saving last positions, velocities and accelerations
checkpoint.dump(ptcls.pos, ptcls.vel, ptcls.acc, Nt)

time_stamp[its] = time.time(); its += 1

if(glb.verbose):
    verbose.time_stamp(time_stamp)

# end of the code
