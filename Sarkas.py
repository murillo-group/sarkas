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

# Python modules
import numpy as np
import time
import sys
import os

# Importing MD modules, non class
import S_calc_force as calc_force 
import S_global_names as glb
import S_constants as const

# import MD modules, class
from S_thermostat import Thermostat
from S_integrator import Integrator
from S_particles import Particles
from S_verbose import Verbose
from S_params import Params
from S_checkpoint import Checkpoint
time_stamp= np.zeros(10)
its = 0
time_stamp[its] = time.time(); its += 1

input_file = sys.argv[1]

params = Params()
params.setup(input_file)                # Read initial conditions and setup parameters
glb.init(params)                        # Setup global variables
verbose = Verbose(params, glb)
checkpoint = Checkpoint(params)         # For restart and pva backups.
calc_force = calc_force.force_pot
integrator = Integrator(params, glb)    # Setup a velocity integrator
thermostat = Thermostat(params, glb)    # Setup a themrostat
###
Nt = params.Control.Nstep    # number of time steps

#######################
# this variable will be moved to observable class
n_q_t = np.zeros((glb.Nt, glb.Nq, 3), dtype="complex128")

# initializing the wave vector vector qv
qv = np.zeros(glb.Nq)

for iqv in range(0, glb.Nq, 3):
    iq = iqv/3.
    qv[iqv] = (iq+1.)*glb.dq
    qv[iqv+1] = (iq+1.)*np.sqrt(2.)*glb.dq
    qv[iqv+2] = (iq+1.)*np.sqrt(3.)*glb.dq
########################
if(glb.verbose):
    verbose.sim_setting_summary()   # simulation setting summary

# array for temperature, total energy, kinetic energy, potential energy
t_Tp_E_K_U2 = np.zeros((1, 5))

N = params.total_num_ptcls

# Initializing particle positions and velocities
ptcls = Particles(params)
time_stamp[its] = time.time(); its += 1
ptcls.load()
time_stamp[its] = time.time(); its += 1
N = len(ptcls.pos[:, 0])

# Calculating initial forces and potential energy
U = calc_force(ptcls)

if(params.Control.units == "Yukawa"):
    U *= 3

K = 0
species_start = 0
for i in range(params.num_species):
    species_end = species_start + params.species[i].num
    K += 0.5*params.species[i].mass*np.ndarray.sum(ptcls.vel[species_start:species_end, :]**2)
    species_start = species_end

Tp = (2/3)*K/float(N)/const.kb

E = K + U
P = np.ndarray.sum(ptcls.pos**2)

print("intial: T, E, K, U = ", Tp, E, K, U)

if not (params.load_method == "restart"):
#    print("\n------------- Equilibration -------------")
    for it in range(glb.Neq):

        U = thermostat.update(ptcls, it)
        if(params.Control.units == "Yukawa"):
            U *= 3

        K = 0
        species_start = 0
        for i in range(params.num_species):
            species_end = species_start + params.species[i].num
            K += 0.5*params.species[i].mass*np.ndarray.sum(ptcls.vel[species_start:species_end, :]**2)
            species_start = species_end

        Tp = (2/3)*K/float(N)/const.kb

        E = K + U

        if(it % glb.snap_int == 0 and glb.verbose):
            print("Equilibration: timestep, T, E, K, U = ", it, Tp, E, K, U)
#        print("1: pos = ", ptcls.pos)
#$        print("1: vel = ", ptcls.vel)

# saving the 0th step
checkpoint.dump(ptcls, 0)
time_stamp[its] = time.time(); its += 1

#print("\n------------- Production -------------")
# Opening files for writing particle positions, velcoities and forces
f_output_E = open("t_T_totalE_kinE_potE.out", "w")
f_xyz = open("p_v_a.xyz", "w")

if (params.load_method == "restart"):
    it_start = params.load_restart_step+0
else:
    it_start = 0


for it in range(it_start, Nt):
    U = integrator.update(ptcls)
    if(params.Control.units == "Yukawa"):
        U *= 3

    K = 0
    species_start = 0
    for i in range(params.num_species):
        species_end = species_start + params.species[i].num
        K += 0.5*params.species[i].mass*np.ndarray.sum(ptcls.vel[species_start:species_end, :]**2)
        species_start = species_end

    Tp = (2/3)*K/float(N)/const.kb
    E = K + U
    if(it % glb.snap_int == 0 and glb.verbose):
        print("Production: timestep, T, E, K, U = ", it, Tp, E, K, U)


    t_Tp_E_K_U = np.array([glb.dt*it, Tp, E, K, U])
    t_Tp_E_K_U2[:] = t_Tp_E_K_U

    # writing particle positions and velocities to file
    if((it+1) % params.Control.dump_step == 0):
        checkpoint.dump(ptcls, it+1)

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
        np.savetxt(f_xyz, np.c_[ptcls.pos, ptcls.vel, ptcls.acc])

# will be moved to observable class
np.save("n_qt", n_q_t)
time_stamp[its] = time.time(); its += 1

# closing output files
f_output_E.close()
f_xyz.close()

if(glb.verbose):
    verbose.time_stamp(time_stamp)

# end of the code
