"""
                       Sarkas

A code that executes molecular dynamics simulation for a Yukawa plasma using
the efficient Particle-Particle-Particle-Mesh algorithm for force computation.
The code constitutes a number of functions that are in separate files.

Developed by the research group of:
Professor Michael S. Murillo
murillom@msu.edu
Dept. of Computational Mathematics, Science, and Engineering,
Michigan State University
"""

# Python modules
import numpy as np
import time
import sys
from tqdm import tqdm

# import MD modules, class
from S_thermostat import Thermostat, calc_kin_temp, remove_drift
from S_integrator import Integrator, calc_pot_acc
from S_particles import Particles
from S_verbose import Verbose
from S_params import Params
from S_checkpoint import Checkpoint
from S_postprocessing import Thermodynamics, RadialDistributionFunction

time0 = time.time()

input_file = sys.argv[1]
params = Params()
params.setup(input_file)  # Read initial conditions and setup parameters

if params.Control.verbose:
    print('\nSarkas Ver. 1.0')
    print('Input file read.')
    print('Params Class created.')

integrator = Integrator(params)
checkpoint = Checkpoint(params)  # For restart and pva backups.
thermostat = Thermostat(params)
verbose = Verbose(params)
#######################
Nt = params.Control.Nsteps  # number of time steps
N = params.total_num_ptcls

########################
verbose.sim_setting_summary()  # simulation setting summary
if params.Control.verbose:
    print('\nLog file created.')

ptcls = Particles(params)
ptcls.load(params)

if params.Control.verbose:
    print('\nParticles initialized.')

# Calculate initial forces and potential energy
U_init = calc_pot_acc(ptcls, params)
# Calculate initial kinetic energy and temperature
Ks_init, Tps_init = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
K_init = np.ndarray.sum(Ks_init)
Tp_init = np.ndarray.sum(Tps_init) / params.num_species
E_init = K_init + U_init
remove_drift(ptcls.vel, ptcls.species_num, ptcls.species_mass)
#
time_init = time.time()
verbose.time_stamp("Initialization", time_init - time0)
#
f_log = open(params.Control.log_file, 'a+')
print("\nInitial: T = {:2.6e}, E = {:2.6e}, K = {:2.6e}, U = {:2.6e}".format(Tp_init, E_init, K_init, U_init), file=f_log)
f_log.close()

if not (params.load_method == "restart"):
    if params.Control.verbose:
        print("\n------------- Equilibration -------------")
    for it in tqdm(range(params.Control.Neq), disable=not(params.Control.verbose)):
        # Calculate the Potential energy and update particles' data
        U_therm = integrator.update(ptcls, params)
        # Thermostate
        thermostat.update(ptcls.vel, it)

    remove_drift(ptcls.vel, ptcls.species_num, ptcls.species_mass)

checkpoint.dump(ptcls, 0)
time_eq = time.time()
verbose.time_stamp("Equilibration", time_eq - time_init)
# Turn on magnetic field, if not on already, and thermalize
if params.Magnetic.on == 1 and params.Magnetic.elec_therm == 1:
    params.Integrator.type = params.Integrator.mag_type
    integrator = Integrator(params)

    if params.Control.verbose:
        print("\n------------- Magnetic Equilibration -------------")

    for it in tqdm(range(params.Magnetic.Neq_mag), disable=(not params.Control.verbose)):
        # Calculate the Potential energy and update particles' data
        U_therm = integrator.update(ptcls, params)
        # Thermostate
        thermostat.update(ptcls.vel, it)

    remove_drift(ptcls.vel, ptcls.species_num, ptcls.species_mass)
    # saving the 0th step
    checkpoint.dump(ptcls, 0)
    time_mag = time.time()
    verbose.time_stamp("Magnetic Equilibration", time_mag - time_eq)
    time_eq = time_mag

# Open output files
if params.load_method == "restart":
    it_start = params.load_restart_step
    if params.Control.writexyz:
        # Particles' positions, velocities, accelerations for OVITO
        f_xyz = open(params.Control.checkpoint_dir + "/" + "pva_" + params.Control.fname_app + ".xyz", "a+")

else:
    it_start = 0
    if params.Control.writexyz:
        # Particles' positions, velocities, accelerations for OVITO
        f_xyz = open(params.Control.checkpoint_dir + "/" + "pva_" + params.Control.fname_app + ".xyz", "w+")

if params.Control.verbose:
    print("\n------------- Production -------------")

pscale = 1.0 / params.aws
vscale = 1.0 / (params.aws * params.wp)
ascale = 1.0 / (params.aws * params.wp ** 2)

# Update measurement flag for rdf
params.Control.measure = True
# Restart the pbc counter
ptcls.pbc_cntr.fill(0.0)

Ndumps = int(params.Control.Nsteps/params.Control.dump_step) + 1
K = np.zeros(Ndumps)
U = np.zeros(Ndumps)
E = np.zeros(Ndumps)
Ks = np.zeros((Ndumps, params.num_species))
Tps = np.zeros((Ndumps, params.num_species))
Time = np.zeros(Ndumps)
# Save the initial Thermodynamics
Ks[0, :], Tps[0, :] = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
K[0] = np.ndarray.sum(Ks[0, :])
U[0] = U_therm
E[0] = K[0] + U[0]
Time[0] = 0.0
dump_indx = 0
for it in tqdm(range(it_start, Nt), disable=(not params.Control.verbose)):
    # Move the particles and calculate the potential
    U_prod = integrator.update(ptcls, params)
    if (it + 1) % params.Control.dump_step == 0:
        # Save particles' data for restart
        checkpoint.dump(ptcls, it + 1)
        dump_indx += 1
        # Calculate Kinetic Energy and Temperature
        Ks[dump_indx, :], Tps[dump_indx, :] = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
        K[dump_indx] = np.ndarray.sum(Ks[dump_indx, :])
        U[dump_indx] = U_prod
        E[dump_indx] = K[dump_indx] + U_prod
        Time[dump_indx] = it * params.Control.dt
        # Write particles' data to XYZ file for OVITO Visualization
        if params.Control.writexyz:
            f_xyz.writelines("{0:d}\n".format(N))
            f_xyz.writelines("name x y z vx vy vz ax ay az\n")
            np.savetxt(f_xyz,
                       np.c_[ptcls.species_name, ptcls.pos * pscale, ptcls.vel * vscale, ptcls.acc * ascale],
                       fmt="%s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")
time_prod = time.time()
verbose.time_stamp("Production", time_prod - time_eq)
if params.Control.writexyz:
    f_xyz.close()

# Make a dictionary for PostProcessing
data = {"Time": Time, "Total Energy": E, "Kinetic Energy": K, "Potential Energy": U}
if params.num_species > 1:
    Tot_Temp = np.zeros(Ndumps)
    for sp in range(params.num_species):
        Tot_Temp[:] += params.species[sp].concentration*Tps[:, sp]
        data["{} Temperature".format(params.species[sp].name)] = Tps[:, sp]
        data["{} Kinetic Energy".format(params.species[sp].name)] = Ks[:, sp]
    data["Temperature"] = Tot_Temp
    data["Gamma"] = params.Potential.Gamma_eff * params.T_desired / Tot_Temp
else:
    data["Temperature"] = Tps[:, 0]
    data["Gamma"] = params.Potential.Gamma_eff * params.T_desired / Tps[:, 0]

Boltzmann = Thermodynamics(params)
Boltzmann.save(data)
Boltzmann.plot('Total Energy', True)
Boltzmann.plot('Temperature', True)

rdf = RadialDistributionFunction(params)
rdf.save(ptcls.rdf_hist)
rdf.plot()

# Print elapsed times to screen
time_tot = time.time()
verbose.time_stamp("Total", time_tot - time0)
# end of the code
