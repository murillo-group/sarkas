"""
Module for running the simulation.
"""

# Python modules
import numpy as np
import time
# Progress bar
from tqdm import tqdm

# Sarkas MD modules
# from sarkas.simulation.params import Params
from sarkas.simulation.particles import Particles
from sarkas.thermostats.thermostat import Thermostat, calc_kin_temp, remove_drift
from sarkas.integrators.integrator import Integrator, calc_pot_acc#, #calc_pot_acc_fmm
from sarkas.io.verbose import Verbose
from sarkas.io.checkpoint import Checkpoint
from sarkas.tools.postprocessing import RadialDistributionFunction


def run(params):
    """
    Run a Molecular Dynamics simulation with the parameters given by the input YAML file.

    Parameters
    ----------
    params: object
        Simulation's parameters.

    """
    time0 = time.time()
    # For restart and pva backups.
    checkpoint = Checkpoint(params)
    checkpoint.save_pickle(params)

    verbose = Verbose(params)
    verbose.sim_setting_summary(params)  # simulation setting summary

    integrator = Integrator(params)
    thermostat = Thermostat(params)

    ptcls = Particles(params)
    ptcls.load(params)

    if params.Control.verbose:
        print('\nParticles initialized.')

    # Calculate initial kinetic energy and temperature
    if not params.Potential.method == "FMM":
        U_init = calc_pot_acc(ptcls, params)
    # else:
        # U_init = calc_pot_acc_fmm(ptcls, params)

    Ks, Tps = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
    Tot_Kin = Ks.sum()
    Temperature = ptcls.species_conc.transpose() @ Tps
    E_init = U_init + Tot_Kin
    #
    time_init = time.time()
    verbose.time_stamp("Initialization", time_init - time0)
    #
    f_log = open(params.Control.log_file, 'a+')
    print("\nInitial State:", file=f_log)
    if params.Control.units == "cgs":
        print("T = {:2.6e} [K], E = {:2.6e} [erg], K = {:2.6e} [erg], U = {:2.6e} [erg]".format(Temperature,
                                                                                                E_init, Tot_Kin,
                                                                                                U_init), file=f_log)
    else:
        print("T = {:2.6e} [K], E = {:2.6e} [J], K = {:2.6e} [J], U = {:2.6e} [J]".format(Temperature,
                                                                                          E_init, Tot_Kin, U_init),
              file=f_log)
    f_log.close()
    ##############################################
    # Equilibration Phase
    ##############################################
    if not params.load_method == "restart":
        if params.load_method == "therm_restart":
            it_start = params.load_therm_restart_step
        else:
            it_start = 0
        checkpoint.therm_dump(ptcls, Ks, Tps, U_init, it_start)
        if params.Control.verbose:
            print("\n------------- Equilibration -------------")
        for it in tqdm(range(it_start, params.Control.Neq), disable=not params.Control.verbose):
            # Calculate the Potential energy and update particles' data
            U_therm = integrator.update(ptcls, params)
            if (it + 1) % params.Control.therm_dump_step == 0:
                Ks, Tps = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
                checkpoint.therm_dump(ptcls, Ks, Tps, U_therm, it + 1)
            # Thermostate
            thermostat.update(ptcls.vel, it)
        remove_drift(ptcls.vel, ptcls.species_num, ptcls.species_mass)
        # Save the current state
        Ks, Tps = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
        U_therm = U_init
        checkpoint.dump(ptcls, Ks, Tps, U_therm, 0)

    time_eq = time.time()
    verbose.time_stamp("Equilibration", time_eq - time_init)
    ##############################################
    # Magnetic Equilibration Phase
    ##############################################
    # Turn on magnetic field, if not on already, and thermalize
    if params.Magnetic.on and params.Magnetic.elec_therm:
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
        # Save the current state
        Ks, Tps = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
        checkpoint.dump(ptcls, Ks, Tps, U_therm, 0)
        #
        time_mag = time.time()
        verbose.time_stamp("Magnetic Equilibration", time_mag - time_eq)
        time_eq = time_mag

    ##############################################
    # Prepare for Production Phase
    ##############################################

    # Open output files
    if params.load_method == "restart":
        it_start = params.load_restart_step
        if params.Control.writexyz:
            # Particles' positions, velocities, accelerations for OVITO
            f_xyz = open(params.Control.checkpoint_dir + "/" + "pva_" + params.Control.fname_app + ".xyz", "a+")
    else:
        it_start = 0
        # Restart the pbc counter

        ptcls.pbc_cntr.fill(0.0)
        # Create array for storing energy information
        if params.Control.writexyz:
            # Particles' positions, velocities, accelerations for OVITO
            f_xyz = open(params.Control.checkpoint_dir + "/" + "pva_" + params.Control.fname_app + ".xyz", "w+")

    pscale = 1.0 / params.aws
    vscale = 1.0 / (params.aws * params.wp)
    ascale = 1.0 / (params.aws * params.wp ** 2)

    # Update measurement flag for rdf
    params.Control.measure = True

    ##############################################
    # Production Phase
    ##############################################
    if params.Control.verbose:
        print("\n------------- Production -------------")
    time_eq = time.time()
    for it in tqdm(range(it_start, params.Control.Nsteps), disable=(not params.Control.verbose)):
        # Move the particles and calculate the potential
        U_prod = integrator.update(ptcls, params)
        if (it + 1) % params.Control.dump_step == 0:
            # Save particles' data for restart
            Ks, Tps = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
            checkpoint.dump(ptcls, Ks, Tps, U_prod, it + 1)
            # Write particles' data to XYZ file for OVITO Visualization
            if params.Control.writexyz:
                f_xyz.writelines("{0:d}\n".format(params.total_num_ptcls))
                f_xyz.writelines("name x y z vx vy vz ax ay az\n")
                np.savetxt(f_xyz,
                           np.c_[ptcls.species_name, ptcls.pos * pscale, ptcls.vel * vscale, ptcls.acc * ascale],
                           fmt="%s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")
    # Save production time
    time_prod = time.time()
    verbose.time_stamp("Production", time_prod - time_eq)
    if params.Control.writexyz:
        f_xyz.close()
    ##############################################
    # Finalization Phase
    ##############################################
    rdf = RadialDistributionFunction(params)
    rdf.save(ptcls.rdf_hist)
    rdf.plot(show=False)

    # Print elapsed times to screen
    time_tot = time.time()
    verbose.time_stamp("Total", time_tot - time0)
    ##############################################
    # End of simulation
    ##############################################


if __name__ == '__main__':
    from optparse import OptionParser
    from sarkas.simulation.params import Params
    # Construct the option parser
    op = OptionParser()

    # Add the arguments to the parser
    op.add_option("-t", "--pre_run_testing", action='store_true', dest='test', default=False,
                  help="Test input parameters")
    op.add_option("-v", "--verbose", action='store_true', dest='verbose', default=False, help="Verbose output")
    op.add_option("-p", "--plot_show", action='store_true', dest='plot_show', default=False, help="Show plots")
    op.add_option("-c", "--check", type='choice', choices=['therm', 'prod'],
                  action='store', dest='check', help="Check current state of run")
    op.add_option("-d", "--job_dir", action='store', dest='job_dir', help="Job Directory")
    op.add_option("-j", "--job_id", action='store', dest='job_id', help="Job ID")
    op.add_option("-s", "--seed", action='store', dest='seed', type='int', help="Random Number Seed")
    op.add_option("-i", "--input", action='store', dest='input_file', help="YAML Input file")
    op.add_option("-r", "--restart", action='store', dest='restart', type=int, help="Restart step")

    options, _ = op.parse_args()

    params = Params()
    params.setup(vars(options))
    # Read initial conditions and setup parameters
    # Update rand seed with option. This supersedes the input file.
    if options.seed:
        params.load_rand_seed = int(options.seed)

    # Verbose output. This does not supersede the input file if False.
    # That is if you don't give this option and the input file has Control.verbose=Yes, then you will
    # still have a verbose output
    if options.verbose:
        params.Control.verbose = True

    if options.restart is not None:
        params.load_method = 'restart'
        params.load_restart_step = options.restart
        
    run(params)

