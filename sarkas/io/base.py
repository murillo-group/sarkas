import os
import sys
import csv
import pickle
import numpy as np
from pyfiglet import print_figlet, Figlet

FONTS = ['speed',
         'starwars',
         'graffiti',
         'chunky',
         'epic',
         'larry3d',
         'ogre']

# FG_COLORS = ['255;255;255',
#              '13;177;75',
#              '153;162;162',
#              '240;133;33',
#              '144;154;183',
#              '209;222;63',
#              '232;217;181',
#              '200;154;88',
#              '148;174;74',
#              '203;90;40'
#              ]


FG_COLORS = ['24;69;49',
             '0;129;131',
             '83;80;84',
             '110;0;95'
             ]


def screen_figlet():
    """
    Print a colored figlet of Sarkas to screen.
    """
    fg = FG_COLORS[np.random.randint(0, len(FG_COLORS))]
    # bg = BG_COLORS[np.random.randint(0, len(BG_COLORS))]
    fnt = FONTS[np.random.randint(0, len(FONTS))]
    clr = fg  # + ':' + bg
    print_figlet('\nSarkas\n', font=fnt, colors=clr)

    print("\nAn open-source pure-python molecular dynamics code for non-ideal plasmas.")


class Verbose:
    """
    Class to handle verbose output to screen.

    Attributes
    ----------
    pre_run: bool
        Flag for pre run testing.

    verbose: bool
        Flag for verbose output to screen.

    io_file: str
        Filename of the file to write.

    f_log_name: str
        Log file name.

    f_pre_run: str
        Pre Run file name.

    Parameters
    ----------
    params : object
        Simulation's parameters

    """

    def __init__(self, params):
        self.verbose = params.verbose
        self.pre_run = params.pre_run
        # Create job folder if non existent
        if not os.path.exists(params.job_dir):
            os.mkdir(params.job_dir)

        pre_run_path = os.path.join(params.job_dir, 'Pre_Run_Test')
        if not os.path.exists(pre_run_path):
            os.mkdir(pre_run_path)
        params.pre_run_dir = pre_run_path
        # Pre run file name
        self.f_pre_run = os.path.join(params.pre_run_dir, 'pre_run_' + params.job_id + '.out')
        # Log File name
        self.f_log_name = os.path.join(params.job_dir, "log_" + params.job_id + ".out")
        # Save it in params too
        params.log_file = self.f_log_name

        # Pre run testing: assign self.io_file to the correct file to open/write
        self.io_file = self.f_pre_run if params.pre_run else self.f_log_name

        # Print figlet to file if not a restart run
        if not params.load_method == "prod_restart" and not params.load_method == "eq_restart":
            with open(self.io_file, "w+") as f_log:
                figlet_obj = Figlet(font='starwars')
                print(figlet_obj.renderText('Sarkas'), file=f_log)
                print("An open-source pure-Python molecular dynamics code for non-ideal plasmas.", file=f_log)

        # Print figlet to screen if verbose
        if self.verbose:
            screen_figlet()

    def sim_setting_summary(self, simulation):
        """
        Print out to file a summary of simulation's parameters.
        If verbose output then it will print twice: the first time to file and second time to screen.

        Parameters
        ----------
        simulation : cls
            Simulation's parameters

        """

        screen = sys.stdout
        f_log = open(self.io_file, 'a+')
        repeat = 2 if self.verbose else 1
        # redirect printing to file
        sys.stdout = f_log

        # Print to file first then to screen if repeat == 2
        while repeat > 0:

            if simulation.params.load_method == 'prod_restart':
                print('\n\n--------------------------- Restart -------------------------------------')
                self.time_info(simulation.params)
            elif simulation.params.load_method == 'eq_restart':
                print('\n\n------------------------ Therm Restart ----------------------------------')
                self.time_info(simulation.params)
            else:
                # Choose the correct heading
                if self.pre_run:
                    print('\n\n-------------- Pre Run Details ----------------------')
                else:
                    print('\n\n----------------- Simulation -----------------------')

                print('\nJob ID: ', simulation.params.job_id)
                print('Job directory: ', simulation.params.job_dir)
                print('Dump directory: ', simulation.params.prod_dump_dir)
                print('\nUnits: ', simulation.params.units)
                print('Total No. of particles = ', simulation.params.total_num_ptcls)

                print('\nParticle Species:')
                self.species_info(simulation)

                print('\nLengths scales:')
                self.length_info(simulation)

                print('\nBoundary conditions: {}'.format(simulation.params.boundary_conditions))

                print("\nThermostat: ", simulation.thermostat.type)
                self.thermostat_info(simulation)

                print('\nPotential: ', simulation.potential.type)
                self.potential_info(simulation)

                if simulation.params.magnetized:
                    print('\nMagnetized Plasma:')
                    for ic in range(simulation.params.num_species):
                        print('Cyclotron frequency of Species {:2} = {:2.6e}'.format(ic + 1,
                                                                                     simulation.species[ic].omega_c))
                        print('beta_c of Species {:2} = {:2.6e}'.format(ic + 1,
                                                                        simulation.species[ic].omega_c
                                                                        / simulation.species[ic].wp))
                print("\nAlgorithm: ", simulation.potential.method)
                self.algorithm_info(simulation)

                print("\nIntegrator: ", simulation.integrator.type)

                print("\nTime scales:")
                self.time_info(simulation)

            repeat -= 1
            sys.stdout = screen  # Restore the original sys.stdout

        f_log.close()

    def time_stamp(self, time_stamp, t):
        """
        Print out to screen elapsed times. If verbose output, print to file first and then to screen.

        Parameters
        ----------
        time_stamp : array
            Array of time stamps.

        t : float
            Elapsed time.
        """
        screen = sys.stdout
        f_log = open(self.io_file, 'a+')
        repeat = 2 if self.verbose else 1

        # redirect printing to file
        sys.stdout = f_log
        while repeat > 0:
            t_hrs = int(t / 3600)
            t_min = int((t - t_hrs * 3600) / 60)
            t_sec = int((t - t_hrs * 3600 - t_min * 60))
            print('\n{} Time = {} hrs {} mins {} secs'.format(time_stamp, t_hrs, t_min, t_sec))

            repeat -= 1
            sys.stdout = screen

        f_log.close()

    def timing_study(self, simulation):
        """
        Info specific for timing study.

        Parameters
        ----------
        params : object
            Simulation's parameters

        """
        screen = sys.stdout
        f_log = open(self.io_file, 'a+')
        repeat = 2 if self.verbose else 1

        # redirect printing to file
        sys.stdout = f_log

        # Print to file first then to screen if repeat == 2
        while repeat > 0:

            print('\n\n------------ Conclusion ------------\n')
            print('Suggested Mesh = [ {} , {} , {} ]'.format(*simulation.potential.pppm_mesh))
            print('Suggested Ewald parameter alpha = {:2.4f} / a_ws = {:1.6e} '.format(
                simulation.potential.pppm_alpha_ewald * simulation.params.aws,
                simulation.potential.pppm_alpha_ewald), end='')
            print("[1/cm]" if simulation.params.units == "cgs" else "[1/m]")
            print('Suggested rcut = {:2.4f} a_ws = {:2.6e} '.format(simulation.potential.rc / simulation.params.aws,
                                                                    simulation.potential.rc), end='')
            print("[cm]" if simulation.params.units == "cgs" else "[m]")

            self.algorithm_info(self, simulation)
            repeat -= 1
            sys.stdout = screen  # Restore the original sys.stdout

        f_log.close()

    @staticmethod
    def time_info(simulation):
        """
        Print time simulation's parameters.

        Parameters
        ----------
        params: object
            Simulation's parameters.

        """
        print('Time step = {:2.6e} [s]'.format(simulation.integrator.dt))
        if simulation.potential.type in ['Yukawa', 'EGS', 'Coulomb', 'Moliere']:
            print('(total) plasma frequency = {:1.6e} [Hz]'.format(simulation.params.total_plasma_frequency))
            print('wp dt = {:2.4f}'.format(simulation.integrator.dt * simulation.params.total_plasma_frequency))
        elif simulation.potential.type == 'QSP':
            print('e plasma frequency = {:2.6e} [Hz]'.format(simulation.species[0].wp))
            print('ion plasma frequency = {:2.6e} [Hz]'.format(simulation.species[1].wp))
            print('w_pe dt = {:2.4f}'.format(simulation.integrator.dt * simulation.species[0].wp))
        elif simulation.potential.type == 'LJ':
            print('(total) equivalent plasma frequency = {:1.6e} [Hz]'.format(simulation.params.total_plasma_frequency))
            print('wp dt = {:2.4f}'.format(simulation.integrator.dt * simulation.params.total_plasma_frequency))

        if simulation.params.load_method == 'prod_restart':
            print("Restart step: {}".format(simulation.params.load_restart_step))
            print('Total post-equilibration steps = {} ~ {} wp T_prod'.format(
                simulation.integrator.production_steps,
                int(simulation.integrator.production_steps * simulation.params.total_plasma_frequency * simulation.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                simulation.integrator.prod_dump_step,
                simulation.integrator.prod_dump_step * simulation.integrator.dt * simulation.params.total_plasma_frequency))
        elif simulation.params.load_method == 'eq_restart':
            print("Restart step: {}".format(simulation.params.load_therm_restart_step))
            print('Total equilibration steps = {} ~ {} wp T_prod'.format(
                simulation.integrator.equilibration_steps,
                int(simulation.integrator.eq_dump_step * simulation.params.total_plasma_frequency * simulation.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                simulation.integrator.eq_dump_step,
                simulation.integrator.eq_dump_step * simulation.integrator.dt * simulation.params.total_plasma_frequency))
        else:
            print('No. of equilibration steps = {} ~ {} wp T_eq'.format(
                simulation.integrator.equilibration_steps,
                int(simulation.integrator.equilibration_steps * simulation.params.total_plasma_frequency * simulation.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                simulation.integrator.eq_dump_step,
                simulation.integrator.eq_dump_step * simulation.integrator.dt * simulation.params.total_plasma_frequency))
            print('No. of post-equilibration steps = {} ~ {} wp T_prod'.format(
                simulation.integrator.production_steps,
                int(simulation.integrator.production_steps * simulation.params.total_plasma_frequency * simulation.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                simulation.integrator.prod_dump_step,
                simulation.integrator.prod_dump_step * simulation.integrator.dt * simulation.params.total_plasma_frequency))

    @staticmethod
    def algorithm_info(simulation):
        """
        Print algorithm information.

        Parameters
        ----------
        params: object
            Simulation's parameters.


        """
        if simulation.potential.method == 'P3M':
            print('Ewald parameter alpha = {:2.4f} / a_ws = {:1.6e} '.format(
                simulation.potential.pppm_alpha_ewald * simulation.params.aws,
                simulation.potential.pppm_alpha_ewald), end='')
            print("[1/cm]" if simulation.params.units == "cgs" else "[1/m]")
            print('Mesh size * Ewald_parameter (h * alpha) = {:2.4f}, {:2.4f}, {:2.4f} '.format(
                simulation.potential.pppm_h_array[0] * simulation.potential.pppm_alpha_ewald,
                simulation.potential.pppm_h_array[1] * simulation.potential.pppm_alpha_ewald,
                simulation.potential.pppm_h_array[2] * simulation.potential.pppm_alpha_ewald) )
            print('                                        ~ 1/{}, 1/{}, 1/{}'.format(
                int(1. / (simulation.potential.pppm_h_array[0] * simulation.potential.pppm_alpha_ewald)),
                int(1. / (simulation.potential.pppm_h_array[1] * simulation.potential.pppm_alpha_ewald)),
                int(1. / (simulation.potential.pppm_h_array[2] * simulation.potential.pppm_alpha_ewald)),
            ))
            print(
                'rcut = {:2.4f} a_ws = {:2.6e} '.format(simulation.potential.rc / simulation.params.aws,
                                                        simulation.potential.rc), end='')
            print("[cm]" if simulation.params.units == "cgs" else "[m]")
            print('Mesh = {} x {} x {}'.format(*simulation.potential.pppm_mesh))
            print('No. of PP cells per dimension = {:2}, {:2}, {:2}'.format(
                int(simulation.params.box_lengths[0] / simulation.potential.rc),
                int(simulation.params.box_lengths[1] / simulation.potential.rc),
                int(simulation.params.box_lengths[2] / simulation.potential.rc)))
            print('No. of particles in PP loop = {:6}'.format(
                int(simulation.params.total_num_density * (3 * simulation.potential.rc) ** 3)))
            print('No. of PP neighbors per particle = {:6}'.format(
                int(simulation.params.total_num_ptcls * 4.0 / 3.0 * np.pi * (
                        simulation.potential.rc / simulation.params.box_lengths.min()) ** 3.0)))
            print('PM Force Error = {:2.6e}'.format(simulation.params.pppm_pm_err))
            print('PP Force Error = {:2.6e}'.format(simulation.params.pppm_pp_err))

        elif simulation.potential.method == 'PP':
            print(
                'rcut = {:2.4f} a_ws = {:2.6e} '.format(simulation.potential.rc / simulation.params.aws,
                                                        simulation.potential.rc),
                end='')
            print("[cm]" if simulation.params.units == "cgs" else "[m]")
            print('No. of PP cells per dimension = {:2}, {:2}, {:2}'.format(
                int(simulation.params.box_lengths[0] / simulation.potential.rc),
                int(simulation.params.box_lengths[1] / simulation.potential.rc),
                int(simulation.params.box_lengths[2] / simulation.potential.rc)))
            print('No. of particles in PP loop = {:6}'.format(
                int(simulation.params.total_num_density * (3 * simulation.potential.rc) ** 3)))
            print('No. of PP neighbors per particle = {:6}'.format(
                int(simulation.params.total_num_ptcls * 4.0 / 3.0 * np.pi * (
                        simulation.potential.rc / simulation.params.box_lengths.min()) ** 3.0)))

        print('Tot Force Error = {:2.6e}'.format(simulation.params.force_error))

    @staticmethod
    def potential_info(simulation):
        """
        Print potential information.

        Parameters
        ----------
        params: object
            Simulation's parameters.

        """
        if simulation.potential.type == 'Yukawa':
            print('kappa = {:1.4e}'.format(simulation.params.aws / simulation.params.lambda_TF))
            print('lambda_TF = {:1.4e}'.format(simulation.params.lambda_TF))
            print('Gamma_eff = {:4.2f}'.format(simulation.params.coupling_constant))

        elif simulation.potential.type == 'EGS':
            print('kappa = {:1.4e}'.format(simulation.params.aws / simulation.params.lambda_TF))
            print('lambda_TF = {:1.4e}'.format(simulation.params.lambda_TF))
            print('nu = {:1.4e}'.format(simulation.params.nu))
            if simulation.params.nu < 1:
                print('Exponential decay:')
                print('lambda_p = {:1.4e}'.format(simulation.params.lambda_p))
                print('lambda_m = {:1.4e}'.format(simulation.params.lambda_m))
                print('alpha = {:1.4e}'.format(simulation.params.alpha))
                print('Theta = {:1.4e}'.format(simulation.params.electron_degeneracy_parameter))
                print('b = {:1.4e}'.format(simulation.params.b))

            else:
                print('Oscillatory potential:')
                print('gamma_p = {:1.4e}'.format(simulation.params.gamma_p))
                print('gamma_m = {:1.4e}'.format(simulation.params.gamma_m))
                print('alpha = {:1.4e}'.format(simulation.params.alphap))
                print('Theta = {:1.4e}'.format(simulation.params.theta))
                print('b = {:1.4e}'.format(simulation.params.b))

            print('Gamma_eff = {:4.2f}'.format(simulation.params.coupling_constant))

        elif simulation.potential.type == 'Coulomb':
            print('Gamma_eff = {:4.2f}'.format(simulation.params.coupling_constant))

        elif simulation.potential.type == 'LJ':
            print('epsilon = {:2.6e}'.format(simulation.potential.matrix[0, 0, 0]))
            print('sigma = {:2.6e}'.format(simulation.potential.matrix[1, 0, 0]))
            print('Gamma_eff = {:4.2f}'.format(simulation.params.coupling_constant))

        elif simulation.potential.type == "QSP":
            print("e de Broglie wavelength = {:2.4f} ai = {:2.6e} ".format(
                2.0 * np.pi / simulation.potential.matrix[1, 0, 0] / (np.sqrt(2.0) * simulation.params.ai),
                2.0 * np.pi / simulation.potential.matrix[1, 0, 0] / np.sqrt(2.0)), end='')
            print("[cm]" if simulation.params.units == "cgs" else "[m]")
            print("e-e screening parameter = {:2.4f}".format(
                simulation.potential.matrix[1, 0, 0] * simulation.params.aws))
            print("ion de Broglie wavelength  = {:2.4f} ai = {:2.6e} ".format(
                2.0 * np.pi / simulation.potential.matrix[1, 0, 0] / (np.sqrt(2.0) * simulation.params.ai),
                2.0 * np.pi / simulation.potential.matrix[1, 0, 0] / np.sqrt(2.0)), end='')
            print("[cm]" if simulation.params.units == "cgs" else "[m]")
            print("i-i screening parameter = {:2.4f}".format(
                simulation.potential.matrix[1, 1, 1] * simulation.params.aws))
            print("e-i screening parameter = {:2.4f}".format(
                simulation.potential.matrix[1, 0, 1] * simulation.params.aws))
            print("e-i Coupling Parameter = {:3.3f} ".format(simulation.params.coupling_constant))
            print("rs Coupling Parameter = {:3.3f} ".format(simulation.params.rs))

    @staticmethod
    def thermostat_info(simulation):
        """
        Print thermostat information.

        Parameters
        ----------
        params: object
            Simulation's parameters.

        """
        print("Berendsen Relaxation rate: {:1.3f}".format(simulation.thermostat.relaxation_rate))
        print("Thermostating Temperatures: ", simulation.thermostat.temperatures)

    @staticmethod
    def length_info(simulation):
        """
        Print length information.

        Parameters
        ----------
        params: object
            Simulation's parameters.

        """
        print('Wigner-Seitz radius = {:2.6e} '.format(simulation.params.aws), end='')
        print("[cm]" if simulation.params.units == "cgs" else "[m]")
        print('No. of non-zero box dimensions = ', int(simulation.params.dimensions))
        print('Box length along x axis = {:2.6e} a_ws = {:2.6e} '.format(
            simulation.params.box_lengths[0] / simulation.params.aws, simulation.params.box_lengths[0]), end='')
        print("[cm]" if simulation.params.units == "cgs" else "[m]")

        print('Box length along y axis = {:2.6e} a_ws = {:2.6e} '.format(
            simulation.params.box_lengths[1] / simulation.params.aws, simulation.params.box_lengths[1]), end='')
        print("[cm]" if simulation.params.units == "cgs" else "[m]")

        print('Box length along z axis = {:2.6e} a_ws = {:2.6e} '.format(
            simulation.params.box_lengths[2] / simulation.params.aws, simulation.params.box_lengths[2]), end='')
        print("[cm]" if simulation.params.units == "cgs" else "[m]")

        print("The remaining lengths scales are given in ", end='')
        print("[cm]" if simulation.params.units == "cgs" else "[m]")

    @staticmethod
    def species_info(simulation):
        print('No. of species = ', len(simulation.species))
        for isp, sp in enumerate(simulation.species):
            print("Species {} : {}".format(isp + 1, sp.name))
            print("\tSpecies ID: {}".format(isp))
            print("\tNo. of particles = {} ".format(sp.num))
            print("\tNumber density = {:2.6e} ".format(sp.number_density), end='')
            print("[N/cc]" if simulation.params.units == "cgs" else "[N/m^3]")
            print("\tMass = {:2.6e} ".format(sp.mass), end='')
            print("[g]" if simulation.params.units == "cgs" else "[kg]")
            print('\tTemperature = {:2.6e} [K]'.format(sp.temperature))


class Checkpoint:
    """
    Class to handle restart dumps.

    Parameters
    ----------
    params: object
        Simulation's parameters.

    Attributes
    ----------
    dt : float
        Simulation timestep.

    dump_dir : str
        Path to directory where simulations dump will be stored.

    energy_filename : str
        CSV file for storing energy values.

    ptcls_file_name : str
        Prefix of dumps filenames.

    params_pickle : str
        Pickle file where all simulation parameters will be stored.

    species_names: list
        Names of each particle species.

    job_dir : str
        Output directory.
    """

    def __init__(self, params, species):
        self.dt = params.dt
        self.kB = params.kB
        self.job_dir = params.job_dir
        self.params_pickle = os.path.join(self.job_dir, "simulation_parameters.pickle")
        # Production directory and filenames
        self.production_dir = params.production_dir
        self.prod_dump_dir = params.prod_dump_dir
        self.prod_energy_filename = os.path.join(self.production_dir,
                                                 "ProductionEnergy_" + params.job_id + '.csv')
        self.prod_ptcls_file_name = os.path.join(self.prod_dump_dir, "checkpoint_")
        # Thermalization directory and filenames
        self.equilibration_dir = params.equilibration_dir
        self.eq_dump_dir = params.eq_dump_dir
        self.eq_energy_filename = os.path.join(self.equilibration_dir,
                                               "EquilibrationEnergy_" + params.job_id + '.csv')
        self.eq_ptcls_file_name = os.path.join(self.eq_dump_dir, "checkpoint_")

        self.species_names = params.species_names
        self.coupling = params.coupling_constant * params.T_desired

        if not os.path.exists(self.prod_energy_filename):
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(species) > 1:
                for i, sp in enumerate(species):
                    dkeys.append("{} Kinetic Energy".format(sp.name))
                    dkeys.append("{} Temperature".format(sp.name))
            dkeys.append("Gamma")
            data = dict.fromkeys(dkeys)

            with open(self.prod_energy_filename, 'w+') as f:
                w = csv.writer(f)
                w.writerow(data.keys())

        if not os.path.exists(self.eq_energy_filename) and not params.load_method[-7:] == 'restart':
            # Create the Energy file
            dkeys = ["Time", "Total Energy", "Total Kinetic Energy", "Potential Energy", "Temperature"]
            if len(species) > 1:
                for i, sp_name in enumerate(params.species_names):
                    dkeys.append("{} Kinetic Energy".format(sp_name))
                    dkeys.append("{} Temperature".format(sp_name))
            dkeys.append("Gamma")
            data = dict.fromkeys(dkeys)

            with open(self.eq_energy_filename, 'w+') as f:
                w = csv.writer(f)
                w.writerow(data.keys())

    def save_pickle(self, params):
        """
        Save all simulations parameters in a pickle file.

        Parameters
        ----------
        params: object
            Simulation's parameters.

        """
        pickle_file = open(self.params_pickle, "wb")
        pickle.dump(params, pickle_file)
        pickle_file.close()

    def dump(self, production, ptcls, potential_energy, it):
        """
        Save particles' data to binary file for future restart.

        Parameters
        ----------
        production: bool
            Flag indicating whether to phase production or equilibration data.

        ptcls: object
            Particles data.

        potential_energy : float
            Potential energy.

        it : int
            Timestep number.
        """
        if production:
            ptcls_file = self.prod_ptcls_file_name + str(it)
            tme = it * self.dt
            np.savez(ptcls_file,
                  id=ptcls.id,
                  names=ptcls.names,
                  pos=ptcls.pos,
                  vel=ptcls.vel,
                  acc=ptcls.acc,
                  cntr=ptcls.pbc_cntr,
                  rdf_hist=ptcls.rdf_hist,
                  time=tme)

            energy_file = self.prod_energy_filename

        else:
            ptcls_file = self.prod_ptcls_file_name + str(it)
            tme = it * self.dt
            np.savez(ptcls_file,
                  id=ptcls.id,
                  name=ptcls.names,
                  pos=ptcls.pos,
                  vel=ptcls.vel,
                  acc=ptcls.acc,
                  time=tme)

            energy_file = self.prod_energy_filename

        kinetic_energies, temperatures = ptcls.kinetic_temperature(self.kB)
        # Prepare data for saving
        data = {"Time": it * self.dt,
                "Total Energy": np.sum(kinetic_energies) + potential_energy,
                "Total Kinetic Energy": np.sum(kinetic_energies),
                "Potential Energy": potential_energy,
                "Total Temperature": np.sum(temperatures)
                }
        for sp, kin in enumerate(kinetic_energies):
            data["{} Kinetic Energy".format(self.species_names[sp])] = kin
            data["{} Temperature".format(self.species_names[sp])] = temperatures[sp]
        with open(energy_file, 'a') as f:
            w = csv.writer(f)
            w.writerow(data.values())
