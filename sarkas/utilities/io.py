import os
import sys
import re
import yaml
import csv
import pickle
import numpy as np
from pyfiglet import print_figlet, Figlet
from tqdm import tqdm

FONTS = ['speed',
         'starwars',
         'graffiti',
         'chunky',
         'epic',
         'larry3d',
         'ogre']

FG_COLORS = ['255;255;255',
             '13;177;75',
             '153;162;162',
             '240;133;33',
             '144;154;183',
             '209;222;63',
             '232;217;181',
             '200;154;88',
             '148;174;74',
             '203;90;40'
             ]

BG_COLORS = ['24;69;49',
             '0;129;131',
             '83;80;84',
             '110;0;95'
             ]


class InputOutput:

    def __init__(self):
        """Set default directory names."""
        self.input_file = None
        self.simulations_dir = "Simulations"
        self.production_dir = 'Production'
        self.equilibration_dir = 'Equilibration'
        self.preprocessing_dir = "PreProcessing"
        self.postprocessing_dir = "PostProcessing"
        self.prod_dump_dir = 'dumps'
        self.eq_dump_dir = 'dumps'
        self.job_dir = None
        self.job_id = None
        self.log_file = None
        self.preprocess_file = None
        self.preprocessing = False
        self.verbose = False
        self.check_status = False
        self.xyz_dir = None
        self.xyz_filename = None

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = 'InputOuput( \n'
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ')'
        return disp

    def from_dict(self, input_dict: dict):
        """
        Update attributes from input dictionary.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """
        self.__dict__.update(input_dict)

    def setup(self):
        self.create_file_paths()
        self.make_directories()
        self.file_header()

    def from_yaml(self, filename):
        """
        Parse inputs from YAML file.

        Parameters
        ----------
        filename: str
            Input YAML file.

        Returns
        -------
        dics : dict
            Content of YAML file parsed in a nested dictionary
        """

        self.input_file = filename
        with open(filename, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)
            self.__dict__.update(dics["IO"])

        if 'Control' in dics.keys():
            for key, value in dics['Control'].items():

                if key == 'verbose':
                    self.verbose = value

                if key == 'load_method':
                    self.load_method = value
                    if value[-7:] == 'restart':
                        self.restart = True
                    else:
                        self.restart = False

                if key == 'preprocessing':
                    self.preprocessing = value

        return dics

    def create_file_paths(self):

        if self.job_dir is None:
            self.job_dir = os.path.basename(self.input_file).split('.')[0]

        if self.job_id is None:
            self.job_id = self.job_dir

        self.job_dir = os.path.join(self.simulations_dir, self.job_dir)

        # Equilibration directory and sub_dir
        self.equilibration_dir = os.path.join(self.job_dir, self.equilibration_dir)
        self.eq_dump_dir = os.path.join(self.equilibration_dir, 'dumps')
        # Production dir and sub_dir
        self.production_dir = os.path.join(self.job_dir, self.production_dir)
        self.prod_dump_dir = os.path.join(self.production_dir, "dumps")

        # Preprocessing dir
        self.preprocessing_dir = os.path.join(self.job_dir, self.preprocessing_dir)

        # Postprocessing dir
        self.postprocessing_dir = os.path.join(self.job_dir, self.postprocessing_dir)

        if self.log_file is None:
            self.log_file = os.path.join(self.job_dir, "log_" + self.job_id + ".out")

        # Pre run file name
        self.preprocess_file = os.path.join(self.preprocessing_dir, 'PreProcessing_' + self.job_id + '.out')

        # Production phase filenames
        self.prod_energy_filename = os.path.join(self.production_dir, "ProductionEnergy_" + self.job_id + '.csv')
        self.prod_ptcls_filename = os.path.join(self.prod_dump_dir, "checkpoint_")
        # Equilibration phase filenames
        self.eq_energy_filename = os.path.join(self.equilibration_dir, "EquilibrationEnergy_" + self.job_id + '.csv')
        self.eq_ptcls_filename = os.path.join(self.eq_dump_dir, "checkpoint_")

        if self.preprocessing:
            self.io_file = self.preprocess_file
        else:
            self.io_file = self.log_file

    def make_directories(self):

        # Check if the directories exist
        if not os.path.exists(self.simulations_dir):
            os.mkdir(self.simulations_dir)

        if not os.path.exists(self.job_dir):
            os.mkdir(self.job_dir)

        if not os.path.exists(self.equilibration_dir):
            os.mkdir(self.equilibration_dir)

        if not os.path.exists(self.eq_dump_dir):
            os.mkdir(self.eq_dump_dir)

        if not os.path.exists(self.production_dir):
            os.mkdir(self.production_dir)

        if not os.path.exists(self.prod_dump_dir):
            os.mkdir(self.prod_dump_dir)

        if self.preprocessing:
            if not os.path.exists(self.preprocessing_dir):
                os.mkdir(self.preprocessing_dir)

        if not os.path.exists(self.postprocessing_dir):
            os.mkdir(self.postprocessing_dir)

    def file_header(self):

        # Print figlet to file if not a restart run
        if not self.restart or not self.check_status:
            with open(self.io_file, "w+") as f_log:
                figlet_obj = Figlet(font='starwars')
                print(figlet_obj.renderText('Sarkas'), file=f_log)
                print("An open-source pure-Python molecular dynamics code for non-ideal plasmas.", file=f_log)

        # Print figlet to screen if verbose
        if self.verbose:
            self.screen_figlet()

    def simulation_summary(self, simulation):
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

            if simulation.parameters.load_method == 'prod_restart':
                print('\n\n--------------------------- Restart -------------------------------------')
                self.time_info(simulation)
            elif simulation.parameters.load_method == 'eq_restart':
                print('\n\n------------------------ Therm Restart ----------------------------------')
                self.time_info(simulation)
            else:
                # Choose the correct heading
                if self.preprocessing:
                    print('\n\n-------------- Pre Processing ----------------------')
                else:
                    print('\n\n----------------- Simulation -----------------------')

                print('\nJob ID: ', self.job_id)
                print('Job directory: ', self.job_dir)
                print('Equilibration dumps directory: ', self.eq_dump_dir)
                print('Production dumps directory: ', self.prod_dump_dir)
                print('\nUnits: ', simulation.parameters.units)
                print('Total No. of particles = ', simulation.parameters.total_num_ptcls)

                print('\nParticle Species:')
                self.species_info(simulation)

                print('\nLengths scales:')
                self.length_info(simulation)

                print('\nBoundary conditions: {}'.format(simulation.parameters.boundary_conditions))

                print("\nThermostat: ", simulation.thermostat.type)
                self.thermostat_info(simulation)

                print('\nPotential: ', simulation.potential.type)
                self.potential_info(simulation)

                if simulation.parameters.magnetized:
                    print('\nMagnetized Plasma:')
                    for ic in range(simulation.parameters.num_species):
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
        t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = t
        # redirect printing to file
        sys.stdout = f_log
        while repeat > 0:
            if t_hrs == 0 and t_min == 0 and t_sec <= 2:
                print('\n{} Time: {} sec {} msec {} usec {} nsec'.format(time_stamp,
                                                                         int(t_sec),
                                                                         int(t_msec),
                                                                         int(t_usec),
                                                                         int(t_nsec)))
            else:
                print('\n{} Time: {} hrs {} min {} sec'.format(time_stamp, int(t_hrs), int(t_min), int(t_sec)))

            repeat -= 1
            sys.stdout = screen

        f_log.close()

    def timing_study(self, simulation):
        """
        Info specific for timing study.

        Parameters
        ----------
        simulation: sarkas.base.Simulation
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
                simulation.potential.pppm_alpha_ewald * simulation.parameters.a_ws,
                simulation.potential.pppm_alpha_ewald), end='')
            print("[1/cm]" if simulation.parameters.units == "cgs" else "[1/m]")
            print('Suggested rcut = {:2.4f} a_ws = {:2.6e} '.format(simulation.potential.rc / simulation.parameters.a_ws,
                                                                    simulation.potential.rc), end='')
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")

            self.algorithm_info(simulation)
            repeat -= 1
            sys.stdout = screen  # Restore the original sys.stdout

        f_log.close()

    def preprocess_timing(self, str_id, t, loops):
        """Print times estimates of simulation."""
        t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = t
        screen = sys.stdout
        f_log = open(self.io_file, 'a+')
        repeat = 2 if self.verbose else 1
        t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = t
        # redirect printing to file
        sys.stdout = f_log
        while repeat > 0:
            if str_id == "GF":
                print('\n\n\n----------------- Force Calculation Times ----------------------\n')
                print("Optimal Green's Function Time: \n"
                      '{} min {} sec {} msec {} usec {} nsec \n'.format(loops,
                                                                        int(t_min),
                                                                        int(t_sec),
                                                                        int(t_msec),
                                                                        int(t_usec),
                                                                        int(t_nsec)))

            elif str_id == "PP":
                print('Time of PP acceleration calculation averaged over {} loops: \n'
                      '{} min {} sec {} msec {} usec {} nsec \n'.format(loops,
                                                                        int(t_min),
                                                                        int(t_sec),
                                                                        int(t_msec),
                                                                        int(t_usec),
                                                                        int(t_nsec)))

            elif str_id == "PM":
                print('Time of PM acceleration calculation averaged over {} loops: \n'
                      '{} min {} sec {} msec {} usec {} nsec \n'.format(loops,
                                                                        int(t_min),
                                                                        int(t_sec),
                                                                        int(t_msec),
                                                                        int(t_usec),
                                                                        int(t_nsec)))

            elif str_id == "Equilibration":
                print('\n\n----------------- Averaged Evolution Times ---------------------\n')
                print('Time of a single equilibration step averaged over {} loops: \n'
                      '{} min {} sec {} msec {} usec {} nsec \n'.format(loops,
                                                                        int(t_min),
                                                                        int(t_sec),
                                                                        int(t_msec),
                                                                        int(t_usec),
                                                                        int(t_nsec)))

            elif str_id == "Production":
                print('Time of a single production step averaged over {} loops: \n'
                      '{} min {} sec {} msec {} usec {} nsec \n'.format(loops,
                                                                        int(t_min),
                                                                        int(t_sec),
                                                                        int(t_msec),
                                                                        int(t_usec),
                                                                        int(t_nsec)))

                print('\n\n----------------- Total Simulation Times -----------------------')

            repeat -= 1
            sys.stdout = screen

        f_log.close()

    @staticmethod
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

    @staticmethod
    def time_info(simulation):
        """
        Print time simulation's parameters.

        Parameters
        ----------
        simulation: sarkas.base.Simulation
            Simulation's parameters.

        """
        print('Time step = {:2.6e} [s]'.format(simulation.integrator.dt))
        if simulation.potential.type in ['Yukawa', 'EGS', 'Coulomb', 'Moliere']:
            print('(total) plasma frequency = {:1.6e} [Hz]'.format(simulation.parameters.total_plasma_frequency))
            print('wp dt = {:2.4f}'.format(simulation.integrator.dt * simulation.parameters.total_plasma_frequency))
        elif simulation.potential.type == 'QSP':
            print('e plasma frequency = {:2.6e} [Hz]'.format(simulation.species[0].wp))
            print('ion plasma frequency = {:2.6e} [Hz]'.format(simulation.species[1].wp))
            print('w_pe dt = {:2.4f}'.format(simulation.integrator.dt * simulation.species[0].wp))
        elif simulation.potential.type == 'LJ':
            print('(total) equivalent plasma frequency = {:1.6e} [Hz]'.format(
                simulation.parameters.total_plasma_frequency))
            print('wp dt = {:2.4f}'.format(simulation.integrator.dt * simulation.parameters.total_plasma_frequency))

        if simulation.parameters == 'prod_restart':
            print("Restart step: {}".format(simulation.parameters.restart_step))
            print('Total post-equilibration steps = {} ~ {} wp T_prod'.format(
                simulation.integrator.production_steps,
                int(
                    simulation.integrator.production_steps * simulation.parameters.total_plasma_frequency * simulation.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                simulation.integrator.prod_dump_step,
                simulation.integrator.prod_dump_step * simulation.integrator.dt * simulation.parameters.total_plasma_frequency))
        elif simulation.parameters == 'eq_restart':
            print("Restart step: {}".format(simulation.parameters.load_therm_restart_step))
            print('Total equilibration steps = {} ~ {} wp T_prod'.format(
                simulation.integrator.equilibration_steps,
                int(
                    simulation.integrator.eq_dump_step * simulation.parameters.total_plasma_frequency * simulation.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                simulation.integrator.eq_dump_step,
                simulation.integrator.eq_dump_step * simulation.integrator.dt * simulation.parameters.total_plasma_frequency))
        else:
            print('No. of equilibration steps = {} ~ {} wp T_eq'.format(
                simulation.integrator.equilibration_steps,
                int(
                    simulation.integrator.equilibration_steps * simulation.parameters.total_plasma_frequency * simulation.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                simulation.integrator.eq_dump_step,
                simulation.integrator.eq_dump_step * simulation.integrator.dt * simulation.parameters.total_plasma_frequency))
            print('No. of post-equilibration steps = {} ~ {} wp T_prod'.format(
                simulation.integrator.production_steps,
                int(
                    simulation.integrator.production_steps * simulation.parameters.total_plasma_frequency * simulation.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                simulation.integrator.prod_dump_step,
                simulation.integrator.prod_dump_step * simulation.integrator.dt * simulation.parameters.total_plasma_frequency))

    @staticmethod
    def algorithm_info(simulation):
        """
        Print algorithm information.

        Parameters
        ----------
        simulation: sarkas.base.Simulation
            Simulation's parameters.


        """
        if simulation.potential.method == 'P3M':
            print('Ewald parameter alpha = {:2.4f} / a_ws = {:1.6e} '.format(
                simulation.potential.pppm_alpha_ewald * simulation.parameters.a_ws,
                simulation.potential.pppm_alpha_ewald), end='')
            print("[1/cm]" if simulation.parameters.units == "cgs" else "[1/m]")
            print('Mesh size * Ewald_parameter (h * alpha) = {:2.4f}, {:2.4f}, {:2.4f} '.format(
                simulation.potential.pppm_h_array[0] * simulation.potential.pppm_alpha_ewald,
                simulation.potential.pppm_h_array[1] * simulation.potential.pppm_alpha_ewald,
                simulation.potential.pppm_h_array[2] * simulation.potential.pppm_alpha_ewald))
            print('                                        ~ 1/{}, 1/{}, 1/{}'.format(
                int(1. / (simulation.potential.pppm_h_array[0] * simulation.potential.pppm_alpha_ewald)),
                int(1. / (simulation.potential.pppm_h_array[1] * simulation.potential.pppm_alpha_ewald)),
                int(1. / (simulation.potential.pppm_h_array[2] * simulation.potential.pppm_alpha_ewald)),
            ))
            print(
                'rcut = {:2.4f} a_ws = {:2.6e} '.format(simulation.potential.rc / simulation.parameters.a_ws,
                                                        simulation.potential.rc), end='')
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print('Mesh = {} x {} x {}'.format(*simulation.potential.pppm_mesh))
            print('No. of PP cells per dimension = {:2}, {:2}, {:2}'.format(
                int(simulation.parameters.box_lengths[0] / simulation.potential.rc),
                int(simulation.parameters.box_lengths[1] / simulation.potential.rc),
                int(simulation.parameters.box_lengths[2] / simulation.potential.rc)))
            print('No. of particles in PP loop = {:6}'.format(
                int(simulation.parameters.total_num_density * (3 * simulation.potential.rc) ** 3)))
            print('No. of PP neighbors per particle = {:6}'.format(
                int(simulation.parameters.total_num_ptcls * 4.0 / 3.0 * np.pi * (
                        simulation.potential.rc / simulation.parameters.box_lengths.min()) ** 3.0)))
            print('PM Force Error = {:2.6e}'.format(simulation.parameters.pppm_pm_err))
            print('PP Force Error = {:2.6e}'.format(simulation.parameters.pppm_pp_err))

        elif simulation.potential.method == 'PP':
            print(
                'rcut = {:2.4f} a_ws = {:2.6e} '.format(simulation.potential.rc / simulation.parameters.a_ws,
                                                        simulation.potential.rc),
                end='')
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print('No. of PP cells per dimension = {:2}, {:2}, {:2}'.format(
                int(simulation.parameters.box_lengths[0] / simulation.potential.rc),
                int(simulation.parameters.box_lengths[1] / simulation.potential.rc),
                int(simulation.parameters.box_lengths[2] / simulation.potential.rc)))
            print('No. of particles in PP loop = {:6}'.format(
                int(simulation.parameters.total_num_density * (3 * simulation.potential.rc) ** 3)))
            print('No. of PP neighbors per particle = {:6}'.format(
                int(simulation.parameters.total_num_ptcls * 4.0 / 3.0 * np.pi * (
                        simulation.potential.rc / simulation.parameters.box_lengths.min()) ** 3.0)))

        print('Tot Force Error = {:2.6e}'.format(simulation.parameters.force_error))

    @staticmethod
    def potential_info(simulation):
        """
        Print potential information.

        Parameters
        ----------
        simulation: sarkas.base.Simulation
            Simulation's parameters.

        """
        if simulation.potential.type == 'Yukawa':
            print('kappa = {:1.4e}'.format(simulation.parameters.a_ws / simulation.parameters.lambda_TF))
            print('lambda_TF = {:1.4e}'.format(simulation.parameters.lambda_TF))
            print('Gamma_eff = {:4.2f}'.format(simulation.parameters.coupling_constant))

        elif simulation.potential.type == 'EGS':
            print('kappa = {:1.4e}'.format(simulation.parameters.a_ws / simulation.parameters.lambda_TF))
            print('lambda_TF = {:1.4e}'.format(simulation.parameters.lambda_TF))
            print('nu = {:1.4e}'.format(simulation.parameters.nu))
            if simulation.parameters.nu < 1:
                print('Exponential decay:')
                print('lambda_p = {:1.4e}'.format(simulation.parameters.lambda_p))
                print('lambda_m = {:1.4e}'.format(simulation.parameters.lambda_m))
                print('alpha = {:1.4e}'.format(simulation.parameters.alpha))
                print('Theta = {:1.4e}'.format(simulation.parameters.electron_degeneracy_parameter))
                print('b = {:1.4e}'.format(simulation.parameters.b))

            else:
                print('Oscillatory potential:')
                print('gamma_p = {:1.4e}'.format(simulation.parameters.gamma_p))
                print('gamma_m = {:1.4e}'.format(simulation.parameters.gamma_m))
                print('alpha = {:1.4e}'.format(simulation.parameters.alphap))
                print('Theta = {:1.4e}'.format(simulation.parameters.theta))
                print('b = {:1.4e}'.format(simulation.parameters.b))

            print('Gamma_eff = {:4.2f}'.format(simulation.parameters.coupling_constant))

        elif simulation.potential.type == 'Coulomb':
            print('Gamma_eff = {:4.2f}'.format(simulation.parameters.coupling_constant))

        elif simulation.potential.type == 'LJ':
            print('epsilon = {:2.6e}'.format(simulation.potential.matrix[0, 0, 0]))
            print('sigma = {:2.6e}'.format(simulation.potential.matrix[1, 0, 0]))
            print('Gamma_eff = {:4.2f}'.format(simulation.parameters.coupling_constant))

        elif simulation.potential.type == "QSP":
            print("e de Broglie wavelength = {:2.4f} ai = {:2.6e} ".format(
                2.0 * np.pi / simulation.potential.matrix[1, 0, 0] / (np.sqrt(2.0) * simulation.parameters.ai),
                2.0 * np.pi / simulation.potential.matrix[1, 0, 0] / np.sqrt(2.0)), end='')
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print("e-e screening parameter = {:2.4f}".format(
                simulation.potential.matrix[1, 0, 0] * simulation.parameters.a_ws))
            print("ion de Broglie wavelength  = {:2.4f} ai = {:2.6e} ".format(
                2.0 * np.pi / simulation.potential.matrix[1, 0, 0] / (np.sqrt(2.0) * simulation.parameters.ai),
                2.0 * np.pi / simulation.potential.matrix[1, 0, 0] / np.sqrt(2.0)), end='')
            print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
            print("i-i screening parameter = {:2.4f}".format(
                simulation.potential.matrix[1, 1, 1] * simulation.parameters.a_ws))
            print("e-i screening parameter = {:2.4f}".format(
                simulation.potential.matrix[1, 0, 1] * simulation.parameters.a_ws))
            print("e-i Coupling Parameter = {:3.3f} ".format(simulation.parameters.coupling_constant))
            print("rs Coupling Parameter = {:3.3f} ".format(simulation.parameters.rs))

    @staticmethod
    def thermostat_info(simulation):
        """
        Print thermostat information.

        Parameters
        ----------
        simulation: sarkas.base.Simulation
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
        simulation: sarkas.base.Simulation
            Simulation's parameters.

        """
        print('Wigner-Seitz radius = {:2.6e} '.format(simulation.parameters.a_ws), end='')
        print("[cm]" if simulation.parameters.units == "cgs" else "[m]")
        print('No. of non-zero box dimensions = ', int(simulation.parameters.dimensions))
        print('Box length along x axis = {:2.6e} a_ws = {:2.6e} '.format(
            simulation.parameters.box_lengths[0] / simulation.parameters.a_ws, simulation.parameters.box_lengths[0]),
            end='')
        print("[cm]" if simulation.parameters.units == "cgs" else "[m]")

        print('Box length along y axis = {:2.6e} a_ws = {:2.6e} '.format(
            simulation.parameters.box_lengths[1] / simulation.parameters.a_ws, simulation.parameters.box_lengths[1]),
            end='')
        print("[cm]" if simulation.parameters.units == "cgs" else "[m]")

        print('Box length along z axis = {:2.6e} a_ws = {:2.6e} '.format(
            simulation.parameters.box_lengths[2] / simulation.parameters.a_ws, simulation.parameters.box_lengths[2]),
            end='')
        print("[cm]" if simulation.parameters.units == "cgs" else "[m]")

        print("The remaining lengths scales are given in ", end='')
        print("[cm]" if simulation.parameters.units == "cgs" else "[m]")

    @staticmethod
    def species_info(simulation):
        """
        Print Species information.

        Parameters
        ----------
        simulation: sarkas.base.Simulation
            Simulation's parameters.

        """
        print('No. of species = ', len(simulation.species))
        for isp, sp in enumerate(simulation.species):
            print("Species {} : {}".format(isp + 1, sp.name))
            print("\tSpecies ID: {}".format(isp))
            print("\tNo. of particles = {} ".format(sp.num))
            print("\tNumber density = {:2.6e} ".format(sp.number_density), end='')
            print("[N/cc]" if simulation.parameters.units == "cgs" else "[N/m^3]")
            print("\tMass = {:2.6e} ".format(sp.mass), end='')
            print("[g]" if simulation.parameters.units == "cgs" else "[kg]")
            print("\tCharge = {:2.6e} ".format(sp.charge), end='')
            print("[esu]" if simulation.parameters.units == "cgs" else "[C]")
            print('\tTemperature = {:2.6e} [K]'.format(sp.temperature))

    def setup_checkpoint(self, params, species):
        """
        Assign attributes needed for saving dumps.

        Parameters
        ----------
        params: sarkas.base.Parameters
            General simulation parameters.

        species: sarkas.base.Species
            List of Species classes.

        """
        self.dt = params.dt
        self.a_ws = params.a_ws
        self.total_num_ptcls = params.total_num_ptcls
        self.total_plasma_frequency = params.total_plasma_frequency
        self.species_names = np.copy(params.species_names)
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

    def save_pickle(self, simulation):
        """
        Save all simulations parameters in pickle files.
        """
        file_list = ['parameters', 'integrator', 'thermostat', 'potential', 'species']
        for fl in file_list:
            filename = os.path.join(self.job_dir, fl + ".pickle")
            pickle_file = open(filename, "wb")
            pickle.dump(simulation.__dict__[fl], pickle_file)
            pickle_file.close()

    def read_pickle(self, process):
        """
        Read pickle files containing all the simulation information.

        Parameters
        ----------
        process: cls
            Simulation's parameters. It can be one of three (sarkas.tools.PreProcess,
            sarkas.base.Simulation, sarkas.tools.PostProcess)
        """
        import copy as py_copy
        file_list = ['parameters', 'integrator', 'thermostat', 'potential', 'species']
        for fl in file_list:
            filename = os.path.join(self.job_dir, fl + ".pickle")
            data = np.load(filename, allow_pickle=True)
            process.__dict__[fl] = py_copy.copy(data)

    def read_pickle_single(self, class_to_read):
        """
        Read the desired pickle file.

        Parameters
        ----------
        class_to_read : str
            Name of the class to read.

        Returns
        -------
        : cls
            Desired class.

        """
        import copy as py_copy
        file_list = ['parameters', 'integrator', 'thermostat', 'potential', 'species']

        filename = os.path.join(self.job_dir, class_to_read + ".pickle")
        data = np.load(filename, allow_pickle=True)
        return py_copy.copy(data)

    def dump(self, production, ptcls, it):
        """
        Save particles' data to binary file for future restart.

        Parameters
        ----------
        production: bool
            Flag indicating whether to phase production or equilibration data.

        ptcls: sarkas.base.Particles
            Particles data.

        it : int
            Timestep number.
        """
        if production:
            ptcls_file = self.prod_ptcls_filename + str(it)
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
            ptcls_file = self.eq_ptcls_filename + str(it)
            tme = it * self.dt
            np.savez(ptcls_file,
                     id=ptcls.id,
                     name=ptcls.names,
                     pos=ptcls.pos,
                     vel=ptcls.vel,
                     acc=ptcls.acc,
                     time=tme)

            energy_file = self.eq_energy_filename

        kinetic_energies, temperatures = ptcls.kinetic_temperature()
        # Prepare data for saving
        data = {"Time": it * self.dt,
                "Total Energy": np.sum(kinetic_energies) + ptcls.potential_energy,
                "Total Kinetic Energy": np.sum(kinetic_energies),
                "Potential Energy": ptcls.potential_energy,
                "Total Temperature": ptcls.species_num.transpose() @ temperatures / ptcls.total_num_ptcls
                }
        for sp, kin in enumerate(kinetic_energies):
            data["{} Kinetic Energy".format(self.species_names[sp])] = kin
            data["{} Temperature".format(self.species_names[sp])] = temperatures[sp]
        with open(energy_file, 'a') as f:
            w = csv.writer(f)
            w.writerow(data.values())

    def dump_xyz(self, phase='production'):
        """
        Save the XYZ file by reading Sarkas dumps.

        Parameters
        ----------
        phase : str
            Phase from which to read dumps. 'equilibration' or 'production'.

        dump_skip : int
            Interval of dumps to skip. Default = 1

        """

        if phase == 'equilibration':
            self.xyz_filename = os.path.join(self.equilibration_dir, "pva_" + self.job_id + '.xyz')
            dump_dir = self.eq_dump_dir

        else:
            self.xyz_filename = os.path.join(self.production_dir, "pva_" + self.job_id + '.xyz')
            dump_dir = self.prod_dump_dir

        f_xyz = open(self.xyz_filename, "w+")

        if not hasattr(self, 'a_ws'):
            params = self.read_pickle_single('parameters')
            self.a_ws = params.a_ws
            self.total_num_ptcls = params.total_num_ptcls
            self.total_plasma_frequency = params.total_plasma_frequency

        # Rescale constants. This is needed since OVITO has a small number limit.
        pscale = 1.0 / self.a_ws
        vscale = 1.0 / (self.a_ws * self.total_plasma_frequency)
        ascale = 1.0 / (self.a_ws * self.total_plasma_frequency ** 2)

        # Read the list of dumps and sort them in the correct (natural) order
        dumps = os.listdir(dump_dir)
        dumps.sort(key=num_sort)
        for dump in tqdm(dumps, disable=not self.verbose):
            data = self.read_npz(dump_dir, dump)
            data["pos_x"] *= pscale
            data["pos_y"] *= pscale
            data["pos_z"] *= pscale

            data["vel_x"] *= vscale
            data["vel_y"] *= vscale
            data["vel_z"] *= vscale

            data["acc_x"] *= ascale
            data["acc_y"] *= ascale
            data["acc_z"] *= ascale

            f_xyz.writelines("{0:d}\n".format(self.total_num_ptcls))
            f_xyz.writelines("name x y z vx vy vz ax ay az\n")
            np.savetxt(f_xyz, data, fmt="%s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")

        f_xyz.close()

    @staticmethod
    def read_npz(fldr, it):
        """
        Load particles' data from dumps.

        Parameters
        ----------
        fldr : str
            Folder containing dumps.

        it : str
            Timestep to load.

        Returns
        -------
        struct_array : numpy.ndarray
            Structured data array.

        """

        file_name = os.path.join(fldr, it)
        data = np.load(file_name, allow_pickle=True)
        # Dev Notes: the old way of saving the xyz file by
        # np.savetxt(f_xyz, np.c_[data["names"],data["pos"] ....]
        # , fmt="%10s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")
        # was not working, because the columns of np.c_[] all have the same data type <U32
        # which is in conflict with the desired fmt. i.e. data["names"] was not recognized as a string.
        # So I have to create a new structured array and pass this. I could not think of a more Pythonic way.
        struct_array = np.zeros(data["names"].size,
                                dtype=[('names', 'U6'),
                                       ('pos_x', np.float64),
                                       ('pos_y', np.float64),
                                       ('pos_z', np.float64),
                                       ('vel_x', np.float64),
                                       ('vel_y', np.float64),
                                       ('vel_z', np.float64),
                                       ('acc_x', np.float64),
                                       ('acc_y', np.float64),
                                       ('acc_z', np.float64)]
                                )
        struct_array["names"] = data["names"]
        struct_array["pos_x"] = data["pos"][:, 0]
        struct_array["pos_y"] = data["pos"][:, 1]
        struct_array["pos_z"] = data["pos"][:, 2]

        struct_array["vel_x"] = data["vel"][:, 0]
        struct_array["vel_y"] = data["vel"][:, 1]
        struct_array["vel_z"] = data["vel"][:, 2]

        struct_array["acc_x"] = data["acc"][:, 0]
        struct_array["acc_y"] = data["acc"][:, 1]
        struct_array["acc_z"] = data["acc"][:, 2]

        return struct_array


def alpha_to_int(text):
    return int(text) if text.isdigit() else text


def num_sort(text):
    """
    Method copied from
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside

    Notes
    -----
    originally from http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    Parameters
    ----------
    text : str
        Text to be split into str and int

    Returns
    -------
     : list
        List containing text and integers

    """

    return [alpha_to_int(c) for c in re.split(r'(\d+)', text)]
