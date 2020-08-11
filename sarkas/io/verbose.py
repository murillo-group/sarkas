import os
import sys
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
        self.verbose = params.control.verbose
        self.pre_run = params.control.pre_run
        # Create job folder if non existent
        if not os.path.exists(params.control.job_dir):
            os.mkdir(params.control.job_dir)

        pre_run_path = os.path.join(params.control.job_dir, 'Pre_Run_Test')
        if not os.path.exists(pre_run_path):
            os.mkdir(pre_run_path)
        params.control.pre_run_dir = pre_run_path
        # Pre run file name
        self.f_pre_run = os.path.join(params.control.pre_run_dir, 'pre_run_' + params.control.job_id + '.out')
        # Log File name
        self.f_log_name = os.path.join(params.control.job_dir, "log_" + params.control.job_id + ".out")
        # Save it in params too
        params.control.log_file = self.f_log_name

        # Pre run testing: assign self.io_file to the correct file to open/write
        self.io_file = self.f_pre_run if params.control.pre_run else self.f_log_name

        # Print figlet to file if not a restart run
        if not params.control.restart == "restart" and not params.control.restart == "therm_restart":
            with open(self.io_file, "w+") as f_log:
                figlet_obj = Figlet(font='starwars')
                print(figlet_obj.renderText('Sarkas'), file=f_log)
                print("An open-source pure-Python molecular dynamics code for non-ideal plasmas.", file=f_log)

        # Print figlet to screen if verbose
        if self.verbose:
            screen_figlet()

    def sim_setting_summary(self, params):
        """
        Print out to file a summary of simulation's parameters.
        If verbose output then it will print twice: the first time to file and second time to screen.

        Parameters
        ----------
        params : pbject
            Simulation's parameters

        """

        screen = sys.stdout
        f_log = open(self.io_file, 'a+')
        repeat = 2 if self.verbose else 1
        print(repeat)
        # redirect printing to file
        sys.stdout = f_log

        # Print to file first then to screen if repeat == 2
        while repeat > 0:

            if params.control.restart == 'restart':
                print('\n\n--------------------------- Restart -------------------------------------')
                self.time_info(params)
            elif params.control.restart == 'therm_restart':
                print('\n\n------------------------ Therm Restart ----------------------------------')
                self.time_info(params)
            else:
                # Choose the correct heading
                if self.pre_run:
                    print('\n\n-------------- Pre Run Details ----------------------')
                else:
                    print('\n\n----------------- Simulation -----------------------')

                print('\nJob ID: ', params.control.job_id)
                print('Job directory: ', params.control.job_dir)
                print('Dump directory: ', params.control.prod_dump_dir)
                print('\nUnits: ', params.control.units)
                print('Total No. of particles = ', params.total_num_ptcls)

                print('\nParticle Species:')
                self.species_info(params)

                print('\nLengths scales:')
                self.length_info(params)

                print('\nBoundary conditions:')
                if params.BC.pbc_axes:
                    print('Periodic BC along axes : ', params.BC.pbc_axes)
                if params.BC.mm_axes:
                    print('Momentum Mirror BC along axes : ', params.BC.mm_axes)
                if params.BC.open_axes:
                    print('Open BC along axes : ', params.BC.open_axes)

                if params.Langevin.on:
                    print('Langevin model : ', params.Langevin.type)

                print("\nThermostat: ", params.thermostat.type)
                self.thermostat_info(params)

                print('\nPotential: ', params.potential.type)
                self.potential_info(params)

                if params.Magnetic.on:
                    print('\nMagnetized Plasma:')
                    for ic in range(params.num_species):
                        print('Cyclotron frequency of Species {:2} = {:2.6e}'.format(ic + 1,
                                                                                     params.species[ic].omega_c))
                        print('beta_c of Species {:2} = {:2.6e}'.format(ic + 1,
                                                                        params.species[ic].omega_c / params.species[
                                                                            ic].wp))
                print("\nAlgorithm: ", params.potential.method)
                self.algorithm_info(params)

                print("\nIntegrator: ", params.integrator.type)

                print("\nTime scales:")
                self.time_info(params)

            repeat -= 1
            print(repeat)
            sys.stdout = screen  # Restore the original sys.stdout

        f_log.close()

    def time_stamp(self, time_stamp, t):
        """
        Print out to screen elapsed times. If verbose output, print to file first and then to screen.

        Para meters
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

    def timing_study(self, params):
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
            print('Suggested Mesh = [ {} , {} , {} ]'.format(*params.pppm.MGrid))
            print('Suggested Ewald parameter alpha = {:2.4f} / a_ws = {:1.6e} '.format(params.pppm.G_ew * params.aws,
                                                                                       params.pppm.G_ew), end='')
            print("[1/cm]" if params.control.units == "cgs" else "[1/m]")
            print('Suggested rcut = {:2.4f} a_ws = {:2.6e} '.format(params.potential.rc / params.aws,
                                                                    params.potential.rc), end='')
            print("[cm]" if params.control.units == "cgs" else "[m]")

            self.algorithm_info(self, params)
            # print("\nAlgorithm : ", params.potential.method)
            # if params.potential.method == 'P3M':
            #     print('Mesh size * Ewald_parameter (h * alpha) = {:2.4f} ~ 1/{} '.format(
            #         params.pppm.hx * params.pppm.G_ew, int(1. / (params.pppm.hx * params.pppm.G_ew))))
            #     print('No. of PP cells per dimension = {:2}, {:2}, {:2}'.format(
            #         int(params.Lv[0] / params.potential.rc),
            #         int(params.Lv[1] / params.potential.rc),
            #         int(params.Lv[2] / params.potential.rc)))
            #     print('No. of particles in PP loop = {:6}'.format(
            #         int(params.total_num_density * (3 * params.potential.rc) ** 3)))
            #     print('No. of PP neighbors per particle = {:6}'.format(
            #         int(params.total_num_ptcls * 4.0 / 3.0 * np.pi * (params.potential.rc / params.Lv.min()) ** 3.0)))
            #     print('PM Force Error = {:2.6e}'.format(params.pppm.PM_err))
            #     print('PP Force Error = {:2.6e}'.format(params.pppm.PP_err))
            #     print('Tot Force Error = {:2.6e}'.format(params.pppm.F_err))
            # elif params.potential.method == 'PP':
            #     print('rcut/a_ws = {:2.6e}'.format(params.potential.rc / params.aws))
            #     print(
            #         'No. of cells per dimension = {:2}, {:2}, {:2}'.format(int(params.Lv[0] / params.potential.rc),
            #                                                                int(params.Lv[1] / params.potential.rc),
            #                                                                int(params.Lv[2] / params.potential.rc)))
            #     print('No. of neighbors per particle = {:4}'.format(
            #         int(params.total_num_ptcls * 4.0 / 3.0 * np.pi * (
            #                 params.potential.rc / params.Lv.min()) ** 3.0)))
            #     print('PP Force Error = {:2.6e}'.format(params.PP_err))

            repeat -= 1
            sys.stdout = screen  # Restore the original sys.stdout

        f_log.close()

    @staticmethod
    def time_info(params):
        """
        Print time simulation's parameters.

        Parameters
        ----------
        params: object
            Simulation's parameters.

        """
        print('Time step = {:2.6e} [s]'.format(params.integrator.dt))
        if params.potential.type in ['Yukawa', 'EGS', 'Coulomb', 'Moliere']:
            print('(total) plasma frequency = {:1.6e} [Hz]'.format(params.wp))
            print('wp dt = {:2.4f}'.format(params.integrator.dt * params.wp))
        elif params.potential.type == 'QSP':
            print('e plasma frequency = {:2.6e} [Hz]'.format(params.species[0].wp))
            print('ion plasma frequency = {:2.6e} [Hz]'.format(params.species[1].wp))
            print('w_pe dt = {:2.4f}'.format(params.integrator.dt * params.species[0].wp))
        elif params.potential.type == 'LJ':
            print('(total) equivalent plasma frequency = {:1.6e} [Hz]'.format(params.wp))
            print('wp dt = {:2.4f}'.format(params.integrator.dt * params.wp))

        if params.control.restart == 'restart':
            print("Restart step: {}".format(params.load_restart_step))
            print('Total post-equilibration steps = {} ~ {} wp T_prod'.format(
                params.integrator.nsteps_prod, int(params.integrator.nsteps_prod * params.wp * params.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                params.integrator.prod_dump_step, params.integrator.prod_dump_step * params.integrator.dt * params.wp))
        elif params.control.restart == 'therm_restart':
            print("Restart step: {}".format(params.load_therm_restart_step))
            print('Total thermalization steps = {} ~ {} wp T_prod'.format(
                params.integrator.nsteps_prod, int(params.integrator.nsteps_prod * params.wp * params.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                params.integrator.prod_dump_step, params.integrator.prod_dump_step * params.integrator.dt * params.wp))
        else:
            print('No. of equilibration steps = {} ~ {} wp T_eq'.format(
                params.integrator.nsteps_eq, int(params.integrator.nsteps_eq * params.wp * params.integrator.dt)))
            print('No. of post-equilibration steps = {} ~ {} wp T_prod'.format(
                params.integrator.nsteps_prod, int(params.integrator.nsteps_prod * params.wp * params.integrator.dt)))
            print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                params.integrator.prod_dump_step, params.integrator.prod_dump_step * params.integrator.dt * params.wp))

    @staticmethod
    def algorithm_info(params):
        """
        Print algorithm information.

        Parameters
        ----------
        params: object
            Simulation's parameters.


        """
        if params.potential.method == 'P3M':
            print('Ewald parameter alpha = {:2.4f} / a_ws = {:1.6e} '.format(params.pppm.G_ew * params.aws,
                                                                             params.pppm.G_ew), end='')
            print("[1/cm]" if params.control.units == "cgs" else "[1/m]")
            print('Mesh size * Ewald_parameter (h * alpha) = {:2.4f} ~ 1/{} '.format(
                params.pppm.hx * params.pppm.G_ew, int(1. / (params.pppm.hx * params.pppm.G_ew))), )
            print(
                'rcut = {:2.4f} a_ws = {:2.6e} '.format(params.potential.rc / params.aws, params.potential.rc),
                end='')
            print("[cm]" if params.control.units == "cgs" else "[m]")
            print('Mesh = {} x {} x {}'.format(*params.pppm.MGrid))
            print('No. of PP cells per dimension = {:2}, {:2}, {:2}'.format(
                int(params.Lv[0] / params.potential.rc),
                int(params.Lv[1] / params.potential.rc),
                int(params.Lv[2] / params.potential.rc)))
            print('No. of particles in PP loop = {:6}'.format(
                int(params.total_num_density * (3 * params.potential.rc) ** 3)))
            print('No. of PP neighbors per particle = {:6}'.format(
                int(params.total_num_ptcls * 4.0 / 3.0 * np.pi * (
                        params.potential.rc / params.Lv.min()) ** 3.0)))
            print('PM Force Error = {:2.6e}'.format(params.pppm.PM_err))
            print('PP Force Error = {:2.6e}'.format(params.pppm.PP_err))
            print('Tot Force Error = {:2.6e}'.format(params.pppm.F_err))
        elif params.potential.method == 'PP':
            print(
                'rcut = {:2.4f} a_ws = {:2.6e} '.format(params.potential.rc / params.aws, params.potential.rc),
                end='')
            print("[cm]" if params.control.units == "cgs" else "[m]")
            print(
                'No. of cells per dimension = {:2}, {:2}, {:2}'.format(int(params.Lv[0] / params.potential.rc),
                                                                       int(params.Lv[1] / params.potential.rc),
                                                                       int(params.Lv[2] / params.potential.rc)))
            print('No. of neighbors per particle = {:4}'.format(
                int(params.total_num_ptcls * 4.0 / 3.0 * np.pi * (
                        params.potential.rc / params.Lv.min()) ** 3.0)))
            print('PP Force Error = {:2.6e}'.format(params.PP_err))

    @staticmethod
    def potential_info(params):
        """
        Print potential information.

        Parameters
        ----------
        params: object
            Simulation's parameters.

        """
        if params.potential.type == 'Yukawa':
            print('kappa = {:1.4e}'.format(params.potential.matrix[1, 0, 0] * params.aws))
            print('lambda_TF = {:1.4e}'.format(params.lambda_TF))
            print('Gamma_eff = {:4.2f}'.format(params.potential.Gamma_eff))

        elif params.potential.type == 'EGS':
            print('Gamma_eff = {:4.2f}'.format(params.potential.Gamma_eff))
            print('lambda_TF = {:1.4e}'.format(params.lambda_TF))
            print('kappa = {:1.4e}'.format(params.potential.matrix[0, 0, 0] * params.aws))
            print('nu = {:1.4e}'.format(params.potential.nu))
            if params.potential.nu < 1:
                print('Exponential decay:')
                print('lambda_p = {:1.4e}'.format(params.potential.lambda_p))
                print('lambda_m = {:1.4e}'.format(params.potential.lambda_m))
                print('alpha = {:1.4e}'.format(params.potential.alpha))
                print('Theta = {:1.4e}'.format(params.potential.theta))
                print('b = {:1.4e}'.format(params.potential.b))

            else:
                print('Oscillatory potential:')
                print('gamma_p = {:1.4e}'.format(params.potential.gamma_p))
                print('gamma_m = {:1.4e}'.format(params.potential.gamma_m))
                print('alpha = {:1.4e}'.format(params.potential.alphap))
                print('Theta = {:1.4e}'.format(params.potential.theta))
                print('b = {:1.4e}'.format(params.potential.b))

        elif params.potential.type == 'Coulomb':
            print('Gamma_eff = {:4.2f}'.format(params.potential.Gamma_eff))

        elif params.potential.type == 'LJ':
            print('epsilon = {:2.6e}'.format(params.potential.matrix[0, 0, 0]))
            print('sigma = {:2.6e}'.format(params.potential.matrix[1, 0, 0]))

        elif params.potential.type == "QSP":
            print("e de Broglie wavelength = {:2.4f} ai = {:2.6e} ".format(
                2.0 * np.pi / params.potential.matrix[1, 0, 0] / (np.sqrt(2.0) * params.ai),
                2.0 * np.pi / params.potential.matrix[1, 0, 0] / np.sqrt(2.0)), end='')
            print("[cm]" if params.control.units == "cgs" else "[m]")
            print("e-e screening parameter = {:2.4f}".format(params.potential.matrix[1, 0, 0] * params.aws))
            print("ion de Broglie wavelength  = {:2.4f} ai = {:2.6e} ".format(
                2.0 * np.pi / params.potential.matrix[1, 0, 0] / (np.sqrt(2.0) * params.ai),
                2.0 * np.pi / params.potential.matrix[1, 0, 0] / np.sqrt(2.0)), end='')
            print("[cm]" if params.control.units == "cgs" else "[m]")
            print("i-i screening parameter = {:2.4f}".format(params.potential.matrix[1, 1, 1] * params.aws))
            print("e-i screening parameter = {:2.4f}".format(params.potential.matrix[1, 0, 1] * params.aws))
            print("e-i Coupling Parameter = {:3.3f} ".format(params.potential.Gamma_eff))
            print("rs Coupling Parameter = {:3.3f} ".format(params.rs))

    @staticmethod
    def thermostat_info(params):
        """
        Print thermostat information.

        Parameters
        ----------
        params: object
            Simulation's parameters.

        """
        print("Berendsen Relaxation rate: {:1.3f}".format(1.0 / params.thermostat.relaxation_rate))
        print("Thermostating Temperatures: ", params.thermostat.temperatures)

    @staticmethod
    def length_info(params):
        """
        Print length information.

        Parameters
        ----------
        params: object
            Simulation's parameters.

        """
        print('Wigner-Seitz radius = {:2.6e} '.format(params.aws), end='')
        print("[cm]" if params.control.units == "cgs" else "[m]")
        print('No. of non-zero box dimensions = ', int(params.dimensions))
        print('Box length along x axis = {:2.6e} a_ws = {:2.6e} '.format(params.Lv[0] / params.aws,
                                                                         params.Lv[0]), end='')
        print("[cm]" if params.control.units == "cgs" else "[m]")
        print('Box length along y axis = {:2.6e} a_ws = {:2.6e} '.format(params.Lv[1] / params.aws,
                                                                         params.Lv[1]), end='')
        print("[cm]" if params.control.units == "cgs" else "[m]")
        print('Box length along z axis = {:2.6e} a_ws = {:2.6e} '.format(params.Lv[2] / params.aws,
                                                                         params.Lv[2]), end='')
        print("[cm]" if params.control.units == "cgs" else "[m]")
        print("The remaining lengths scales are given in ", end='')
        print("[cm]" if params.control.units == "cgs" else "[m]")

    @staticmethod
    def species_info(params):
        print('No. of species = ', len(params.species))
        for isp, sp in enumerate(params.species):
            print("Species {} : {}".format(isp + 1, sp.name))
            print("\tSpecies ID: {}".format(isp))
            print("\tNo. of particles = {} ".format(sp.num))
            print("\tNumber density = {:2.6e} ".format(sp.number_density), end='')
            print("[N/cc]" if params.control.units == "cgs" else "[N/m^3]")
            print("\tMass = {:2.6e} ".format(sp.mass), end='')
            print("[g]" if params.control.units == "cgs" else "[kg]")
            print('\tTemperature = {:2.6e} [K]'.format(sp.temperature))

