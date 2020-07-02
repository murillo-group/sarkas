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


# BG_COLORS = ['24;69;49',
#              '0;129;131',
#              '83;80;84',
#              '110;0;95'
#              ]


def screen_figlet():
    """
    Print a colored figlet of Sarkas to screen.
    """
    fg = FG_COLORS[np.random.randint(0, len(FG_COLORS))]
    # bg = BG_COLORS[np.random.randint(0, len(BG_COLORS))]
    fnt = FONTS[np.random.randint(0, len(FONTS))]
    clr = fg  # + ':' + bg
    print_figlet('\n\tSarkas\n', font=fnt, colors=clr)

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
    params : class
        Simulation's parameters

    """

    def __init__(self, params):
        self.verbose = params.Control.verbose
        self.pre_run = params.Control.pre_run
        # Create job folder if non existent
        if not os.path.exists(params.Control.checkpoint_dir):
            os.mkdir(params.Control.checkpoint_dir)

        # Pre run file name
        self.f_pre_run = os.path.join(params.Control.checkpoint_dir, 'pre_run_' + params.Control.fname_app + '.out')
        # Log File name
        self.f_log_name = os.path.join(params.Control.checkpoint_dir, "log_" + params.Control.fname_app + ".out")
        # Save it in params too
        params.Control.log_file = self.f_log_name

        # Pre run testing: assign self.io_file to the correct file to open/write
        self.io_file = self.f_pre_run if params.Control.pre_run else self.f_log_name

        # Print figlet to file if not a restart run
        if not params.load_method == "restart":
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
        params : class
            Simulation parameters to print.
        """

        screen = sys.stdout
        f_log = open(self.io_file, 'a+')
        repeat = 2 if self.verbose else 1

        # redirect printing to file
        sys.stdout = f_log

        # Print to file first then to screen if repeat == 2
        while repeat > 0:

            if params.load_method == 'restart':
                print('\n\n--------------------------- Restart -------------------------------------')
                print("Restart step: {}".format(params.load_restart_step))
                print("Total production steps: {}".format(params.Control.Nsteps))

            else:
                # Choose the correct heading
                if self.pre_run:
                    print('\n\n-------------- Pre Run Details ----------------------')
                else:
                    print('\n\n----------------- Simulation -----------------------')

                print('\nJob ID: ', params.Control.fname_app)
                print('Job directory: ', params.Control.checkpoint_dir)
                print('Dump directory: ', params.Control.dump_dir)
                print('\nUnits: ', params.Control.units)
                print('Total No. of particles = ', params.total_num_ptcls)
                print('No. of species = ', len(params.species))
                for isp, sp in enumerate(params.species):
                    print("Species {} : {}".format(isp + 1, sp.name))
                    print("\tSpecies ID: {}".format(isp))
                    print("\tNo. of particles = {} ".format(sp.num))
                    print("\tNumber density = {:2.6e} ".format(sp.num_density), end='')
                    print("[N/cc]" if params.Control.units == "cgs" else "[N/m^3]")
                    print("\tMass = {:2.6e} ".format(sp.mass), end='')
                    print("[g]" if params.Control.units == "cgs" else "[kg]")
                    print('\tTemperature = {:2.6e} [K]'.format(sp.temperature))

                print('\nLengths scales:')
                print('Wigner-Seitz radius = {:2.6e} '.format(params.aws), end='')
                print("[cm]" if params.Control.units == "cgs" else "[m]")
                print('No. of non-zero box dimensions = ', int(params.dimensions))
                print('Box length along x axis = {:2.6e} a_ws = {:2.6e} '.format(params.Lv[0] / params.aws,
                                                                                 params.Lv[0]), end='')
                print("[cm]" if params.Control.units == "cgs" else "[m]")
                print('Box length along y axis = {:2.6e} a_ws = {:2.6e} '.format(params.Lv[1] / params.aws,
                                                                                 params.Lv[1]), end='')
                print("[cm]" if params.Control.units == "cgs" else "[m]")
                print('Box length along z axis = {:2.6e} a_ws = {:2.6e} '.format(params.Lv[2] / params.aws,
                                                                                 params.Lv[2]), end='')
                print("[cm]" if params.Control.units == "cgs" else "[m]")
                print("The remaining lengths scales are given in ", end='')
                print("[cm]" if params.Control.units == "cgs" else "[m]")

                print('\nBoundary conditions:')
                if params.BC.pbc_axes:
                    print('Periodic BC along axes : ', params.BC.pbc_axes)
                if params.BC.mm_axes:
                    print('Momentum Mirror BC along axes : ', params.BC.mm_axes)
                if params.BC.open_axes:
                    print('Open BC along axes : ', params.BC.open_axes)

                if params.Langevin.on:
                    print('Langevin model : ', params.Langevin.type)

                print("\nIntegrator: ", params.Integrator.type)
                print("\nThermostat: ", params.Thermostat.type)
                print("Berendsen Relaxation rate: {:1.3f}".format(1.0 / params.Thermostat.tau))
                # print("Thermostating Temperatures: ", params.Thermostat.temperatures)

                print('\nPotential: ', params.Potential.type)
                if params.Potential.type == 'Yukawa':
                    print('kappa = {:1.4e}'.format(params.Potential.matrix[1, 0, 0] * params.aws))
                    print('lambda_TF = {:1.4e}'.format(params.lambda_TF))
                    print('Gamma_eff = {:4.2f}'.format(params.Potential.Gamma_eff))

                elif params.Potential.type == 'EGS':
                    print('lambda_TF = {:1.4e}'.format(params.lambda_TF))
                    print('kappa = {:1.4e}'.format(params.Potential.matrix[1, 0, 0] * params.aws))
                    print('nu = {:1.4e}'.format(params.Potential.nu))
                    if params.Potential.nu < 1:
                        print('Exponential decay:')
                        print('lambda_p = {:1.4e}'.format(params.Potential.lambda_p))
                        print('lambda_m = {:1.4e}'.format(params.Potential.lambda_m))
                        print('alpha = {:1.4e}'.format(params.Potential.alpha))
                        print('Theta = {:1.4e}'.format(params.Potential.theta))
                        print('b = {:1.4e}'.format(params.Potential.b))

                    else:
                        print('Oscillatory potential:')
                        print('gamma_p = {:1.4e}'.format(params.Potential.gamma_p))
                        print('gamma_m = {:1.4e}'.format(params.Potential.gamma_m))
                        print('alpha = {:1.4e}'.format(params.Potential.alphap))
                        print('Theta = {:1.4e}'.format(params.Potential.theta))
                        print('b = {:1.4e}'.format(params.Potential.b))

                elif params.Potential.type == 'Coulomb':
                    print('Gamma_eff = {:4.2f}'.format(params.Potential.Gamma_eff))

                elif params.Potential.type == 'LJ':
                    print('epsilon = {:2.6e}'.format(params.Potential.matrix[0, 0, 0]))
                    print('sigma = {:2.6e}'.format(params.Potential.matrix[1, 0, 0]))

                elif params.Potential.type == "QSP":
                    print("e de Broglie wavelength = {:2.4f} ai = {:2.6e} ".format(
                        2.0 * np.pi / params.Potential.matrix[1, 0, 0] / (np.sqrt(2.0) * params.ai),
                        2.0 * np.pi / params.Potential.matrix[1, 0, 0] / np.sqrt(2.0)), end='')
                    print("[cm]" if params.Control.units == "cgs" else "[m]")
                    print("e-e screening parameter = {:2.4f}".format(params.Potential.matrix[1, 0, 0] * params.aws))
                    print("ion de Broglie wavelength  = {:2.4f} ai = {:2.6e} ".format(
                        2.0 * np.pi / params.Potential.matrix[1, 0, 0] / (np.sqrt(2.0) * params.ai),
                        2.0 * np.pi / params.Potential.matrix[1, 0, 0] / np.sqrt(2.0)), end='')
                    print("[cm]" if params.Control.units == "cgs" else "[m]")
                    print("i-i screening parameter = {:2.4f}".format(params.Potential.matrix[1, 1, 1] * params.aws))
                    print("e-i screening parameter = {:2.4f}".format(params.Potential.matrix[1, 0, 1] * params.aws))
                    print("e-i Coupling Parameter = {:3.3f} ".format(params.Potential.Gamma_eff))
                    print("rs Coupling Parameter = {:3.3f} ".format(params.rs))

                if params.Magnetic.on:
                    print('\nMagnetized Plasma:')
                    for ic in range(params.num_species):
                        print('Cyclotron frequency of Species {:2} = {:2.6e}'.format(ic + 1,
                                                                                     params.species[ic].omega_c))
                        print('beta_c of Species {:2} = {:2.6e}'.format(ic + 1,
                                                                        params.species[ic].omega_c / params.species[
                                                                            ic].wp))

                print("\nAlgorithm : ", params.Potential.method)
                if params.Potential.method == 'P3M':
                    print('Ewald parameter alpha = {:2.4f} / a_ws = {:1.6e} '.format(params.P3M.G_ew * params.aws,
                                                                                     params.P3M.G_ew), end='')
                    print("[1/cm]" if params.Control.units == "cgs" else "[1/m]")
                    print('Mesh size * Ewald_parameter (h * alpha) = {:2.4f} ~ 1/{} '.format(
                        params.P3M.hx * params.P3M.G_ew, int(1. / (params.P3M.hx * params.P3M.G_ew))), )
                    print(
                        'rcut = {:2.4f} a_ws = {:2.6e} '.format(params.Potential.rc / params.aws, params.Potential.rc),
                        end='')
                    print("[cm]" if params.Control.units == "cgs" else "[m]")
                    print('Mesh = ', params.P3M.MGrid)
                    print('No. of PP cells per dimension = {:2}, {:2}, {:2}'.format(
                        int(params.Lv[0] / params.Potential.rc),
                        int(params.Lv[1] / params.Potential.rc),
                        int(params.Lv[2] / params.Potential.rc)))
                    print('No. of PP neighbors per particle = {:6}'.format(
                        int(params.total_num_ptcls * 4.0 / 3.0 * np.pi * (
                                params.Potential.rc / params.Lv.min()) ** 3.0)))
                    print('PM Force Error = {:2.6e}'.format(params.P3M.PM_err))
                    print('PP Force Error = {:2.6e}'.format(params.P3M.PP_err))
                    print('Tot Force Error = {:2.6e}'.format(params.P3M.F_err))
                elif params.Potential.method == 'PP':
                    print('rcut/a_ws = {:2.6e}'.format(params.Potential.rc / params.aws))
                    print(
                        'No. of cells per dimension = {:2}, {:2}, {:2}'.format(int(params.Lv[0] / params.Potential.rc),
                                                                               int(params.Lv[1] / params.Potential.rc),
                                                                               int(params.Lv[2] / params.Potential.rc)))
                    print('No. of neighbors per particle = {:4}'.format(
                        int(params.total_num_ptcls * 4.0 / 3.0 * np.pi * (
                                params.Potential.rc / params.Lv.min()) ** 3.0)))
                    print('PP Force Error = {:2.6e}'.format(params.PP_err))

                print("\nTime scales:")
                print('Time step = {:2.6e} [s]'.format(params.Control.dt))
                if params.Potential.type == 'Yukawa' or params.Potential.type == 'EGS':
                    print('(total) ion plasma frequency = {:1.6e} [Hz]'.format(params.wp))
                    print('wp dt = {:2.4f}'.format(params.Control.dt * params.wp))
                elif params.Potential.type == 'Coulomb' or params.Potential.type == 'Moliere':
                    print('(total) plasma frequency = {:1.6e} [Hz]'.format(params.wp))
                    print('wp dt = {:2.4f}'.format(params.Control.dt * params.wp))
                elif params.Potential.type == 'QSP':
                    print('e plasma frequency = {:2.6e} [Hz]'.format(params.species[0].wp))
                    print('ion plasma frequency = {:2.6e} [Hz]'.format(params.species[1].wp))
                    print('w_pe dt = {:2.4f}'.format(params.Control.dt * params.species[0].wp))
                elif params.Potential.type == 'LJ':
                    print('(total) equivalent plasma frequency = {:1.6e} [Hz]'.format(params.wp))
                    print('wp dt = {:2.4f}'.format(params.Control.dt * params.wp))

                print('No. of equilibration steps = {} ~ {} wp T_eq'.format(
                    params.Control.Neq, int(params.Control.Neq * params.wp * params.Control.dt)))
                print('No. of post-equilibration steps = {} ~ {} wp T_prod'.format(
                    params.Control.Nsteps, int(params.Control.Nsteps * params.wp * params.Control.dt)))
                print('snapshot interval = {} = {:1.3f} wp T_snap'.format(
                    params.Control.dump_step, params.Control.dump_step * params.Control.dt * params.wp))

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
