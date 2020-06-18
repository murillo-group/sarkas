import os
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

BG_COLORS = ['24;69;49',
             '0;129;131',
             '83;80;84',
             '110;0;95'
             ]


def screen_figlet():
    """
    Print a colored figlet of Sarkas to screen.
    """
    fg = FG_COLORS[np.random.randint(0, len(FG_COLORS))]
    bg = BG_COLORS[np.random.randint(0, len(BG_COLORS))]
    fnt = FONTS[np.random.randint(0, len(FONTS))]
    clr = fg  # + ':' + bg
    print_figlet('\n\tSarkas\n', font=fnt, colors=clr)

    print("\nAn open-source pure-python molecular dynamics code for non-ideal plasmas.")
    print('\n\n--------------------------- Simulation -------------------------------------')
    print('\nInput file read.')
    print('\nParams Class created.')
    print('\nLog file created.')


class Verbose:
    """ 
    Class to handle verbose output to screen.

    Parameters
    ----------
    params : class
        Simulation's parameters

    """

    def __init__(self, params):

        self.f_log_name = os.path.join(params.Control.checkpoint_dir, "log_" + params.Control.fname_app + ".out")
        params.Control.log_file = self.f_log_name
        if params.load_method == "restart":
            f_log = open(self.f_log_name, "a+")
            print('\n\n--------------------------- Restart -------------------------------------', file=f_log)
            print("Restart step: {}".format(params.load_restart_step), file=f_log)
            print("Total production steps: {}".format(params.Control.Nsteps), file=f_log)
        else:
            f_log = open(self.f_log_name, "w+")
            figlet_obj = Figlet(font='starwars')
            print(figlet_obj.renderText('Sarkas'), file=f_log)
            print("An open-source pure-python molecular dynamics code for non-ideal plasmas.", file=f_log)

        f_log.close()

        if params.Control.verbose:
            screen_figlet()

    def sim_setting_summary(self, params):
        """
        Print out to file a summary of simulation's parameters.
        """
        f_log = open(self.f_log_name, 'a+')

        print('\n\n-------------- Simulation ----------------------', file=f_log)
        print('\nJob ID: ', params.Control.fname_app, file=f_log)
        print('Job directory: ', params.Control.checkpoint_dir, file=f_log)
        print('Dump directory: ', params.Control.dump_dir, file=f_log)
        print('\nUnits: ', params.Control.units, file=f_log)
        print('Total No. of particles = ', params.total_num_ptcls, file=f_log)
        print('No. of species = ', len(params.species), file=f_log)
        for sp in range(params.num_species):
            print("Species {} : {}".format(sp + 1, params.species[sp].name), file=f_log)
            print("\tSpecies ID: {}".format(sp), file=f_log)
            print("\tNo. of particles = {} ".format(params.species[sp].num), file=f_log)
            print("\tNumber density = {:2.6e} ".format(params.species[sp].num_density), end='', file=f_log)
            print("[N/cc]" if params.Control.units == "cgs" else "[N/m^3]", file=f_log)
            print("\tMass = {:2.6e} ".format(params.species[sp].mass), end='', file=f_log)
            print("[g]" if params.Control.units == "cgs" else "[kg]", file=f_log)
            print('\tTemperature = {:2.6e} [K]'.format(params.T_desired), file=f_log)

        print('\nLengths scales:', file=f_log)
        print('Wigner-Seitz radius = {:2.6e} '.format(params.aws), end='', file=f_log)
        print("[cm]" if params.Control.units == "cgs" else "[m]", file=f_log)
        print('No. of non-zero box dimensions = ', int(params.dimensions), file=f_log)
        print('Box length along x axis = {:2.6e} a_ws = {:2.6e} '.format(params.Lv[0] / params.aws, params.Lv[0]),
              end='', file=f_log)
        print("[cm]" if params.Control.units == "cgs" else "[m]", file=f_log)
        print('Box length along x axis = {:2.6e} a_ws = {:2.6e} '.format(params.Lv[1] / params.aws, params.Lv[1]),
              end='', file=f_log)
        print("[cm]" if params.Control.units == "cgs" else "[m]", file=f_log)
        print('Box length along x axis = {:2.6e} a_ws = {:2.6e} '.format(params.Lv[2] / params.aws, params.Lv[2]),
              end='', file=f_log)
        print("[cm]" if params.Control.units == "cgs" else "[m]", file=f_log)
        print("The remaining lengths scales are given in ", end='', file=f_log)
        print("[cm]" if params.Control.units == "cgs" else "[m]", file=f_log)

        print("\nIntegrator: ", params.Integrator.type, file=f_log)
        print("\nThermostat: ", params.Thermostat.type, file=f_log)

        print('\nPotential: ', params.Potential.type, file=f_log)
        if params.Potential.type == 'Yukawa':
            print('kappa = {:1.4e}'.format(params.Potential.matrix[0, 0, 0] * params.aws), file=f_log)
            print('lambda_TF = {:1.4e}'.format(params.lambda_TF), file=f_log)
            if len(params.species) > 1:
                print('Gamma_eff = {:4.2f}'.format(params.Potential.Gamma_eff), file=f_log)
            else:
                print('Gamma = {:4.2e}'.format(params.Potential.matrix[1, 0, 0]), file=f_log)
        elif params.Potential.type == 'EGS':
            print('lambda_TF = {:1.4e}'.format(params.lambda_TF), file=f_log)
            print('kappa = {:1.4e}'.format(params.Potential.matrix[0, 0, 0] * params.aws), file=f_log)
            print('nu = {:1.4e}'.format(params.Potential.nu), file=f_log)
            if params.Potential.nu < 1:
                print('Exponential decay:', file=f_log)
                print('lambda_p = {:1.4e}'.format(params.Potential.lambda_p), file=f_log)
                print('lambda_m = {:1.4e}'.format(params.Potential.lambda_m), file=f_log)
                print('alpha = {:1.4e}'.format(params.Potential.alpha), file=f_log)
                print('Theta = {:1.4e}'.format(params.Potential.theta), file=f_log)
                print('b = {:1.4e}'.format(params.Potential.b), file=f_log)

            else:
                print('Oscillatory potential:', file=f_log)
                print('gamma_p = {:1.4e}'.format(params.Potential.gamma_p), file=f_log)
                print('gamma_m = {:1.4e}'.format(params.Potential.gamma_m), file=f_log)
                print('alpha = {:1.4e}'.format(params.Potential.alphap), file=f_log)
                print('Theta = {:1.4e}'.format(params.Potential.theta), file=f_log)
                print('b = {:1.4e}'.format(params.Potential.b), file=f_log)

        elif params.Potential.type == 'Coulomb':
            if len(params.species) > 1:
                print('Gamma_eff = {:4.2f}'.format(params.Potential.Gamma_eff), file=f_log)
            else:
                print('Gamma = {:4.2e}'.format(params.Potential.matrix[0, 0, 0]), file=f_log)

        elif params.Potential.type == 'LJ':
            print('epsilon = {:2.6e}'.format(params.Potential.matrix[0, 0, 0]), file=f_log)
            print('sigma = {:2.6e}'.format(params.Potential.matrix[1, 0, 0]), file=f_log)

        elif params.Potential.type == "QSP":
            print("e de Broglie wavelength = {:2.4f} ai = {:2.6e} ".format(
                params.Potential.matrix[0, 0, 0] / np.sqrt(2.0) / params.ai,
                params.Potential.matrix[0, 0, 0] / np.sqrt(2.0)), end='', file=f_log)
            print("[cm]" if params.Control.units == "cgs" else "[m]", file=f_log)
            print("e-e screening parameter = {:2.4f}".format(params.Potential.matrix[2, 0, 0]*params.aws), file=f_log)
            print("ion de Broglie wavelength  = {:2.4f} ai = {:2.6e} ".format(
                params.Potential.matrix[0, 1, 1] / np.sqrt(2.0) / params.ai,
                params.Potential.matrix[0, 1, 1] / np.sqrt(2.0)), end='', file=f_log)
            print("[cm]" if params.Control.units == "cgs" else "[m]", file=f_log)
            print("i-i screening parameter = {:2.4f}".format(params.Potential.matrix[2, 1, 1]*params.aws), file=f_log)
            print("e-i screening parameter = {:2.4f}".format(params.Potential.matrix[2, 0, 1]*params.aws), file=f_log)
            print("e-i Coupling Parameter = {:3.3f} ".format(params.Potential.Gamma_eff), file=f_log)
            print("rs Coupling Parameter = {:3.3f} ".format(params.rs), file=f_log)

        if params.Magnetic.on:
            print('\nMagnetized Plasma:', file=f_log)
            for ic in range(params.num_species):
                print('Cyclotron frequency of Species {:2} = {:2.6e}'.format(ic + 1, params.species[ic].omega_c),
                      file=f_log)
                print('beta_c of Species {:2} = {:2.6e}'.format(ic + 1,
                                                                params.species[ic].omega_c / params.species[ic].wp),
                      file=f_log)

        print("\nAlgorithm : ", params.Potential.method, file=f_log)
        if params.Potential.method == 'P3M':
            print('Ewald parameter alpha = {:1.6e} '.format(params.P3M.G_ew), end='', file=f_log)
            print("[1/cm]" if params.Control.units == "cgs" else "[1/m]", file=f_log)
            print('Grid_size * Ewald_parameter (h * alpha) = {:2.6e}'.format(params.P3M.hx * params.P3M.G_ew),
                  file=f_log)
            print('alpha * a_ws = {:2.6e}'.format(params.Potential.matrix[-1, 0, 0] * params.aws), file=f_log)

            print('rcut/a_ws = {:2.6f}'.format(params.Potential.rc / params.aws), file=f_log)
            print('Mesh = ', params.P3M.MGrid, file=f_log)
            print('No. of PP cells per dimension = {:2}, {:2}, {:2}'.format(int(params.Lv[0] / params.Potential.rc),
                                                                            int(params.Lv[1] / params.Potential.rc),
                                                                            int(params.Lv[2] / params.Potential.rc)),
                  file=f_log)
            print('No. of PP neighbors per particle = {:6}'.format(
                int(params.total_num_ptcls * 4.0 / 3.0 * np.pi * (params.Potential.rc / params.Lv.min()) ** 3.0)),
                file=f_log)
            print('PM Force Error = {:2.6e}'.format(params.P3M.PM_err), file=f_log)
            print('PP Force Error = {:2.6e}'.format(params.P3M.PP_err), file=f_log)
            print('Tot Force Error = {:2.6e}'.format(params.P3M.F_err), file=f_log)
        elif params.Potential.method == 'PP':
            print('rcut/a_ws = {:2.6e}'.format(params.Potential.rc / params.aws), file=f_log)
            print('No. of cells per dimension = {:2}, {:2}, {:2}'.format(int(params.Lv[0] / params.Potential.rc),
                                                                         int(params.Lv[1] / params.Potential.rc),
                                                                         int(params.Lv[2] / params.Potential.rc)),
                  file=f_log)
            print('No. of neighbors per particle = {:4}'.format(int(params.total_num_ptcls * 4.0 / 3.0 * np.pi
                                                                    * (params.Potential.rc / params.Lv.min()) ** 3.0)),
                  file=f_log)
            print('PP Force Error = {:2.6e}'.format(params.PP_err), file=f_log)

        print("\nTime scales:", file=f_log)
        print('Time step = {:2.6e} [s]'.format(params.Control.dt), file=f_log)
        if params.Potential.type == 'Yukawa' or params.Potential.type == 'EGS':
            print('(total) ion plasma frequency = {:1.6e} [Hz]'.format(params.wp), file=f_log)
            print('wp dt = {:2.4f}'.format(params.Control.dt * params.wp), file=f_log)
        elif params.Potential.type == 'Coulomb' or params.Potential.type == 'Moliere':
            print('(total) plasma frequency = {:1.6e} [Hz]'.format(params.wp), file=f_log)
            print('wp dt = {:2.4f}'.format(params.Control.dt * params.wp), file=f_log)
        elif params.Potential.type == 'QSP':
            print('e plasma frequency = {:2.6e} [Hz]'.format(params.species[0].wp), file=f_log)
            print('ion plasma frequency = {:2.6e} [Hz]'.format(params.species[1].wp), file=f_log)
            print('w_pe dt = {:2.4f}'.format(params.Control.dt * params.species[0].wp), file=f_log)
        elif params.Potential.type == 'LJ':
            print('(total) equivalent plasma frequency = {:1.6e} [Hz]'.format(params.wp), file=f_log)
            print('wp dt = {:2.4f}'.format(params.Control.dt * params.wp), file=f_log)

        print('No. of equilibration steps = ', params.Control.Neq, file=f_log)
        print('No. of post-equilibration steps = ', params.Control.Nsteps, file=f_log)
        print('snapshot interval = ', params.Control.dump_step, file=f_log)
        print('\nBoundary conditions:', file=f_log)
        if params.BC.pbc_axes:
            print('Periodic BC along axes : ', params.BC.pbc_axes, file=f_log)
        if params.BC.mm_axes:
            print('Momentum Mirror BC along axes : ', params.BC.mm_axes, file=f_log)
        if params.BC.open_axes:
            print('Open BC along axes : ', params.BC.open_axes, file=f_log)

        if params.Langevin.on:
            print('Langevin model = ', params.Langevin.type, file=f_log)

        f_log.close()

    def time_stamp(self, time_stamp, t):
        """
        Print out to screen elapsed times.

        Parameters
        ----------
        time_stamp : array
            Array of time stamps.

        t : float
            Elapsed time.
        """
        f_log = open(self.f_log_name, "a+")
        t_hrs = int(t / 3600)
        t_min = int((t - t_hrs * 3600) / 60)
        t_sec = int((t - t_hrs * 3600 - t_min * 60))
        print('\n{} Time = {} hrs {} mins {} secs'.format(time_stamp, t_hrs, t_min, t_sec), file=f_log)
        f_log.close()
