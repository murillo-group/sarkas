import numpy as np


class Verbose:
    """ 
    Class to handle verbose output to screen.

    Parameters
    ----------
    params : class
        Simulation's parameters

    """

    def __init__(self, params):

        f_log_name = params.Control.checkpoint_dir + "/" + "log_" + params.Control.fname_app + ".out"
        params.Control.log_file = f_log_name
        f_log = open(f_log_name, "w+")
        print("Sarkas Ver. 1.0", file=f_log)
        f_log.close()
        self.params = params

    def sim_setting_summary(self):
        """
        Print out to file a summary of simulation's parameters.
        """
        params = self.params
        f_log = open(params.Control.log_file, 'a+')

        print('\n\n----------- Molecular Dynamics Simulation ----------------------', file=f_log)
        print('No. of particles = ', params.total_num_ptcls, file=f_log)
        print('No. of species = ', len(params.species), file=f_log)
        print('units: ', params.Control.units, file=f_log)
        print('Temperature = {:2.6e} [K]'.format(params.T_desired), file=f_log)

        print('\nNo. of non-zero box dimensions = ', int(params.dimensions), file=f_log)
        print('Wigner-Seitz radius = {:2.6e}'.format(params.aws), file=f_log)
        print('Box length along x axis = {:2.6e} = {:2.6e} a_ws'.format(params.Lv[0], params.Lv[0] / params.aws),
              file=f_log)
        print('Box length along y axis = {:2.6e} = {:2.6e} a_ws'.format(params.Lv[1], params.Lv[1] / params.aws),
              file=f_log)
        print('Box length along z axis = {:2.6e} = {:2.6e} a_ws'.format(params.Lv[2], params.Lv[2] / params.aws),
              file=f_log)

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
                print('Exponential decay', file=f_log)
                print('lambda_p = {:1.4e}'.format(params.Potential.lambda_p), file=f_log)
                print('lambda_m = {:1.4e}'.format(params.Potential.lambda_m), file=f_log)
                print('alpha = {:1.4e}'.format(params.Potential.alpha), file=f_log)
                print('Theta = {:1.4e}'.format(params.Potential.theta), file=f_log)
                print('b = {:1.4e}'.format(params.Potential.b), file=f_log)

            else:
                print('Oscillatory potential', file=f_log)
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
            print("e de Broglie wavelength = {:2.6e} ".format(params.Potential.matrix[0, 0, 0] / np.sqrt(2.0)),
                  file=f_log)
            print("e de Broglie wavelength / a_ws = {:2.6e} ".format(
                params.Potential.matrix[0, 0, 0] / (np.sqrt(2.0) * params.ai)), file=f_log)
            print("ion de Broglie wavelength = {:2.6e} ".format(params.Potential.matrix[0, 1, 1] / np.sqrt(2.0)),
                  file=f_log)
            print("ion de Broglie wavelength / a_ws = {:2.6e} ".format(
                params.Potential.matrix[0, 1, 1] / np.sqrt(2.0) / params.ai), file=f_log)
            print("e-i Coupling Parameter = {:3.3f} ".format(params.Potential.Gamma_eff), file=f_log)
            print("rs Coupling Parameter = {:3.3f} ".format(params.rs), file=f_log)

        if params.Magnetic.on:
            print('\nMagnetized Plasma', file=f_log)
            for ic in range(params.num_species):
                print('Cyclotron frequency of Species {:2} = {:2.6e}'.format(ic + 1, params.species[ic].omega_c),
                      file=f_log)
                print('beta_c of Species {:2} = {:2.6e}'.format(ic + 1,
                                                                params.species[ic].omega_c / params.species[ic].wp),
                      file=f_log)

        print("\nAlgorithm = ", params.Potential.method, file=f_log)
        if params.Potential.method == 'P3M':
            print('Ewald parameter alpha = {:1.6e}'.format(params.P3M.G_ew), file=f_log)
            print('alpha * a_ws = {:2.6e}'.format(params.Potential.matrix[-1, 0, 0] * params.aws), file=f_log)
            print('Grid_size * Ewald_parameter (h * alpha) = {:2.6e}'.format(params.P3M.hx * params.P3M.G_ew),
                  file=f_log)
            print('rcut/a_ws = {:2.6e}'.format(params.Potential.rc / params.aws), file=f_log)
            print('Mesh = ', params.P3M.MGrid, file=f_log)
            print('PM Force Error = {:2.6e}'.format(params.P3M.PM_err), file=f_log)
            print('No. of cells per dimension = {:2}, {:2}, {:2}'.format(int(params.Lv[0] / params.Potential.rc),
                                                                         int(params.Lv[1] / params.Potential.rc),
                                                                         int(params.Lv[2] / params.Potential.rc)),
                  file=f_log)
            print('No. of neighbors per particle = {:6}'.format(
                int(params.total_num_ptcls * 4.0 / 3.0 * np.pi * (params.Potential.rc / params.Lv.min()) ** 3.0)),
                file=f_log)
            print('PP Force Error = {:2.6e}'.format(params.P3M.PP_err), file=f_log)
            print('Tot Force Error = {:2.6e}'.format(params.P3M.F_err), file=f_log)
        else:
            print('rcut/a_ws = {:2.6e}'.format(params.Potential.rc / params.aws), file=f_log)
            print('No. of cells per dimension = {:2}, {:2}, {:2}'.format(int(params.Lv[0] / params.Potential.rc),
                                                                         int(params.Lv[1] / params.Potential.rc),
                                                                         int(params.Lv[2] / params.Potential.rc)),
                  file=f_log)
            print('No. of neighbors per particle = {:4}'.format(int(params.total_num_ptcls * 4.0 / 3.0 * np.pi
                                                                    * (params.Potential.rc / params.Lv.min()) ** 3.0)),
                file=f_log)
            print('PP Force Error = {:2.6e}'.format(params.PP_err), file=f_log)

        print('\ntime step = {:2.6e} [s]'.format(params.Control.dt), file=f_log)
        if params.Potential.type == 'Yukawa' or params.Potential.type == 'EGS':
            print('(total) ion plasma frequency = {:1.6e} [Hz]'.format(params.wp), file=f_log)
            print('wp dt = {:2.4f}'.format(params.Control.dt * params.wp), file=f_log)
        elif params.Potential.type == 'Coulomb':
            print('(total) plasma frequency = {:1.6e} [Hz]'.format(params.wp), file=f_log)
            print('wp dt = {:2.4f}'.format(params.Control.dt * params.wp), file=f_log)
        elif params.Potential.type == 'QSP':
            print('e plasma frequency = {:2.6e} [Hz]'.format(params.species[0].wp), file=f_log)
            print('ion plasma frequency = {:2.6e} [Hz]'.format(params.species[1].wp), file=f_log)
            print('w_pe dt = {:2.4f}'.format(params.Control.dt * params.species[0].wp), file=f_log)
        elif params.Potential.type == 'LJ':
            print('(total) equivalent plasma frequency = {:1.6e} [Hz]'.format(params.wp), file=f_log)
            print('wp dt = {:2.4f}'.format(params.Control.dt * params.wp), file=f_log)

        print('\nNo. of equilibration steps = ', params.Control.Neq, file=f_log)
        print('No. of post-equilibration steps = ', params.Control.Nsteps, file=f_log)
        print('snapshot interval = ', params.Control.dump_step, file=f_log)
        print('Periodic boundary condition {1=yes, 0=no} =', params.Control.PBC, file=f_log)

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
        f_log = open(self.params.Control.log_file, "a+")
        t_hrs = int(t / 3600)
        t_min = int((t - t_hrs * 3600) / 60)
        t_sec = int((t - t_hrs * 3600 - t_min * 60))
        print('\n{} Time = {} hrs {} mins {} secs'.format(time_stamp, t_hrs, t_min, t_sec), file=f_log)
        f_log.close()
