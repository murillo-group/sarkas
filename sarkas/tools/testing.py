"""
Module for testing simulation parameters
"""
# Python modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
from optparse import OptionParser
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Sarkas modules
from sarkas.io.verbose import Verbose
from sarkas.simulation.params import Params
from sarkas.simulation.particles import Particles
from sarkas.tools import force_error

#
# plt.style.use(os.path.join(os.path.join(os.getcwd(), 'src'), 'MSUstyle.mplstyle'))


def quadratic(x, a, b, c):
    """
    Quadratic function for fitting.

    Parameters
    ----------
    x : array
        Values at which to calculate the function.

    a: float
        Intercept.

    b: float
        Coefficient of linear term.

    c: float
        Coefficient of quadratic term.

    Returns
    -------
    quadratic formula
    """
    return a + b * x + c * x * x


def linear(x, a):
    """
    Linear function for fitting.

    Parameters
    ----------
    x : array
        Values at which to calculate the function.

    a: float
        Coefficient of linear term.

    Returns
    -------
    linear formula
    """
    return a * x


def make_line_plot(rcuts, alphas, chosen_alpha, chosen_rcut, DeltaF_tot, params):
    """
    Plot selected values of the total force error approximation.

    Parameters
    ----------
    rcuts: array
        Cut off distances.
    alphas: array
        Ewald parameters.

    chosen_alpha: float
        Chosen Ewald parameter.

    chosen_rcut: float
        Chosen cut off radius.

    DeltaF_tot: ndarray
        Force error matrix.

    params: class
        Simulation's parameters.

    """
    # Plot the calculate Force error
    kappa_title = 0.0 if params.Potential.type == "Coulomb" else params.Potential.matrix[1, 0, 0]

    # Plot the results
    fig_path = params.Control.pre_run_dir

    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 7))
    ax[0].plot(rcuts, DeltaF_tot[30, :], label=r'$\alpha a_{ws} = ' + '{:2.2f}$'.format(alphas[30]))
    ax[0].plot(rcuts, DeltaF_tot[40, :], label=r'$\alpha a_{ws} = ' + '{:2.2f}$'.format(alphas[40]))
    ax[0].plot(rcuts, DeltaF_tot[50, :], label=r'$\alpha a_{ws} = ' + '{:2.2f}$'.format(alphas[50]))
    ax[0].plot(rcuts, DeltaF_tot[60, :], label=r'$\alpha a_{ws} = ' + '{:2.2f}$'.format(alphas[60]))
    ax[0].plot(rcuts, DeltaF_tot[70, :], label=r'$\alpha a_{ws} = ' + '{:2.2f}$'.format(alphas[70]))
    ax[0].set_ylabel(r'$\Delta F^{approx}_{tot}$')
    ax[0].set_xlabel(r'$r_c/a_{ws}$')
    ax[0].set_yscale('log')
    ax[0].axvline(chosen_rcut, ls='--', c='k')
    ax[0].axhline(params.P3M.F_err, ls='--', c='k')
    if rcuts[-1] * params.aws > 0.5 * params.Lv.min():
        ax[0].axvline(0.5 * params.Lv.min() / params.aws, c='r', label=r'$L/2$')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc='best')

    ax[1].plot(alphas, DeltaF_tot[:, 30], label=r'$r_c = {:2.2f}'.format(rcuts[30]) + ' a_{ws}$')
    ax[1].plot(alphas, DeltaF_tot[:, 40], label=r'$r_c = {:2.2f}'.format(rcuts[40]) + ' a_{ws}$')
    ax[1].plot(alphas, DeltaF_tot[:, 50], label=r'$r_c = {:2.2f}'.format(rcuts[50]) + ' a_{ws}$')
    ax[1].plot(alphas, DeltaF_tot[:, 60], label=r'$r_c = {:2.2f}'.format(rcuts[60]) + ' a_{ws}$')
    ax[1].plot(alphas, DeltaF_tot[:, 70], label=r'$r_c = {:2.2f}'.format(rcuts[70]) + ' a_{ws}$')
    ax[1].set_xlabel(r'$\alpha \; a_{ws}$')
    ax[1].set_yscale('log')
    ax[1].axhline(params.P3M.F_err, ls='--', c='k')
    ax[1].axvline(chosen_alpha, ls='--', c='k')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc='best')
    fig.suptitle(
        r'Approximate Total Force error  $N = {}, \quad M = {}, \quad \kappa = {:1.2f}$'.format(
            params.total_num_ptcls,
            params.P3M.MGrid[0],
            kappa_title * params.aws))
    fig.savefig(os.path.join(fig_path, 'ForceError_LinePlot_' + params.Control.fname_app + '.png'))
    fig.show()


def make_color_map(rcuts, alphas, chosen_alpha, chosen_rcut, DeltaF_tot, params):
    """
    Plot a color map of the total force error approximation.

    Parameters
    ----------
    rcuts: array
        Cut off distances.

    alphas: array
        Ewald parameters.

    chosen_alpha: float
        Chosen Ewald parameter.

    chosen_rcut: float
        Chosen cut off radius.

    DeltaF_tot: ndarray
        Force error matrix.

    params: class
        Simulation's parameters.

    """
    # Plot the calculate Force error
    kappa_title = 0.0 if params.Potential.type == "Coulomb" else params.Potential.matrix[1, 0, 0]

    # Plot the results
    fig_path = params.Control.pre_run_dir

    r_mesh, a_mesh = np.meshgrid(rcuts, alphas)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    if DeltaF_tot.min() == 0.0:
        minv = 1e-120
    else:
        minv = DeltaF_tot.min()
    CS = ax.contourf(a_mesh, r_mesh, DeltaF_tot, norm=LogNorm(vmin=minv, vmax=DeltaF_tot.max()))
    CS2 = ax.contour(CS, colors='w')
    ax.clabel(CS2, fmt='%1.0e', colors='w')
    ax.scatter(chosen_alpha, chosen_rcut, s=200, c='k')
    if rcuts[-1] * params.aws > 0.5 * params.Lv.min():
        ax.axhline(0.5 * params.Lv.min() / params.aws, c='r', label=r'$L/2$')
    # ax.tick_params(labelsize=fsz)
    ax.set_xlabel(r'$\alpha \;a_{ws}$')
    ax.set_ylabel(r'$r_c/a_{ws}$')
    ax.set_title(
        r'$\Delta F^{approx}_{tot}(r_c,\alpha)$' + r'  for  $N = {}, \quad M = {}, \quad \kappa = {:1.2f}$'.format(
            params.total_num_ptcls, params.P3M.MGrid[0], kappa_title * params.aws))
    fig.colorbar(CS)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, 'ForceError_ClrMap_' + params.Control.fname_app + '.png'))
    fig.show()


def make_fit_plot(pp_xdata, pm_xdata, pp_times, pm_times, pp_opt, pm_opt, pp_xlabels, pm_xlabels, fig_path):
    """
    Make a dual plot of the fitted functions.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    ax[0].plot(pm_xdata, pm_times.mean(axis=-1), 'o', label='Measured times')
    ax[0].plot(pm_xdata, quadratic(pm_xdata, *pm_opt), '--r', label="Fit $f(x) = a + b x + c x^2$")
    ax[1].plot(pp_xdata, pp_times.mean(axis=-1), 'o', label='Measured times')
    ax[1].plot(pp_xdata, linear(pp_xdata, *pp_opt), '--r', label="Fit $f(x) = a x$")

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')

    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    ax[0].legend()
    ax[1].legend()

    ax[0].set_xticks(pm_xdata)
    ax[0].set_xticklabels(pm_xlabels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax[1].set_xticks(pp_xdata[0:-1:3])
    ax[1].set_xticklabels(pp_xlabels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax[0].set_title("PM calculation")
    ax[1].set_title("PP calculation")

    ax[0].set_xlabel('Mesh sizes')
    ax[1].set_xlabel(r'$r_c / a_{ws}$')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, 'Timing_Fit.png'))
    fig.show()


def main(params, estimate=False):
    """
    Run a test to check the force error and estimate run time.

    Parameters
    ----------
    estimate: bool
        Flag for estimating optimal PPPM parameters.

    params : class
        Simulation parameters.
    """
    loops = 10
    if estimate:
        plt.close('all')
    # Change verbose params for printing to screen
    params.Control.verbose = True
    params.Control.pre_run = True
    params.load_method = 'random_no_reject'

    verbose = Verbose(params)
    verbose.sim_setting_summary(params)  # simulation setting summary

    # Initialize particles and all their attributes. Needed for force calculation
    ptcls = Particles(params)
    params.Control.verbose = False  # Turn it off so it doesnt print S_particles print statements
    ptcls.load(params)

    # Check for too large a cut off
    if params.Potential.rc > params.Lv.min() / 2:
        raise ValueError("Cut-off radius is larger than L/2! L/2 = {:1.4e}".format(params.Lv.min() / 2))

    print('\n\n----------------- Time -----------------------\n')

    if params.P3M.on:

        chosen_alpha = params.P3M.G_ew * params.aws
        chosen_rcut = params.Potential.rc / params.aws
        green_time = force_error.optimal_green_function_timer(params)
        force_error.print_time_report("GF", green_time, 0)

        # Calculate Force error from analytic approximation given in Dharuman et al. J Chem Phys 2017
        DeltaF_tot, DeltaF_PP, DeltaF_PM, rcuts, alphas = force_error.analytical_approx_pppm(params)

        # Color Map
        make_color_map(rcuts, alphas, chosen_alpha, chosen_rcut, DeltaF_tot, params)

        # Line Plot
        make_line_plot(rcuts, alphas, chosen_alpha, chosen_rcut, DeltaF_tot, params)
    else:
        DeltaF_tot, rcuts = force_error.analytical_approx_pp(params)

    # Calculate the average over several force calculations
    PP_acc_time, PM_acc_time = force_error.acceleration_timer(params, ptcls, loops)
    # Calculate the mean excluding the first value because that time include numba compilation time
    PP_mean_time = np.mean(PP_acc_time[1:])
    PM_mean_time = np.mean(PM_acc_time[1:])
    force_error.print_time_report("PP", PP_mean_time, loops)
    if params.P3M.on:
        force_error.print_time_report("PM", PM_mean_time, loops)

    # Print estimate of run times
    eq_time = (PP_mean_time + PM_mean_time) * params.Control.Neq
    verbose.time_stamp('Thermalization', eq_time)

    prod_time = (PP_mean_time + PM_mean_time) * params.Control.Nsteps
    verbose.time_stamp('Production', prod_time)

    tot_time = eq_time + prod_time
    verbose.time_stamp('Total Run', tot_time)

    # Plot the calculate Force error
    kappa = params.Potential.matrix[1, 0, 0] if params.Potential.type == "Yukawa" else 0.0
    if estimate:
        print('\n\n----------------- Timing Study -----------------------')
        Mg = np.array([6, 8, 16, 24, 32, 40, 48, 56], dtype=int)
        max_cells = int(0.5 * params.Lv.min() / params.aws)
        Ncells = np.arange(3, max_cells, dtype=int)
        print(Ncells)
        pm_times = np.zeros(len(Mg))
        pm_errs = np.zeros(len(Mg))

        pp_times = np.zeros((len(Mg), len(Ncells)))
        pp_errs = np.zeros((len(Mg), len(Ncells)))

        pm_xlabels = []
        pp_xlabels = []

        DeltaF_map = np.zeros((len(Mg), len(Ncells)))

        # Average the PM time
        for i, m in enumerate(Mg):
            params.P3M.MGrid = m * np.ones(3, dtype=int)
            params.P3M.G_ew = 0.25 * m / params.Lv.min()
            green_time = force_error.optimal_green_function_timer(params)
            pm_errs[i] = params.P3M.PM_err
            print('\n\nMesh = {} x {} x {} : '.format(*params.P3M.MGrid))
            print('alpha = {:1.4e} / a_ws = {:1.4e} '.format(params.P3M.G_ew * params.aws, params.P3M.G_ew))
            print('PM Err = {:1.4e}  '.format(params.P3M.PM_err), end='')

            force_error.print_time_report("GF", green_time, 0)
            pm_xlabels.append("{}x{}x{}".format(*params.P3M.MGrid))
            for it in range(3):
                pm_times[i] += force_error.pm_acceleration_timer(params, ptcls) / 3.0

            for j, c in enumerate(Ncells):
                params.Potential.rc = params.Lv.min() / c
                kappa_over_alpha = - 0.25 * (kappa / params.P3M.G_ew) ** 2
                alpha_times_rcut = - (params.P3M.G_ew * params.Potential.rc) ** 2
                params.P3M.PP_err = 2.0 * np.exp(kappa_over_alpha + alpha_times_rcut) / np.sqrt(params.Potential.rc)
                params.P3M.PP_err *= np.sqrt(params.N) * params.aws ** 2 / np.sqrt(params.box_volume)
                print('rcut = {:2.4f} a_ws = {:2.6e} '.format(params.Potential.rc / params.aws, params.Potential.rc),
                    end='')
                print("[cm]" if params.Control.units == "cgs" else "[m]")
                print('PP Err = {:1.4e}  '.format(params.P3M.PP_err) )
                pp_errs[i, j] = params.P3M.PP_err
                DeltaF_map[i, j] = np.sqrt(params.P3M.PP_err ** 2 + params.P3M.PM_err ** 2)

                if j == 0:
                    pp_xlabels.append("{:1.2f}".format(params.Potential.rc / params.aws))

                for it in range(3):
                    pp_times[i, j] += force_error.pp_acceleration_timer(params, ptcls) / 3.0

        Lagrangian = np.empty((len(Mg), len(Ncells)))
        for i in range(len(Mg)):
            for j in range(len(Ncells)):
                Lagrangian[i, j] = abs(pp_errs[i, j] ** 2 * pp_times[i, j] - pm_errs[i] ** 2 * pm_times[i])

        best = np.unravel_index(Lagrangian.argmin(), Lagrangian.shape)
        print(Mg.shape, Ncells.shape)
        c_mesh, m_mesh = np.meshgrid(Ncells, Mg)
        print(m_mesh.shape, c_mesh.shape)
        print(m_mesh[best], c_mesh[best])
        # levels = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        fig = plt.figure()
        ax = fig.add_subplot(111) # projection='3d')
        # CS = ax.plot_surface(m_mesh, c_mesh, Lagrangian, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        CS = ax.contourf(m_mesh, c_mesh, Lagrangian, norm=LogNorm(vmin=Lagrangian.min(), vmax=Lagrangian.max()))
        CS2 = ax.contour(CS, colors='w')
        ax.clabel(CS2, fmt='%1.0e', colors='w')
        fig.colorbar(CS)
        ax.scatter(Mg[best[0]], Ncells[best[1]], s=200, c='k')
        ax.set_xlabel('Mesh size')
        ax.set_ylabel(r'No. Cells = $1/r_c$')
        ax.set_title('2D Lagrangian')
        fig.savefig(os.path.join(params.Control.pre_run_dir, '2D_Lagrangian.png'))
        fig.show()

        fig, ax = plt.subplots(1, 1, figsize=(11, 7))
        if DeltaF_tot.min() == 0.0:
            minv = 1e-120
        else:
            minv = DeltaF_tot.min()
        CS = ax.contourf(m_mesh, c_mesh, DeltaF_map, norm=LogNorm(vmin=minv, vmax=DeltaF_tot.max()))
        CS2 = ax.contour(CS, colors='w')
        ax.scatter(Mg[best[0]], Ncells[best[1]], s=200, c='k')
        ax.clabel(CS2, fmt='%1.0e', colors='w')
        fig.colorbar(CS)
        ax.set_xlabel('Mesh size')
        ax.set_ylabel(r'No. Cells = $1/r_c$')
        ax.set_title('Force Error')
        fig.savefig(os.path.join(params.Control.pre_run_dir, 'ForceMap.png'))
        fig.show()

        params.P3M.MGrid = Mg[best[0]] * np.ones(3, dtype=int)
        params.Potential.rc = params.Lv.min() / Ncells[best[1]]
        params.P3M.G_ew = 0.25 * m_mesh[best] / params.Lv.min()
        params.P3M.Mx = params.P3M.MGrid[0]
        params.P3M.My = params.P3M.MGrid[1]
        params.P3M.Mz = params.P3M.MGrid[2]
        params.P3M.hx = params.Lx / float(params.P3M.Mx)
        params.P3M.hy = params.Ly / float(params.P3M.My)
        params.P3M.hz = params.Lz / float(params.P3M.Mz)
        params.P3M.PM_err = pm_errs[best[0]]
        params.P3M.PP_err = pp_errs[best]
        params.P3M.F_err = np.sqrt(params.P3M.PM_err ** 2 + params.P3M.PP_err ** 2)
        verbose.timing_study(params)

        predicted_times = pp_times[best] + pm_times[best[0]]
        # Print estimate of run times
        verbose.time_stamp('Thermalization', predicted_times * params.Control.Neq)
        verbose.time_stamp('Production', predicted_times * params.Control.Nsteps)
        verbose.time_stamp('Total Run', predicted_times * (params.Control.Neq + params.Control.Nsteps))


if __name__ == '__main__':
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
    op.add_option("-r", "--restart", action='store', dest='restart', help="Restart step")
    op.add_option("-e", "--estimate", action='store_true', dest='estimate', help="Estimate optimal parameters")
    options, pot = op.parse_args()

    params = Params()
    params.setup(vars(options))  # Read initial conditions and setup parameters

    main(params, options.estimate)
