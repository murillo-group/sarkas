"""
Module for testing simulation parameters
"""
# Python modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import argparse

# Sarkas modules
from S_verbose import Verbose
from S_params import Params
from S_particles import Particles
import S_force_error as force_error

plt.style.use(os.path.join(os.path.join(os.getcwd(), 'src'), 'MSUstyle'))


def main(params):
    """
    Run a test to check the force error and estimate run time.

    Parameters
    ----------
    params : class
        Simulation parameters.
    """
    loops = 10

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

    print('\n\n----------------- Time -----------------------')

    if hasattr(params.P3M, 'G_ew'):
        # Calculate the analytical formula given in Dharuman et al.
        DeltaF_tot, DeltaF_PP, DeltaF_PM, rcuts, alphas = force_error.analytical_approx_pppm(params)

        chosen_alpha = params.P3M.G_ew * params.aws
        chosen_rcut = params.Potential.rc / params.aws

        green_time = force_error.optimal_green_function_timer(params)
        force_error.print_time_report("GF", green_time, 0)

        # Plot the calculate Force error
        plt.close('all')
        kappa_title = 0.0 if params.Potential.type == "Coulomb" else params.Potential.matrix[1, 0, 0]
        # Plot the results
        fig_path = params.Control.pre_run_dir
        # Color Map
        r_mesh, a_mesh = np.meshgrid(rcuts, alphas)
        cmps = 'viridis'
        origin = 'lower'
        levels = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        CS = ax.contourf(a_mesh, r_mesh, DeltaF_tot[:, :], norm=LogNorm(vmin=DeltaF_tot.min(), vmax=1), cmap=cmps)
        CS2 = ax.contour(CS, colors='w')
        ax.clabel(CS2, fmt='%1.0e', colors='w')
        cbar = fig.colorbar(CS)
        # cbar.ax.tick_params(labelsize=fsz)
        # Add the contour line levels to the colorbar
        # cbar.add_lines(CS2)
        ax.scatter(chosen_alpha, chosen_rcut, s=200, c='k')
        if rcuts[-1] * params.aws > 0.5 * params.Lv.min():
            ax.axhline(0.5 * params.Lv.min() / params.aws, c='r', label=r'$L/2$')
        # ax.tick_params(labelsize=fsz)
        ax.set_xlabel(r'$\alpha \;a_{ws}$')
        ax.set_ylabel(r'$r_c/a_{ws}$')
        ax.set_title(
            r'$\Delta F^{apprx}_{tot}(r_c,\alpha)$' + r'  for  $N = {}, \quad M = {}, \quad \kappa = {:1.3f}$'.format(
                params.total_num_ptcls, params.P3M.MGrid[0], kappa_title * params.aws))
        fig.tight_layout()
        fig.savefig(os.path.join(fig_path, 'ForceError_ClrMap_' + params.Control.fname_app + '.png'))
        fig.show()
        # Line Plot
        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 7))
        ax[0].plot(rcuts, DeltaF_tot[30, :], label=r'$\alpha a_{ws} = ' + '{:2.4f}$'.format(alphas[30]))
        ax[0].plot(rcuts, DeltaF_tot[40, :], label=r'$\alpha a_{ws} = ' + '{:2.4f}$'.format(alphas[40]))
        ax[0].plot(rcuts, DeltaF_tot[50, :], label=r'$\alpha a_{ws} = ' + '{:2.4f}$'.format(alphas[50]))
        ax[0].plot(rcuts, DeltaF_tot[60, :], label=r'$\alpha a_{ws} = ' + '{:2.4f}$'.format(alphas[60]))
        ax[0].plot(rcuts, DeltaF_tot[70, :], label=r'$\alpha a_{ws} = ' + '{:2.4f}$'.format(alphas[70]))
        ax[0].set_ylabel(r'$\Delta F^{apprx}_{tot}$')
        ax[0].set_xlabel(r'$r_c/a_{ws}$')
        ax[0].set_yscale('log')
        ax[0].axvline(chosen_rcut, ls='--', c='k')
        ax[0].axhline(params.P3M.F_err, ls='--', c='k')
        if rcuts[-1] * params.aws > 0.5 * params.Lv.min():
            ax[0].axvline(0.5 * params.Lv.min() / params.aws, c='r', label=r'$L/2$')
        ax[0].grid(True, alpha=0.3)
        # ax[0].tick_params(labelsize=fsz)
        ax[0].legend(loc='best')

        # ax[1].semilogy(alphas, DeltaF_PP[:,50,kp], lw = lwh, label = r'$\kappa = {:2.2f}$'.format(kappas[kp]))
        # ax[1].semilogy(alphas, DeltaF_PM[:,9], lw = lwh, label = r'$\kappa = {:2.2f}$'.format(kappas[9]))
        ax[1].plot(alphas, DeltaF_tot[:, 30], label=r'$r_c = {:2.4f}'.format(rcuts[30]) + ' a_{ws}$')
        ax[1].plot(alphas, DeltaF_tot[:, 40], label=r'$r_c = {:2.4f}'.format(rcuts[40]) + ' a_{ws}$')
        ax[1].plot(alphas, DeltaF_tot[:, 50], label=r'$r_c = {:2.4f}'.format(rcuts[50]) + ' a_{ws}$')
        ax[1].plot(alphas, DeltaF_tot[:, 60], label=r'$r_c = {:2.4f}'.format(rcuts[60]) + ' a_{ws}$')
        ax[1].plot(alphas, DeltaF_tot[:, 70], label=r'$r_c = {:2.4f}'.format(rcuts[70]) + ' a_{ws}$')
        ax[1].set_xlabel(r'$\alpha \; a_{ws}$')
        ax[1].set_yscale('log')
        ax[1].axhline(params.P3M.F_err, ls='--', c='k')
        ax[1].axvline(chosen_alpha, ls='--', c='k')
        # ax[1].set_xscale('log')
        # ax[1].tick_params(labelsize=fsz)
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(loc='best')
        fig.suptitle(
            r'Approximate Total Force error  $N = {}, \quad M = {}, \quad \kappa = {:1.2f}$'.format(
                params.total_num_ptcls,
                params.P3M.MGrid[0],
                kappa_title * params.aws))
        fig.savefig(os.path.join(fig_path, 'ForceError_LinePlot_' + params.Control.fname_app + '.png'))
        fig.show()


    else:

        kappa_title = 0.0 if params.Potential.type == "Coulomb" else 1.0 / params.lambda_TF
        # Plot the results

        DeltaF_tot, rcuts = force_error.analytical_approx_pp(params)
        chosen_rcut = params.Potential.rc / params.aws

        fig_path = params.Control.pre_run_dir

        fig, ax = plt.subplots(1, 1)
        ax.plot(rcuts, DeltaF_tot)
        ax.set_ylabel(r'$\Delta F^{apprx}_{tot}$')
        ax.set_xlabel(r'$r_c/a_{ws}$')
        ax.set_yscale('log')
        ax.axvline(chosen_rcut, ls='--', c='k')
        if rcuts[-1] * params.aws > 0.5 * params.Lv.min():
            ax.axvline(0.5 * params.Lv.min() / params.aws, c='r', label=r'$L/2$')
        # ax.grid(True, alpha=0.3)
        # ax[0].tick_params(labelsize=fsz)
        ax.set_title(
            r'Approximate Total Force error  $N = {}, \quad \kappa = {:1.2f}$'.format(
                params.total_num_ptcls, kappa_title * params.aws))
        fig.savefig(os.path.join(fig_path, 'ForceError_LinePlot_' + params.Control.fname_app + '.png'))
        fig.show()

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


if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-t", "--pre_run_testing", required=False, help="Pre Run Testing Flag")
    ap.add_argument("-i", "--input", required=True, help="YAML Input file")
    ap.add_argument("-d", "--job_dir", required=False, help="Job Directory")
    ap.add_argument("-j", "--job_id", required=False, help="Job ID")
    ap.add_argument("-s", "--randseed", required=False, help="Random Number Seed")
    args = vars(ap.parse_args())

    params = Params()
    params.setup(args)  # Read initial conditions and setup parameters

    main(params)
