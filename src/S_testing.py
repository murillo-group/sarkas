import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time as timer
import sys
from S_verbose import Verbose
from S_params import Params
from S_particles import Particles
import S_calc_force_pp as force_pp
import S_calc_force_pm as force_pm


# Green's function for Yukawa potential
def Gk(x, alpha, kappa):
    return 4.0 * np.pi * np.exp(-(x ** 2 + kappa ** 2) / (2 * alpha) ** 2) / (kappa ** 2 + x ** 2)


def betamp(m, p, alpha, kappa):
    xa = np.linspace(0.0001, 500, 5000)
    # eq.(37) from Dharuman J Chem Phys 146 024112 (2017)
    return np.trapz(Gk(xa, alpha, kappa) * Gk(xa, alpha, kappa) * xa ** (2 * (m + p + 2)), x=xa)


def force_error_approx(params, kappa):
    """

    Parameters
    ----------
    params

    Returns
    -------

    """
    p = params.P3M.cao
    L = params.Lv[0] / params.aws
    h = L / params.P3M.MGrid[0]

    a_min = params.Potential.matrix[-1, 0, 0] * 0.5
    a_max = params.Potential.matrix[-1, 0, 0] * 1.5

    r_min = params.Potential.rc * 0.5
    r_max = params.Potential.rc * 1.5

    alphas = params.aws * np.linspace(a_min, a_max, 200)
    rcuts = np.linspace(r_min, r_max, 200) / params.aws

    DeltaF_PM = np.zeros(len(alphas))
    DeltaF_PP = np.zeros((len(alphas), len(rcuts)))
    Tot_DeltaF = np.zeros((len(alphas), len(rcuts)))

    # Coefficient from
    # Deserno and Holm J Chem Phys 109 7694 (1998)
    if p == 1:
        Cmp = np.array([2 / 3])
    elif p == 2:
        Cmp = np.array([2 / 45, 8 / 189])
    elif p == 3:
        Cmp = np.array([4 / 495, 2 / 225, 8 / 1485])
    elif p == 4:
        Cmp = np.array([2 / 4725, 16 / 10395, 5528 / 3869775, 32 / 42525])
    elif p == 5:
        Cmp = np.array([4 / 93555, 2764 / 11609325, 8 / 25515, 7234 / 32531625, 350936 / 3206852775])
    elif p == 6:
        Cmp = np.array([2764 / 638512875, 16 / 467775, 7234 / 119282625, 1403744 / 25196700375,
                        1396888 / 40521009375, 2485856 / 152506344375])
    elif p == 7:
        Cmp = np.array([8 / 18243225, 7234 / 1550674125, 701872 / 65511420975, 2793776 / 225759909375,
                        1242928 / 132172165125, 1890912728 / 352985880121875, 21053792 / 8533724574375])

    for ia, alpha in enumerate(alphas):
        summa = 0.0
        for m in np.arange(p):
            expp = 2 * (m + p)
            summa += Cmp[m] * (2 / (1 + expp)) * betamp(m, p, alpha, kappa) * (h / 2.) ** expp
        # eq.(36) in Dharuman J Chem Phys 146 024112 (2017)
        DeltaF_PM[ia] = np.sqrt(3.0 * summa) / (2.0 * np.pi)
    # eq.(35)
    DeltaF_PM *= np.sqrt(params.total_num_ptcls * params.aws ** 3 / params.box_volume)

    # Calculate the analytic PP error and the total force error
    for (ir, rc) in enumerate(rcuts):
        for (ia, alfa) in enumerate(alphas):
            # eq.(30) from Dharuman J Chem Phys 146 024112 (2017)
            DeltaF_PP[ia, ir] = 2.0 * np.exp(-(0.5 * kappa / alfa) ** 2) * np.exp(- alfa ** 2 * rc ** 2) / np.sqrt(rc)
            DeltaF_PP[ia, ir] *= np.sqrt(params.total_num_ptcls * params.aws ** 3 / params.box_volume)
            # eq.(42) from Dharuman J Chem Phys 146 024112 (2017)
            Tot_DeltaF[ia, ir] = np.sqrt(DeltaF_PM[ia] ** 2 + DeltaF_PP[ia, ir] ** 2)

    return Tot_DeltaF, DeltaF_PP, DeltaF_PM, rcuts, alphas


def optimal_green_function(params):
    """
    Calculate the Optimal Green Function.

    Parameters
    ----------
    params : class
        Simulation parameters

    Returns
    -------
    green_function_time : float
        Calculation time.
    """
    # Dev notes. I am rewriting the functions here because I need to put a counter around it.
    # Optimized Green's Function
    start = timer.time()
    params.P3M.G_k, params.P3M.kx_v, params.P3M.ky_v, params.P3M.kz_v, params.P3M.PM_err = force_pm.gf_opt(
        params.P3M.MGrid, params.P3M.aliases, params.Lv, params.P3M.cao, pot_mat, rcut, params.fourpie0)
    green_time = timer.time() - start

    # Notice that the equation was derived for a single component plasma.
    params.P3M.PP_err = np.exp(-0.25 * kappa ** 2 / alpha ** 2 - alpha ** 2 * rcut ** 2) / np.sqrt(
        params.box_volume * rcut)
    params.P3M.PP_err *= 2.0 * np.sqrt(params.N) * params.aws ** 2
    params.P3M.PM_err *= np.sqrt(params.N) * params.aws ** 2 * params.fourpie0 / params.box_volume ** (2. / 3.)
    # Total force error
    params.P3M.F_err = np.sqrt(params.P3M.PM_err ** 2 + params.P3M.PP_err ** 2)
    print_time_report("GF", green_time, 0)


def force_calculations(params, ptcls, loops):
    """
    Calculate the average time for force calculation.

    Parameters
    ----------
    params : class
        Simulation parameters.

    ptcls : class
        Particles data.

    loops : int
        Number of loops for averaging.

    Returns
    -------
    PP_force_time : array
        Times for the PP force calculation.

    PM_force_time : array
        Times for the PM force calculation.

    """
    # Dev notes. I am rewriting the functions here because I need to put a counter around it.

    PP_force_time = np.zeros(loops + 1)
    PM_force_time = np.zeros(loops + 1)

    for it in range(loops + 1):
        if params.Potential.LL_on:
            start = timer.time()
            U_short, acc_s_r = force_pp.update(ptcls.pos, ptcls.species_id, ptcls.mass, params.Lv,
                                               params.Potential.rc, params.Potential.matrix, params.force,
                                               params.Control.measure, ptcls.rdf_hist)
            PP_force_time[it] = timer.time() - start
        else:
            start = timer.time()
            U_short, acc_s_r = force_pp.update_0D(ptcls.pos, ptcls.species_id, ptcls.mass, params.Lv,
                                                  params.Potential.rc, params.Potential.matrix, params.force,
                                                  params.Control.measure, ptcls.rdf_hist)
            PP_force_time[it] = timer.time() - start

        if params.P3M.on:
            start = timer.time()
            U_long, acc_l_r = force_pm.update(ptcls.pos, ptcls.charge, ptcls.mass,
                                              params.P3M.MGrid, params.Lv, params.P3M.G_k, params.P3M.kx_v,
                                              params.P3M.ky_v,
                                              params.P3M.kz_v, params.P3M.cao)
            PM_force_time[it] = timer.time() - start

    return PP_force_time, PM_force_time


def print_time_report(str_id, t, loops):
    if str_id == "GF":
        print("\nOptimal Green's Function Time = {:1.3f} sec".format(t))
    elif str_id == "PP":
        print('\nAverage time of PP Force calculation over {} loops: {:1.3f} msec'.format(loops, t * 1e3))
    elif str_id == "P3M":
        print('\nAverage time of P3M Force calculation over {} loops: {:1.3f} msec'.format(loops, t * 1e3))


input_file = sys.argv[1]
try:
    loops = int(sys.argv[2])
except IndexError:
    loops = 10

# Calculate params from input
params = Params()
params.setup(input_file)

params.Control.verbose = True
params.Control.pre_run = True

verbose = Verbose(params)
verbose.sim_setting_summary(params)  # simulation setting summary

rcut = params.Potential.rc
alpha = params.Potential.matrix[-1, 0, 0]
kappa = params.Potential.matrix[0, 0, 0] if params.Potential.type == "Yukawa" else 0.0
pot_mat = np.array([kappa, alpha])

# Calculate the analytical formula given in Dharuman et al.
DeltaF_tot, Delta_F_PP, DeltaF_PM, rcuts, alphas = force_error_approx(params, kappa * params.aws)

chosen_alpha = alpha * params.aws
chosen_rcut = rcut / params.aws

# Plot the results
fsz = 16
lw = 2
fig, ax = plt.subplots(1, 2, figsize=(12, 7))
ax[0].plot(rcuts, DeltaF_tot[50, :], label=r'$\alpha = {:2.4f}$'.format(alphas[50]))
ax[0].plot(rcuts, DeltaF_tot[57, :], label=r'$\alpha = {:2.4f}$'.format(alphas[57]))
ax[0].plot(rcuts, DeltaF_tot[60, :], label=r'$\alpha = {:2.4f}$'.format(alphas[60]))
ax[0].set_xlabel(r'$r_c/a_{ws}$', fontsize=fsz)
ax[0].set_yscale('log')
ax[0].grid(True, alpha=0.3)
ax[0].tick_params(labelsize=fsz)
ax[0].legend(loc='best', fontsize=fsz)

# ax[1].semilogy(alphas, DeltaF_PP[:,50,kp], lw = lwh, label = r'$\kappa = {:2.2f}$'.format(kappas[kp]))
# ax[1].semilogy(alphas, DeltaF_PM[:,9], lw = lwh, label = r'$\kappa = {:2.2f}$'.format(kappas[9]))
ax[1].plot(alphas, DeltaF_tot[:, 50], label=r'$r_c = {:2.4f}$'.format(rcuts[50]))
ax[1].plot(alphas, DeltaF_tot[:, 60], label=r'$r_c = {:2.4f}$'.format(rcuts[60]))
ax[1].plot(alphas, DeltaF_tot[:, 82], label=r'$r_c = {:2.4f}$'.format(rcuts[82]))
ax[1].plot(alphas, DeltaF_tot[:, 150], label=r'$r_c = {:2.4f}$'.format(rcuts[150]))
ax[1].set_xlabel(r'$\alpha \; a_{ws}$', fontsize=fsz)
ax[1].set_yscale('log')
ax[1].axvline(chosen_alpha)
# ax[1].set_xscale('log')
ax[1].tick_params(labelsize=fsz)
ax[0].set_title(r'$N = {} \quad M = {}, \quad \kappa = {:1.3f}$'.format(params.total_num_ptcls,
                                                                        params.P3M.MGrid[0],
                                                                        kappa * params.aws), fontsize=fsz)
ax[1].grid(True, alpha=0.3)
ax[1].legend(loc='best', fontsize=fsz)
fig.tight_layout()
fig.show()

r_mesh, a_mesh = np.meshgrid(rcuts, alphas)
cmps = 'viridis'
origin = 'lower'
levels = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
CS = ax.contourf(a_mesh, r_mesh, DeltaF_tot[:, :], norm=LogNorm(vmin=DeltaF_tot.min(), vmax=1), cmap=cmps)
CS2 = ax.contour(CS, colors='w')
ax.clabel(CS2, fmt='%1.0e', colors='w', fontsize=fsz)
cbar = fig.colorbar(CS)
cbar.ax.tick_params(labelsize=fsz)
# Add the contour line levels to the colorbar
# cbar.add_lines(CS2)
ax.scatter(chosen_alpha, chosen_rcut, s=200, c='k')
ax.tick_params(labelsize=fsz)
ax.set_xlabel(r'$\alpha \;a_{ws}$', fontsize=fsz)
ax.set_ylabel(r'$r_c/a_{ws}$', fontsize=fsz)
ax.set_title(r'$N = {} \quad M = {}, \quad \kappa = {:1.3f}$'.format(params.total_num_ptcls,
                                                                     params.P3M.MGrid[0],
                                                                     kappa * params.aws), fontsize=fsz)
fig.tight_layout()
fig.show()

green_time = optimal_green_function(params)

# Initialize particles and all the attributes needed for force calculation
ptcls = Particles(params)
ptcls.load(params)

# Calculate the average over several force calculations
PP_force_time, PM_force_time = force_calculations(params, ptcls, loops)
# Calculate the mean excluding the first value because that time include numba compilation time
print_time_report("PP", np.mean(PP_force_time[1:]), loops)
print_time_report("P3M", np.mean(PM_force_time[1:]), loops)
