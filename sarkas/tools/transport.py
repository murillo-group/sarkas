import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sarkas Modules
import sarkas.tools.observables as obs


class TransportCoefficient:
    """
    Transport Coefficients class.
    """

    def __init__(self):
        pass

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = 'Transport( \n'
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ')'
        return disp

    @staticmethod
    def electrical_conductivity(params, phase=None, show=False):
        """
        Calculate electrical conductivity from current auto-correlation function.

        Parameters
        ----------
        params : sarkas.base.Parameters
            Simulation's parameters.

        phase : str
            Phase to analyze. Default = 'production'.

        show : bool
            Flag for prompting plot to screen.

        Returns
        -------
        coefficient : pandas.DataFrame
            Pandas dataframe containing the value of the transport coefficient as a function of integration time.

        """
        coefficient = pd.DataFrame()
        energies = obs.Thermodynamics()
        energies.setup(params, phase)
        energies.parse('production')
        j_current = obs.ElectricCurrent()
        j_current.setup(params, phase)
        j_current.parse()
        sigma = np.zeros(j_current.prod_no_dumps)
        integrand = np.array(
            j_current.dataframe["Total Current ACF"] / j_current.dataframe["Total Current ACF"].iloc[0])
        time = np.array(j_current.dataframe["Time"])
        const = params.total_plasma_frequency ** 2
        for it in range(1, j_current.prod_no_dumps):
            sigma[it] = np.trapz(integrand[:it], x=time[:it])
        coefficient["Time"] = time
        coefficient["Electrical Conductivity"] = const * sigma
        # Plot the transport coefficient at different integration times
        xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(j_current.dataframe["Time"], sigma, "Time", "Conductivity",
                                                       j_current.units)
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(10, 9))
        ax21 = ax2.twiny()
        # extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.2)

        ax1.semilogx(xmul * time, integrand, label=r'$j(t)$')
        ax21.semilogx(ymul * sigma)
        ax2.semilogx(xmul * time, ymul * sigma, label=r'$\sigma (t)$')

        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_ylabel(r'Total Current ACF $j(t)$')

        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_ylabel(r'Conductivity' + ylbl)
        ax2.set_xlabel(r'Time' + xlbl)

        ax21.xaxis.set_ticks_position("bottom")
        ax21.xaxis.set_label_position("bottom")
        ax21.set_xscale('log')
        ax21.grid(False)
        # Offset the twin axis below the host
        ax21.spines["bottom"].set_position(("axes", -0.3))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        ax21.set_frame_on(True)
        ax21.patch.set_visible(False)

        for sp in ax21.spines.values():
            sp.set_visible(False)

        ax21.spines["bottom"].set_visible(True)
        ax21.set_xlabel(r"Index")

        # fig.tight_layout()
        fig.savefig(os.path.join(j_current.saving_dir, 'ConductivityPlot_' + j_current.job_id + '.png'))
        if show:
            fig.show()

        coefficient.to_csv(os.path.join(j_current.saving_dir, 'Conductivity_' + j_current.job_id + '.png'),
                           index=False, encoding='utf-8')

        return coefficient

    @staticmethod
    def diffusion(params,
                  phase: str = 'production',
                  no_slices: int = 1,
                  time_averaging: bool = False,
                  timesteps_to_skip: int = 100,
                  show: bool = False,
                  figname: str = None,
                  **kwargs):
        """
        Calculate the self-diffusion coefficient from the velocity auto-correlation function.

        Parameters
        ----------
        figname : str
            Name with which to save the file. It automatically saves it in the correct directory.

        time_averaging: bool
            Flag for VACF time averaging. Default = False.

        timesteps_to_skip: int
            Timestep interval for VACF time averaging. Default = 100

        params : sarkas.base.Parameters
            Simulation's parameters.

        phase : str
            Phase to analyze. Default = 'production'.

        show : bool
            Flag for prompting plot to screen.

        **kwargs:
            Arguments to pass :meth:`sarkas.tools.observables.VelocityAutoCorrelationFunction`

        Returns
        -------
        coefficient : pandas.DataFrame
            Pandas dataframe containing the value of the transport coefficient as a function of integration time.

        """
        print('\n\n{:=^70} \n'.format(' Diffusion Coefficient '))
        coefficient = pd.DataFrame()
        vacf = obs.VelocityAutoCorrelationFunction()
        vacf.setup(params,
                   phase=phase,
                   time_averaging=time_averaging,
                   timesteps_to_skip=timesteps_to_skip,
                   no_slices=no_slices,
                   **kwargs)
        vacf.compute()
        time = np.array(vacf.dataframe["Time"])
        coefficient["Time"] = time

        fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        ax2 = ax1.twinx()  # Diffusion axes
        ax21 = ax2.twiny()  # Index axes
        # extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.1)
        if not params.magnetized:
            D = np.zeros((params.num_species, len(time)))

            const = 1.0 / 3.0
            for i, sp in enumerate(params.species_names):
                integrand = np.array(vacf.dataframe["{} Total Velocity ACF".format(sp)])
                for it in range(1, len(time)):
                    D[i, it] = const * np.trapz(integrand[:it], x=time[:it])

                coefficient["{} Diffusion".format(sp)] = D[i, :]

                xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, D[i, :], "Time", "Diffusion", vacf.units)
                ax1.semilogx(xmul * time, integrand / integrand[0], label=r'$Z_{' + sp + '}(t)$')
                ax21.semilogx(ymul * D[i, :], '--')
                ax2.semilogx(xmul * time, ymul * D[i, :], '--', label=r'$D_{' + sp + '}(t)$')
        else:
            D = np.zeros((params.num_species, 2, len(time)))

            for i, sp in enumerate(params.species_names):
                integrand_par = np.array(vacf.dataframe["{} Z Velocity ACF".format(sp)])
                integrand_perp = np.array(vacf.dataframe["{} X Velocity ACF".format(sp)]) + \
                                 np.array(vacf.dataframe["{} Y Velocity ACF".format(sp)])
                for it in range(1, len(time)):
                    D[i, 0, it] = np.trapz(integrand_par[:it], x=time[:it])
                    D[i, 1, it] = 0.5 * np.trapz(integrand_perp[:it], x=time[:it])

                coefficient["{} Parallel Diffusion".format(sp)] = D[i, 0, :]
                coefficient["{} Perpendicular Diffusion".format(sp)] = D[i, 1, :]

                xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, D[i, 0, :], "Time", "Diffusion", vacf.units)
                zpar = ax1.semilogx(xmul * time, integrand_par / integrand_par[0],
                                    label='{}  '.format(sp) + r'$Z_{\parallel}(t)$')
                zperp = ax1.semilogx(xmul * time, integrand_perp / integrand_perp[0],
                                     label='{}  '.format(sp) + r'$Z_{\perp}(t)$')
                ax21.semilogx(ymul * D[i, 0, :], '--')
                ax21.semilogx(ymul * D[i, 1, :], '--')
                ax2.semilogx(xmul * time, ymul * D[i, 0, :], '--', label='{}  '.format(sp) + r'$D_{\parallel}(t)$')
                ax2.semilogx(xmul * time, ymul * D[i, 1, :], '--', label='{}  '.format(sp) + r'$D_{\perp}(t)$')

        # Complete figure
        ax1.grid(False)
        ax1.legend(loc='upper left')
        ax1.set_ylabel(r'Velocity ACF')
        ax1.legend(loc='best')

        ax1.set_xlabel(r'Time' + xlbl)
        ax2.grid(False)
        ax2.legend(loc='best')
        ax2.set_ylabel(r'Diffusion' + ylbl)
        # ax2.set_xlabel(r'Time' + xlbl)

        ax21.xaxis.set_ticks_position("bottom")
        ax21.xaxis.set_label_position("bottom")
        ax21.set_xscale('log')
        ax21.grid(False)
        # Offset the twin axis below the host
        ax21.spines["bottom"].set_position(("axes", -0.2))

        # Turn on the frame for the twin axis, but then hide all
        # but the bottom spine
        ax21.set_frame_on(True)
        ax21.patch.set_visible(False)

        for sp in ax21.spines.values():
            sp.set_visible(False)
        ax21.spines["bottom"].set_visible(True)
        ax2.set_xlabel(r'Time' + xlbl)
        ax21.set_xlabel(r"Index")
        # ax21.set_xbound(1, len(time))

        fig.tight_layout()
        if figname:
            fig.savefig(os.path.join(vacf.saving_dir, figname))
        else:
            fig.savefig(os.path.join(vacf.saving_dir, 'Plot_Diffusion_' + vacf.job_id + '.png'))

        if show:
            fig.show()
        coefficient.to_csv(os.path.join(vacf.saving_dir, 'Diffusion_' + vacf.job_id + '.csv'),
                           index=False,
                           encoding='utf-8')
        return coefficient

    @staticmethod
    def interdiffusion(params,
                       phase: str = 'production',
                       no_slices: int = 1,
                       time_averaging: bool = False,
                       timesteps_to_skip: int = 100,
                       plot: bool = True,
                       show: bool = False,
                       figname: str = None,
                       **kwargs):
        """
        Calculate the interdiffusion coefficients from the diffusion flux auto-correlation function.

        Parameters
        ----------
        params : sarkas.base.Parameters
            Simulation's parameters.

        phase : str, optional
            Phase to compute. Default = 'production'.

        no_slices : int, optional
            Number of slices of the simulation. Default = 1.

        time_averaging: bool, optional
            Flag for species diffusion flux time averaging. Default = False.

        timesteps_to_skip: int, optional
            Timestep interval for species diffusion flux time averaging. Default = 100

        plot : bool, optional
            Flag to plot transport coefficient with corresponding autocorrelation function. Default = True.

        figname : str, optional
            Name with which to save the file. It automatically saves it in the correct directory.

        show : bool, optional
            Flag for prompting plot to screen when using IPython kernel. Default = False.

        **kwargs:
            Arguments to pass :meth:`sarkas.tools.observables.FluxAutoCorrelationFunction`

        Returns
        -------
        coefficient : pandas.DataFrame
            Pandas dataframe containing the value of the transport coefficient as a function of integration time.

        """
        print('\n\n{:=^70} \n'.format(' Interdiffusion Coefficient '))
        coefficient = pd.DataFrame()
        jc_acf = obs.FluxAutoCorrelationFunction()
        # jc_acf.no_slices = no_slices
        jc_acf.setup(
            params,
            phase=phase,
            time_averaging=time_averaging,
            timesteps_to_skip=timesteps_to_skip,
            no_slices=no_slices,
            **kwargs)
        jc_acf.compute()
        no_int = jc_acf.slice_steps
        no_obs = jc_acf.no_obs

        const = 1. / (3.0 * jc_acf.total_num_ptcls)

        time = np.array(jc_acf.dataframe["Time"])
        coefficient["Time"] = time

        for isl in range(jc_acf.no_slices):
            D_ij = np.zeros((no_obs, jc_acf.slice_steps))
            index = 0
            for i, sp1 in enumerate(params.species_names):
                conc1 = jc_acf.species_concentrations[i] * jc_acf.species_mass_densities[i]
                for j, sp2 in enumerate(params.species_names[i:], i):
                    conc2 = jc_acf.species_concentrations[j] * jc_acf.species_mass_densities[j]

                    integrand = np.array(
                        jc_acf.dataframe["{}-{} Total Diffusion Flux ACF slice {}".format(sp1, sp2, isl)])

                    for it in range(1, no_int):
                        D_ij[index, it] = const / (conc1 * conc2) * np.trapz(integrand[:it], x=time[:it])

                    coefficient["{}-{} Inter Diffusion slice {}".format(sp1, sp2, isl)] = D_ij[index, :]

                    index += 1

        for i, sp1 in enumerate(params.species_names):
            for j, sp2 in enumerate(params.species_names[i:], i):
                col_str = ["{}-{} Inter Diffusion slice {}".format(sp1, sp2, isl) for isl in range(jc_acf.no_slices)]
                coefficient["{}-{} Inter Diffusion avg".format(sp1, sp2)] = coefficient[col_str].mean(axis=1)
                coefficient["{}-{} Inter Diffusion std".format(sp1, sp2)] = coefficient[col_str].std(axis=1)

        if plot or figname:
            # Make the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

            # extra space for the second axis at the bottom
            fig.subplots_adjust(bottom=0.2)
            for i, sp1 in enumerate(params.species_names):
                for j, sp2 in enumerate(params.species_names[i + 1:], i + 1):
                    acf_avg = jc_acf.dataframe["{}-{} Total Diffusion Flux ACF avg".format(sp1, sp2)]
                    acf_std = jc_acf.dataframe["{}-{} Total Diffusion Flux ACF std".format(sp1, sp2)]

                    d_avg = coefficient["{}-{} Inter Diffusion avg".format(sp1, sp2)]
                    d_std = coefficient["{}-{} Inter Diffusion std".format(sp1, sp2)]

                    # Calculate axis multipliers and labels
                    xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, d_avg, "Time", "Diffusion", jc_acf.units)

                    ax1.plot(xmul * time, acf_avg / acf_avg[0], label=r'$J_{' + sp1 + sp2 + '}$')
                    ax1.fill_between(
                        xmul * time,
                        (acf_avg + acf_std) / (acf_avg[0] + acf_std[0]),
                        (acf_avg - acf_std) / (acf_avg[0] - acf_std[0]), alpha=0.2)

                    # ax21.semilogx(ymul * D_ij[index, :], '--')
                    ax2.plot(xmul * time, ymul* d_avg, label=r'$D_{' + sp1 + sp2 + '}$')
                    ax2.fill_between(xmul * time, ymul * (d_avg + d_std), ymul*(d_avg - d_std), alpha=0.2)

            ax1.set(xscale='log', ylabel=r'Diffusion Flux ACF', xlabel=r"Time" + xlbl)
            ax2.set(xscale='log', ylabel=r'Inter Diffusion' + ylbl, xlabel=r"Time" + xlbl)

            # ax21.xaxis.set_ticks_position("bottom")
            # ax21.xaxis.set_label_position("bottom")

            ax1.legend(loc='best')
            ax2.legend(loc='best')
            # ax21.set_xscale('log')
            # Offset the twin axis below the host
            # ax21.spines["bottom"].set_position(("axes", -0.2))

            # Turn on the frame for the twin axis, but then hide all
            # but the bottom spine
            # ax21.set_frame_on(True)
            # ax21.patch.set_visible(False)

            # for sp in ax21.spines.values():
            #     sp.set_visible(False)

            # ax21.spines["bottom"].set_visible(True)
            # ax21.set_xlabel(r"Index")

            fig.tight_layout()
            if figname:
                fig.savefig(os.path.join(jc_acf.saving_dir, figname))
            else:
                fig.savefig(os.path.join(jc_acf.saving_dir, 'Plot_InterDiffusion_' + jc_acf.job_id + '.png'))

            if show:
                fig.show()

        # Save to dataframe
        coefficient.to_csv(os.path.join(jc_acf.saving_dir, 'InterDiffusion_' + jc_acf.job_id + '.csv'),
                           index=False,
                           encoding='utf-8')
        return coefficient

    @staticmethod
    def viscosity(params, phase: str = 'production', show=False):
        """
        TODO: Calculate bulk and shear viscosity from pressure auto-correlation function.

        Parameters
        ----------
        params : sarkas.base.Parameters
            Simulation's parameters.

        phase : str
            Phase to analyze. Default = 'production'.

        show : bool
            Flag for prompting plot to screen.

        Returns
        -------
        coefficient : pandas.DataFrame
            Pandas dataframe containing the value of the transport coefficient as a function of integration time.

        """
        print('\n\n{:=^70} \n'.format(' Viscosity Coefficient '))
        coefficient = pd.DataFrame()
        energies = obs.Thermodynamics()
        energies.setup(params, phase)
        energies.parse('production')
        beta = (energies.kB * energies.dataframe["Temperature"].mean()) ** (-1.0)
        time = np.array(energies.dataframe["Time"])
        coefficient["Time"] = time
        dim_lbl = ['x', 'y', 'z']
        shear_viscosity = np.zeros((params.dimensions, params.dimensions, energies.prod_no_dumps))
        bulk_viscosity = np.zeros(energies.prod_no_dumps)
        const = params.box_volume * beta
        if not 'Pressure Tensor ACF xy' in energies.dataframe.columns:
            print('Pressure not yet calculated')
            print("Calculating Pressure quantities ...")
            energies.compute_pressure_quantities()

        # Calculate the acf of the pressure tensor

        fig, axes = plt.subplots(2, 3, sharex=True, figsize=(16, 10))
        for i, ax1 in enumerate(dim_lbl):
            for j, ax2 in enumerate(dim_lbl):
                integrand = np.array(energies.dataframe["Pressure Tensor ACF {}{}".format(ax1, ax2)])
                for it in range(1, energies.prod_no_dumps):
                    shear_viscosity[i, j, it] = const * np.trapz(integrand[:it], x=time[:it])

                coefficient["{}{} Shear Viscosity Tensor".format(ax1, ax2)] = shear_viscosity[i, j, :]
                xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, shear_viscosity[i, j, :],
                                                               "Time", "Viscosity", energies.units)
                if "{}{}".format(ax1, ax2) in ["yx", "xy"]:
                    axes[0, 0].semilogx(xmul * time, integrand / integrand[0],
                                        label=r"$P_{" + "{}{}".format(ax1, ax2) + " }(t)$")
                    axes[1, 0].semilogx(xmul * time, ymul * shear_viscosity[i, j, :],
                                        label=r"$\eta_{ " + "{}{}".format(ax1, ax2) + " }(t)$")

                elif "{}{}".format(ax1, ax2) in ["xz", "zx"]:
                    axes[0, 1].semilogx(xmul * time, integrand / integrand[0],
                                        label=r"$P_{" + "{}{}".format(ax1, ax2) + " }(t)$")
                    axes[1, 1].semilogx(xmul * time, ymul * shear_viscosity[i, j, :],
                                        label=r"$\eta_{ " + "{}{}".format(ax1, ax2) + " }(t)$")

                elif "{}{}".format(ax1, ax2) in ["yz", "zy"]:
                    axes[0, 2].semilogx(xmul * time, integrand / integrand[0],
                                        label=r"$P_{" + "{}{}".format(ax1, ax2) + " }(t)$")
                    axes[1, 2].semilogx(xmul * time, ymul * shear_viscosity[i, j, :],
                                        label=r"$\eta_{ " + "{}{}".format(ax1, ax2) + " }(t)$")
            axes[0, 0].legend()
            axes[0, 1].legend()
            axes[0, 2].legend()
            axes[1, 0].legend()
            axes[1, 1].legend()
            axes[1, 2].legend()
            axes[0, 0].set_xlabel(r"Time " + xlbl)
            axes[0, 1].set_xlabel(r"Time " + xlbl)
            axes[0, 2].set_xlabel(r"Time " + xlbl)

            axes[1, 0].set_xlabel(r"Time " + xlbl)
            axes[1, 1].set_xlabel(r"Time " + xlbl)
            axes[1, 2].set_xlabel(r"Time " + xlbl)

            axes[0, 0].set_ylabel(r"Pressure Tensor ACF")
            axes[1, 0].set_ylabel(r"Shear Viscosity" + ylbl)

            fig.tight_layout()
            fig.savefig(os.path.join(energies.fldr, "ShearViscosity_Plots_" + energies.job_id + ".png"))
            if show:
                fig.show()

        # Calculate Bulk Viscosity
        pressure_acf = obs.autocorrelationfunction_1D(np.array(energies.dataframe["Pressure"])
                                                      - energies.dataframe["Pressure"].mean())
        bulk_integrand = pressure_acf
        for it in range(1, energies.prod_no_dumps):
            bulk_viscosity[it] = const * np.trapz(bulk_integrand[:it], x=time[:it])

        coefficient["Bulk Viscosity"] = bulk_viscosity

        fig, ax = plt.subplots(2, 1)
        xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, bulk_viscosity, "Time", "Viscosity", energies.units)
        ax[0].semilogx(xmul * time, bulk_integrand / bulk_integrand[0], label=r"$P(t)$")
        ax[1].semilogx(xmul * time, ymul * bulk_viscosity, label=r"$\eta_{V}(t)$")

        ax[0].set_xlabel(r"Time" + xlbl)
        ax[1].set_xlabel(r"Dumps")
        ax[0].set_ylabel(r"Pressure ACF")
        ax[1].set_ylabel(r"Bulk Viscosity" + ylbl)
        ax[0].legend()
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(energies.fldr, "BulkViscosity_Plots_" + energies.job_id + ".png"))
        if show:
            fig.show()

        return coefficient
