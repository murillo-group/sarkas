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
        integrand = np.array(j_current.dataframe["Total Current ACF"] / j_current.dataframe["Total Current ACF"][0])
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
    def diffusion(params, phase=None, show=False):
        """
        Calculate diffusion from velocity auto-correlation function.

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
        vacf = obs.VelocityAutocorrelationFunctions()
        vacf.setup(params, phase)
        vacf.parse()
        time = np.array(vacf.dataframe["Time"])
        D = np.zeros((params.num_species, len(time)))
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(10, 9))
        ax21 = ax2.twiny()
        # extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.2)

        const = 1.0 / 3.0
        if params.num_species > 1:
            const *= params.total_mass_density
        for i, sp in enumerate(params.species_names):
            integrand = np.array(vacf.dataframe["{} Total Velocity ACF".format(sp)])
            for it in range(1, len(time)):
                D[i, it] = const * np.trapz(integrand[:it], x=time[:it])

            coefficient["Time"] = time
            coefficient["{} Diffusion".format(sp)] = D[i, :]

            xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(vacf.dataframe["Time"], D[i, :],
                                                       "Time", "Diffusion", vacf.units)
            ax1.semilogx(xmul * time, integrand / integrand[0], label=r'$Z_{' + sp + '}(t)$')
            ax21.semilogx(ymul * D[i, :])
            ax2.semilogx(xmul * time, ymul * D[i, :], label=r'$D_{' + sp + '}(t)$')

        # Complete figure
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_ylabel(r'Velocity ACF')
        # ax1.set_xlabel(r'Time' + xlbl)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_ylabel(r'Diffusion' + ylbl)
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
        # ax21.set_xbound(1, len(time))
        fig.tight_layout()
        fig.savefig(os.path.join(vacf.saving_dir, 'DiffusionPlot_' + vacf.job_id + '.png'))
        if show:
            fig.show()

        return coefficient

    @staticmethod
    def interdiffusion(params, phase=None, show=True):
        """
        Calculate interdiffusion coefficients from velocity auto-correlation function.

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
        vacf = obs.VelocityAutocorrelationFunctions()
        vacf.setup(params, phase)
        vacf.parse()
        no_int = vacf.prod_no_dumps
        no_dij = vacf.no_obs
        D_ij = np.zeros((no_dij, no_int))

        indx = 0
        fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True, figsize=(10, 9))
        ax21 = ax2.twiny()
        # extra space for the second axis at the bottom
        fig.subplots_adjust(bottom=0.2)

        for i, sp1 in enumerate(params.species_names):
            for j, sp2 in enumerate(params.species_names[i + 1:]):
                integrand = np.array(vacf.dataframe["{}-{} Total Current ACF".format(sp1, sp2)])
                time = np.array(vacf.dataframe["Time"])
                # const = 1.0 / (3.0 * params.total_plasma_frequency * params.a_ws ** 2)
                const = 1. / (3.0 * params.species_concentrations[i] * params.species_concentrations[j])
                for it in range(1, no_int):
                    D_ij[indx, it] = const * np.trapz(integrand[:it], x=time[:it])

                coefficient["{}-{} Inter Diffusion".format(sp1, sp2)] = D_ij[i, :]

                xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(vacf.dataframe["Time"], D_ij[i, :],
                                                           "Time", "Diffusion", vacf.units)
                ax1.semilogx(xmul * time, integrand / integrand[0], label=r'$Z_{' + sp1 + sp2 + '}(t)$')
                ax21.semilogx(ymul * D_ij[i, :])
                ax2.semilogx(xmul * time, ymul * D_ij[i, :], label=r'$D_{' + sp1 + sp2 + '}(t)$')

        # Complete figure
        # ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_ylabel(r'Inter Current ACF')

        # ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_ylabel(r'Inter Diffusion' + ylbl)
        ax2.set_xlabel(r'Time' + xlbl)

        ax21.xaxis.set_ticks_position("bottom")
        ax21.xaxis.set_label_position("bottom")
        ax21.set_xscale('log')

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
        # ax21.set_xbound(1, len(time))

        fig.tight_layout()
        fig.savefig(os.path.join(vacf.saving_dir, 'InterDiffusionPlot_' + vacf.job_id + '.png'))
        if show:
            fig.show()

        return coefficient

    @staticmethod
    def viscosity(params, phase=None, show=False):
        """
        Calculate bulk and shear viscosity from pressure auto-correlation function.

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

        fig, axes = plt.subplots(2, 3, sharex=True, figsize=(16, 9))
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

