from IPython import get_ipython

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

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
    def pretty_print(observable, tc_name):
        """Print to screen the location where data is stored and other relevant information.

        Parameters
        ----------

        observable: sarkas.tools.observables.Observable
            Physical quantity of the ACF.

        tc_name: str
            Name of Transport coefficient to calculate.
        """

        print('Data saved in: \n', os.path.join(observable.saving_dir, tc_name + '_' + observable.job_id + '.h5'))
        print('\nNo. of slices = {}'.format(observable.no_slices))
        print('No. dumps per slice = {}'.format(int(observable.slice_steps / observable.dump_step)))

        print('Time interval of autocorrelation function = {:.4e} [s] ~ {} w_p T'.format(
            observable.dt * observable.slice_steps * observable.dump_step,
            int(observable.dt * observable.slice_steps * observable.dump_step * observable.total_plasma_frequency)))

    @staticmethod
    def electrical_conductivity(params,
                                phase: str = 'production',
                                compute_acf: bool = True,
                                no_slices: int = 1,
                                plot: bool = True,
                                show: bool = False,
                                figname: str = None,
                                **kwargs):
        """
        Calculate electrical conductivity from current auto-correlation function.

        Parameters
        ----------
        params : sarkas.core.Parameters
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

        print('\n\n{:=^70} \n'.format(' Electrical Conductivity '))
        coefficient = pd.DataFrame()

        if compute_acf:
            jc_acf = obs.ElectricCurrent()
            jc_acf.setup(params, phase=phase, no_slices=no_slices, **kwargs)
            jc_acf.compute()

        else:
            jc_acf = obs.ElectricCurrent()
            jc_acf.setup(params, phase=phase, no_slices=no_slices, **kwargs)
            jc_acf.parse()

        # Print some info
        TransportCoefficient.pretty_print(jc_acf, 'ElectricalConductivity')

        no_int = jc_acf.slice_steps

        # to_numpy creates a 2d-array, hence the [:,0]
        time = jc_acf.dataframe[("Time")].to_numpy()[:, 0]

        coefficient["Time"] = time
        jc_str = "Electric Current ACF"
        sigma_str = "Electrical Conductivity"
        for isl in tqdm(range(jc_acf.no_slices), disable=not jc_acf.verbose):
            sigma_ij = np.zeros(jc_acf.slice_steps)

            integrand = np.array(jc_acf.dataframe[(jc_str, "Total", "slice {}".format(isl))])

            for it in range(1, no_int):
                sigma_ij[it] = np.trapz(integrand[:it] / integrand[0], x=time[:it])

                coefficient[sigma_str + "_slice {}".format(isl)] = sigma_ij[:]

        col_str = [sigma_str + "_slice {}".format(isl) for isl in range(jc_acf.no_slices)]
        coefficient[sigma_str + "_Mean"] = coefficient[col_str].mean(axis=1)
        coefficient[sigma_str + "_Std"] = coefficient[col_str].std(axis=1)

        coefficient.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in coefficient.columns])
        coefficient.to_hdf(
            os.path.join(jc_acf.saving_dir, 'ElectricalConductivity_' + jc_acf.job_id + '.h5'),
            mode='w',
            key='conductivity',
            index=False)

        if plot or figname:
            # Make the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            ax3 = ax1.twiny()
            ax4 = ax2.twiny()
            # extra space for the second axis at the bottom
            # fig.subplots_adjust(bottom=0.2)

            acf_avg = jc_acf.dataframe[(jc_str, "Total", "Mean")]
            acf_std = jc_acf.dataframe[(jc_str, "Total", "Std")]

            d_avg = coefficient[(sigma_str, "Mean")]
            d_std = coefficient[(sigma_str, "Std")]

            # Calculate axis multipliers and labels
            xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, d_avg, "Time", "Conductivity", jc_acf.units)

            # ACF
            ax1.plot(xmul * time, acf_avg / acf_avg.iloc[0], label=r'Current $J$')
            ax1.fill_between(
                xmul * time,
                (acf_avg + acf_std) / (acf_avg.iloc[0] + acf_std.iloc[0]),
                (acf_avg - acf_std) / (acf_avg.iloc[0] - acf_std.iloc[0]), alpha=0.2)

            # Coefficient
            ax2.plot(xmul * time, ymul * d_avg, label=r'$\sigma$')
            ax2.fill_between(xmul * time, ymul * (d_avg + d_std), ymul * (d_avg - d_std), alpha=0.2)

            xlims = (xmul * time[1], xmul * time[-1] * 1.5)

            ax1.set(xlim=xlims, xscale='log', ylabel=r'Electric Current ACF', xlabel=r"Time difference" + xlbl)
            ax2.set(xlim=xlims, xscale='log', ylabel=r'Conductivity' + ylbl, xlabel=r"$\tau$" + xlbl)

            ax1.legend(loc='best')
            ax2.legend(loc='best')
            # Finish the index axes
            for axi in [ax3, ax4]:
                axi.grid(alpha=0.1)
                axi.set(xlim=(1, jc_acf.slice_steps * 1.5), xscale='log', xlabel='Index')

            fig.tight_layout()
            if figname:
                fig.savefig(os.path.join(jc_acf.saving_dir, figname))
            else:
                fig.savefig(os.path.join(jc_acf.saving_dir, 'Plot_ElectricConductivity_' + jc_acf.job_id + '.png'))

            if show:
                fig.show()

        return coefficient

    @staticmethod
    def diffusion(params,
                  phase: str = 'production',
                  compute_acf: bool = True,
                  no_slices: int = 1,
                  plot: bool = True,
                  show: bool = False,
                  figname: str = None,
                  **kwargs):
        """
        Calculate the self-diffusion coefficient from the velocity auto-correlation function.

        Parameters
        ----------
        params : sarkas.core.Parameters
            Simulation's parameters.

        phase : str, optional
            Phase to analyze. Default = 'production'.

        compute_acf : bool, optional
            Flag for recalculating the ACF. Default = True.
            If False it will read in the data from the dataframe.

        no_slices : int, optional
            Number of slices of the simulation. Default = 1.

        plot : bool, optional
            Flag to plot transport coefficient with corresponding autocorrelation function. Default = True.

        show : bool, optional
            Flag for prompting plot to screen.

        figname : str, optional
            Name with which to save the file. It automatically saves it in the correct directory.

        **kwargs:
            Arguments to pass :meth:`sarkas.tools.observables.VelocityAutoCorrelationFunction`

        Returns
        -------
        coefficient : pandas.DataFrame
            Pandas dataframe containing the value of the transport coefficient as a function of integration time.

        """
        print('\n\n{:=^70} \n'.format(' Diffusion Coefficient '))
        coefficient = pd.DataFrame()
        if compute_acf:
            vacf = obs.VelocityAutoCorrelationFunction()
            vacf.setup(params, phase=phase, no_slices=no_slices, **kwargs)
            vacf.compute()

        else:
            vacf = obs.VelocityAutoCorrelationFunction()
            vacf.setup(params, phase=phase, no_slices=no_slices, **kwargs)
            vacf.parse()

        TransportCoefficient.pretty_print(vacf, 'Diffusion')
        time = vacf.dataframe["Time"].to_numpy()[:, 0]
        coefficient["Time"] = time
        vacf_str = 'VACF'
        const = 1.0 / 3.0
        if not params.magnetized:
            # Loop over time slices
            for isl in tqdm(range(vacf.no_slices), disable = not vacf.verbose):
                # Initialize the temporary diffusion container
                D = np.zeros((params.num_species, vacf.slice_steps))
                # Iterate over the number of species
                for i, sp in enumerate(params.species_names):
                    sp_vacf_str = "{} ".format(sp) + vacf_str
                    # Grab vacf data of each slice
                    integrand = np.array(vacf.dataframe[(sp_vacf_str, "Total", "slice {}".format(isl))])
                    # Integrate each timestep
                    for it in range(1, len(time)):
                        D[i, it] = const * np.trapz(integrand[:it], x=time[:it])

                    coefficient["{} Diffusion_slice {}".format(sp, isl)] = D[i, :]

            # Average and std of each diffusion coefficient.
            for isp, sp in enumerate(params.species_names):
                col_str = ["{} Diffusion_slice {}".format(sp, isl) for isl in range(vacf.no_slices)]

                coefficient["{} Diffusion_Mean".format(sp)] = coefficient[col_str].mean(axis=1)
                coefficient["{} Diffusion_Std".format(sp)] = coefficient[col_str].std(axis=1)

        else:
            # Loop over time slices
            for isl in tqdm(range(vacf.no_slices), disable=not vacf.verbose):
                # Initialize the temporary diffusion container
                D = np.zeros((params.num_species, 2, len(time)))
                # Iterate over the number of species
                for i, sp in enumerate(params.species_names):
                    sp_vacf_str = "{} ".format(sp) + vacf_str
                    integrand_par = np.array(vacf.dataframe[(sp_vacf_str, 'Z', "slice {}".format(isl))])
                    integrand_perp = np.array(vacf.dataframe[(sp_vacf_str, 'X', "slice {}".format(isl))]) + \
                                     np.array(vacf.dataframe[(sp_vacf_str, 'Y', "slice {}".format(isl))])

                    for it in range(1, len(time)):
                        D[i, 0, it] = np.trapz(integrand_par[:it], x=time[:it])
                        D[i, 1, it] = 0.5 * np.trapz(integrand_perp[:it], x=time[:it])

                    coefficient["{} Diffusion_Parallel_slice {}".format(sp, isl)] = D[i, 0, :]
                    coefficient["{} Diffusion_Perpendicular_slice {}".format(sp, isl)] = D[i, 1, :]

            # Add the average and std of perp and par VACF to its dataframe
            for isp, sp in enumerate(params.species_names):
                sp_vacf_str = "{} ".format(sp) + vacf_str
                sp_diff_str = "{} ".format(sp) + 'Diffusion'
                par_col_str = [(sp_vacf_str, 'Z', "slice {}".format(isl)) for isl in range(vacf.no_slices)]

                vacf.dataframe[(sp_vacf_str, 'Parallel', "Mean")] = vacf.dataframe[par_col_str].mean(axis=1)
                vacf.dataframe[(sp_vacf_str, 'Parallel', "Std")] = vacf.dataframe[par_col_str].std(axis=1)

                x_col_str = [(sp_vacf_str, 'X', "slice {}".format(isl)) for isl in range(vacf.no_slices)]
                y_col_str = [(sp_vacf_str, 'Y', "slice {}".format(isl)) for isl in range(vacf.no_slices)]

                perp_vacf = 0.5 * (np.array(vacf.dataframe[x_col_str]) + np.array(vacf.dataframe[y_col_str]))
                vacf.dataframe[(sp_vacf_str, 'Perpendicular', "Mean")] = perp_vacf.mean(axis=1)
                vacf.dataframe[(sp_vacf_str, 'Perpendicular', "Std")] = perp_vacf.std(axis=1)

                # Average and std of each diffusion coefficient.
                par_col_str = [sp_diff_str + "_Parallel_slice {}".format(isl) for isl in range(vacf.no_slices)]
                perp_col_str = [sp_diff_str + "_Perpendicular_slice {}".format(isl) for isl in range(vacf.no_slices)]

                coefficient[sp_diff_str + "_Parallel_Mean"] = coefficient[par_col_str].mean(axis=1)
                coefficient[sp_diff_str + "_Parallel_Std"] = coefficient[par_col_str].std(axis=1)

                coefficient[sp_diff_str + "_Perpendicular_Mean"] = coefficient[perp_col_str].mean(axis=1)
                coefficient[sp_diff_str + "_Perpendicular_Std"] = coefficient[perp_col_str].std(axis=1)

            # TODO: Fix this hack. We should be able to add data to HDF instead of removing it and rewriting it.
            # Remove the previous hdf
            os.remove(vacf.filename_hdf)
            # Save the updated dataframe
            vacf.dataframe.to_hdf(vacf.filename_hdf, mode='w', key=vacf.__name__)

        # Endif magnetized.
        coefficient.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in coefficient.columns])
        coeff_filename = os.path.join(vacf.saving_dir, 'Diffusion_' + vacf.job_id + '.h5')
        if os.path.exists(coeff_filename):
            # Remove the previous hdf
            os.remove(vacf.filename_hdf)

        # Save the coefficient's data
        coefficient.to_hdf(coeff_filename,
                           mode='w',
                           key='diffusion')

        if plot or figname:
            # Make the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            # index axes
            ax3 = ax1.twiny()
            ax4 = ax2.twiny()
            # extra space for the second axis at the bottom
            fig.subplots_adjust(bottom=0.2)

            if not params.magnetized:
                for isp, sp in enumerate(params.species_names):
                    sp_vacf_str = "{} ".format(sp) + vacf_str
                    acf_avg = vacf.dataframe[(sp_vacf_str, "Total", "Mean")]
                    acf_std = vacf.dataframe[(sp_vacf_str, "Total", "Std")]

                    d_avg = coefficient[("{} Diffusion".format(sp), "Mean")]
                    d_std = coefficient[("{} Diffusion".format(sp), "Std")]

                    # Calculate axis multipliers and labels
                    xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, d_avg, "Time", "Diffusion", vacf.units)

                    ax1.plot(xmul * time, acf_avg / acf_avg[0], label=r'$D_{' + sp + '}$')
                    ax1.fill_between(
                        xmul * time,
                        (acf_avg + acf_std) / (acf_avg[0] + acf_std[0]),
                        (acf_avg - acf_std) / (acf_avg[0] - acf_std[0]), alpha=0.2)

                    ax2.plot(xmul * time, ymul * d_avg, label=r'$D_{' + sp + '}$')
                    ax2.fill_between(xmul * time, ymul * (d_avg + d_std), ymul * (d_avg - d_std), alpha=0.2)
            else:
                for isp, sp in enumerate(params.species_names):
                    sp_vacf_str = "{} ".format(sp) + vacf_str
                    sp_diff_str = "{} ".format(sp) + 'Diffusion'
                    par_acf_avg = vacf.dataframe[(sp_vacf_str, 'Parallel', "Mean")]
                    par_acf_std = vacf.dataframe[(sp_vacf_str, 'Parallel', "Std")]

                    par_d_avg = coefficient[(sp_diff_str, 'Parallel', "Mean")]
                    par_d_std = coefficient[(sp_diff_str, 'Parallel', "Std")]

                    perp_acf_avg = vacf.dataframe[(sp_vacf_str, 'Perpendicular', "Mean")]
                    perp_acf_std = vacf.dataframe[(sp_vacf_str, 'Perpendicular', "Std")]

                    perp_d_avg = coefficient[(sp_diff_str, 'Perpendicular', "Mean")]
                    perp_d_std = coefficient[(sp_diff_str, 'Perpendicular', "Std")]

                    # Calculate axis multipliers and labels
                    xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, perp_d_avg, "Time", "Diffusion", vacf.units)
                    # ACF Parallel plot
                    ax1.plot(xmul * time, par_acf_avg / par_acf_avg.iloc[0], label=sp + r' $Z_{\parallel}$')
                    ax1.fill_between(
                        xmul * time,
                        (par_acf_avg + par_acf_std) / (par_acf_avg.iloc[0] + par_acf_std.iloc[0]),
                        (par_acf_avg - par_acf_std) / (par_acf_avg.iloc[0] - par_acf_std.iloc[0]), alpha=0.2)
                    # Coefficient Parallel plot
                    ax2.plot(xmul * time, ymul * par_d_avg, label=sp + r' $D_{\parallel}$')
                    ax2.fill_between(xmul * time, ymul * (par_d_avg + par_d_std), ymul * (par_d_avg - par_d_std),
                                     alpha=0.2)
                    # ACF Perpendicular Plot
                    ax1.plot(xmul * time, perp_acf_avg / perp_acf_avg.iloc[0], label=sp + r' $Z_{\perp}$')
                    ax1.fill_between(
                        xmul * time,
                        (perp_acf_avg + perp_acf_std) / (perp_acf_avg.iloc[0] + perp_acf_std.iloc[0]),
                        (perp_acf_avg - perp_acf_std) / (perp_acf_avg.iloc[0] - perp_acf_std.iloc[0]), alpha=0.2)
                    # Coefficient Perpendicular Plot
                    ax2.plot(xmul * time, ymul * perp_d_avg, label=sp + r' $D_{\perp}$')
                    ax2.fill_between(xmul * time, ymul * (perp_d_avg + perp_d_std), ymul * (perp_d_avg - perp_d_std),
                                     alpha=0.2)

            xlims = (xmul * time[1], xmul * time[-1] * 1.5)

            ax1.set(xlim=xlims, xscale='log', ylabel=r'Velocity ACF', xlabel=r"Time difference" + xlbl)
            ax2.set(xlim=xlims, xscale='log', ylabel=r'Diffusion' + ylbl, xlabel=r"$\tau$" + xlbl)

            ax1.legend(loc='best')
            ax2.legend(loc='best')

            # Finish the index axes
            for axi in [ax3, ax4]:
                axi.grid(alpha=0.1)
                axi.set(xlim=[1, vacf.slice_steps * 1.5], xscale='log', xlabel='Index')

            fig.tight_layout()
            if figname:
                fig.savefig(os.path.join(vacf.saving_dir, figname))
            else:
                fig.savefig(os.path.join(vacf.saving_dir, 'Plot_Diffusion_' + vacf.job_id + '.png'))

            if show:
                fig.show()

        return coefficient

    @staticmethod
    def interdiffusion(params,
                       phase: str = 'production',
                       compute_acf: bool = True,
                       no_slices: int = 1,
                       plot: bool = True,
                       show: bool = False,
                       figname: str = None,
                       **kwargs):
        """
        Calculate the interdiffusion coefficients from the diffusion flux auto-correlation function.

        Parameters
        ----------
        params : sarkas.core.Parameters
            Simulation's parameters.

        phase : str, optional
            Phase to compute. Default = 'production'.

        compute_acf : bool
            Flag for recalculating the ACF. Default = True.
            If False it will read in the data from the dataframe.

        no_slices : int, optional
            Number of slices of the simulation. Default = 1.

        plot : bool, optional
            Flag to plot transport coefficient with corresponding autocorrelation function. Default = True.

        show : bool, optional
            Flag for prompting plot to screen when using IPython kernel. Default = False.

        figname : str, optional
            Name with which to save the file. It automatically saves it in the correct directory.

        **kwargs:
            Arguments to pass :meth:`sarkas.tools.observables.DiffusionFlux`

        Returns
        -------
        coefficient : pandas.DataFrame
            Pandas dataframe containing the value of the transport coefficient as a function of integration time.

        """
        print('\n\n{:=^70} \n'.format(' Interdiffusion Coefficient '))
        coefficient = pd.DataFrame()
        if compute_acf:
            jc_acf = obs.DiffusionFlux()
            jc_acf.setup(params, phase=phase, no_slices=no_slices, **kwargs)
            jc_acf.compute()

        else:
            jc_acf = obs.DiffusionFlux()
            jc_acf.setup(params, phase=phase, no_slices=no_slices, **kwargs)
            jc_acf.parse()

        # Print some info
        TransportCoefficient.pretty_print(jc_acf, 'Interdiffusion')
        no_int = jc_acf.slice_steps
        no_fluxes_acf = jc_acf.no_fluxes_acf
        # Normalization constant
        const = 1. / (3.0 * jc_acf.total_num_ptcls * jc_acf.species_concentrations.prod())

        # to_numpy creates a 2d-array, hence the [:,0]
        time = jc_acf.dataframe[("Time")].to_numpy()[:, 0]

        coefficient["Time"] = time
        df_str = "Diffusion Flux ACF"
        id_str = "Inter Diffusion Flux"
        for isl in tqdm(range(jc_acf.no_slices), disable=not jc_acf.verbose):
            D_ij = np.zeros((no_fluxes_acf, jc_acf.slice_steps))

            for ij in range(no_fluxes_acf):
                integrand = np.array(jc_acf.dataframe[(df_str + " {}".format(ij), "Total", "slice {}".format(isl))])

                for it in range(1, no_int):
                    D_ij[ij, it] = const * np.trapz(integrand[:it], x=time[:it])

                coefficient[id_str + " {}_slice {}".format(ij, isl)] = D_ij[ij, :]

        # Average and Std of slices
        for ij in range(no_fluxes_acf):
            col_str = [id_str + " {}_slice {}".format(ij, isl) for isl in range(jc_acf.no_slices)]
            coefficient[id_str + " {}_Mean".format(ij)] = coefficient[col_str].mean(axis=1)
            coefficient[id_str + " {}_Std".format(ij)] = coefficient[col_str].std(axis=1)

        coefficient.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in coefficient.columns])
        coefficient.to_hdf(
            os.path.join(jc_acf.saving_dir, 'InterDiffusion_' + jc_acf.job_id + '.h5'),
            mode='w',
            key='interdiffusion',
            index=False)

        if plot or figname:
            # Make the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            ax3 = ax1.twiny()
            ax4 = ax2.twiny()
            # extra space for the second axis at the bottom
            # fig.subplots_adjust(bottom=0.2)
            for flux in range(jc_acf.no_fluxes):
                acf_avg = jc_acf.dataframe[(df_str + " {}".format(flux), "Total", "Mean")]
                acf_std = jc_acf.dataframe[(df_str + " {}".format(flux), "Total", "Std")]

                d_avg = coefficient[(id_str + " {}".format(flux), "Mean")]
                d_std = coefficient[(id_str + " {}".format(flux), "Std")]

                # Calculate axis multipliers and labels
                xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, d_avg, "Time", "Diffusion", jc_acf.units)

                # ACF
                ax1.plot(xmul * time, acf_avg / acf_avg.iloc[0], label=r'Flux $J_{}$'.format(flux))
                ax1.fill_between(
                    xmul * time,
                    (acf_avg + acf_std) / (acf_avg.iloc[0] + acf_std.iloc[0]),
                    (acf_avg - acf_std) / (acf_avg.iloc[0] - acf_std.iloc[0]), alpha=0.2)

                # Coefficient
                ax2.plot(xmul * time, ymul * d_avg, label=r'IDF $D_{}$'.format(flux))
                ax2.fill_between(xmul * time, ymul * (d_avg + d_std), ymul * (d_avg - d_std), alpha=0.2)

            xlims = (xmul * time[1], xmul * time[-1] * 1.5)

            ax1.set(xlim=xlims, xscale='log', ylabel=r'Diffusion Flux ACF', xlabel=r"Time difference" + xlbl)
            ax2.set(xlim=xlims, xscale='log', ylabel=r'Inter Diffusion' + ylbl, xlabel=r"$\tau$" + xlbl)

            ax1.legend(loc='best')
            ax2.legend(loc='best')
            # Finish the index axes
            for axi in [ax3, ax4]:
                axi.grid(alpha=0.1)
                axi.set(xlim=(1, jc_acf.slice_steps * 1.5), xscale='log', xlabel='Index')

            fig.tight_layout()
            if figname:
                fig.savefig(os.path.join(jc_acf.saving_dir, figname))
            else:
                fig.savefig(os.path.join(jc_acf.saving_dir, 'Plot_InterDiffusion_' + jc_acf.job_id + '.png'))

            if show:
                fig.show()

        # Save to dataframe

        return coefficient

    @staticmethod
    def viscosity(params,
                  phase: str = 'production',
                  parse: bool = False,
                  compute_acf: bool = True,
                  no_slices: int = 1,
                  plot: bool = True,
                  plot_quantities: list = ["Bulk Viscosity"],
                  show: bool = False,
                  figname: str = None,
                  **kwargs
                  ):
        """
        Calculate bulk and shear viscosity from pressure auto-correlation function.

        Parameters
        ----------
        params : sarkas.core.Parameters
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
        energies.parse()

        pt = obs.PressureTensor()
        pt.setup(params, phase, no_slices, **kwargs)
        if parse:
            pt.parse()
            time = pt.dataframe["Time"].to_numpy()[:, 0]
            coefficient = pd.read_hdf(os.path.join(pt.saving_dir, 'Viscosities_' + pt.job_id + '.h5'),
                                      mode='r',
                                      key='viscosities',
                                      index=False)
        else:
            if compute_acf:

                pt.compute()
            else:
                pt.parse()

            TransportCoefficient.pretty_print(pt, 'Viscosities')

            time = pt.dataframe["Time"].to_numpy()[:, 0]
            coefficient["Time"] = time
            dim_lbl = ['x', 'y', 'z']

            pt_str_kin = "Pressure Tensor Kinetic ACF"
            pt_str_pot = "Pressure Tensor Potential ACF"
            pt_str_kinpot = "Pressure Tensor Kin-Pot ACF"
            pt_str_potkin = "Pressure Tensor Pot-Kin ACF"
            pt_str = "Pressure Tensor ACF"

            start_steps = 0
            end_steps = 0

            for isl in tqdm(range(pt.no_slices), disable=not pt.verbose):
                end_steps += pt.slice_steps

                beta = (np.mean(energies.dataframe["Temperature"].iloc[start_steps:end_steps]) * pt.kB) ** (-1.0)
                const = params.box_volume * beta
                # Calculate Bulk Viscosity
                bulk_viscosity = np.zeros(pt.slice_steps)
                # Bulk viscosity is calculated from the fluctuations of the pressure eq. 2.124a Allen & Tilsdeley
                bulk_integrand = np.copy(pt.dataframe[("Delta Pressure ACF", "slice {}".format(isl))])

                for it in range(1, pt.slice_steps):
                    bulk_viscosity[it] = const * np.trapz(bulk_integrand[:it], x=time[:it])

                coefficient["Bulk Viscosity_slice {}".format(isl)] = bulk_viscosity

                shear_kinetic = np.zeros((params.dimensions, params.dimensions, pt.slice_steps))
                shear_potential = np.zeros((params.dimensions, params.dimensions, pt.slice_steps))
                shear_kinpot = np.zeros((params.dimensions, params.dimensions, pt.slice_steps))
                shear_potkin = np.zeros((params.dimensions, params.dimensions, pt.slice_steps))

                shear_viscosity = np.zeros((params.dimensions, params.dimensions, pt.slice_steps))

                for i, ax1 in enumerate(dim_lbl):
                    for j, ax2 in enumerate(dim_lbl):

                        # Kinetic terms
                        integrand = pt.dataframe[
                            (pt_str_kin + " {}{}{}{}".format(ax1, ax2, ax1, ax2), 'slice {}'.format(isl))].to_numpy()
                        for it in range(1, pt.slice_steps):
                            shear_kinetic[i, j, it] = const * np.trapz(integrand[:it], x=time[:it])
                        coefficient[
                            "Kinetic Shear Viscosity Tensor {}{}_slice {}".format(ax1, ax2, isl)] = shear_kinetic[i, j]

                        # Potential terms
                        integrand = pt.dataframe[
                            (pt_str_pot + " {}{}{}{}".format(ax1, ax2, ax1, ax2), 'slice {}'.format(isl))].to_numpy()
                        for it in range(1, pt.slice_steps):
                            shear_potential[i, j, it] = const * np.trapz(integrand[:it], x=time[:it])
                        coefficient[
                            "Potential Shear Viscosity Tensor {}{}_slice {}".format(ax1, ax2, isl)] = shear_potential[
                            i, j]

                        # Kinetic-Potential terms
                        integrand = pt.dataframe[
                            (pt_str_kinpot + " {}{}{}{}".format(ax1, ax2, ax1, ax2), 'slice {}'.format(isl))].to_numpy()
                        for it in range(1, pt.slice_steps):
                            shear_kinpot[i, j, it] = const * np.trapz(integrand[:it], x=time[:it])
                        coefficient[
                            "Kin-Pot Shear Viscosity Tensor {}{}_slice {}".format(ax1, ax2, isl)] = shear_kinpot[i, j]

                        # Potential-Kinetic terms
                        integrand = pt.dataframe[
                            (pt_str_potkin + " {}{}{}{}".format(ax1, ax2, ax1, ax2), 'slice {}'.format(isl))].to_numpy()
                        for it in range(1, pt.slice_steps):
                            shear_potkin[i, j, it] = const * np.trapz(integrand[:it], x=time[:it])
                        coefficient[
                            "Pot-Kin Shear Viscosity Tensor {}{}_slice {}".format(ax1, ax2, isl)] = shear_potkin[i, j]

                        # Full terms
                        integrand = pt.dataframe[
                            (pt_str + " {}{}{}{}".format(ax1, ax2, ax1, ax2), 'slice {}'.format(isl))].to_numpy()
                        for it in range(1, pt.slice_steps):
                            shear_viscosity[i, j, it] = const * np.trapz(integrand[:it], x=time[:it])
                        coefficient[
                            "Shear Viscosity Tensor {}{}_slice {}".format(ax1, ax2, isl)] = shear_viscosity[i, j]

            start_steps += pt.slice_steps

            # Now average the slice
            col_str = ["Bulk Viscosity_slice {}".format(isl) for isl in range(pt.no_slices)]
            coefficient["Bulk Viscosity_Mean"] = coefficient[col_str].mean(axis=1)
            coefficient["Bulk Viscosity_Std"] = coefficient[col_str].std(axis=1)

            kin_str = "Kinetic Shear Viscosity Tensor"
            pot_str = "Potential Shear Viscosity Tensor"
            kp_str = "Kin-Pot Shear Viscosity Tensor"
            pk_str = "Pot-Kin Shear Viscosity Tensor"
            sv_str = "Shear Viscosity Tensor"

            for i, ax1 in enumerate(dim_lbl):
                for j, ax2 in enumerate(dim_lbl):
                    # Kinetic Terms
                    col_str = [kin_str + " {}{}_slice {}".format(ax1, ax2, isl) for isl in range(pt.no_slices)]
                    coefficient[kin_str + " {}{}_Mean".format(ax1, ax2)] = coefficient[col_str].mean(axis=1)
                    coefficient[kin_str + " {}{}_Std".format(ax1, ax2)] = coefficient[col_str].std(axis=1)
                    # Potential Terms
                    col_str = [pot_str + " {}{}_slice {}".format(ax1, ax2, isl) for isl in range(pt.no_slices)]
                    coefficient[pot_str + " {}{}_Mean".format(ax1, ax2)] = coefficient[col_str].mean(axis=1)
                    coefficient[pot_str + " {}{}_Std".format(ax1, ax2)] = coefficient[col_str].std(axis=1)
                    # Kin-Pot Terms
                    col_str = [kp_str + " {}{}_slice {}".format(ax1, ax2, isl) for isl in range(pt.no_slices)]
                    coefficient[kp_str + " {}{}_Mean".format(ax1, ax2)] = coefficient[col_str].mean(axis=1)
                    coefficient[kp_str + " {}{}_Std".format(ax1, ax2)] = coefficient[col_str].std(axis=1)
                    # PotKin Terms
                    col_str = [pk_str + " {}{}_slice {}".format(ax1, ax2, isl) for isl in range(pt.no_slices)]
                    coefficient[pk_str + " {}{}_Mean".format(ax1, ax2)] = coefficient[col_str].mean(axis=1)
                    coefficient[pk_str + " {}{}_Std".format(ax1, ax2)] = coefficient[col_str].std(axis=1)
                    # Full
                    col_str = [sv_str + " {}{}_slice {}".format(ax1, ax2, isl) for isl in range(pt.no_slices)]
                    coefficient[sv_str + " {}{}_Mean".format(ax1, ax2)] = coefficient[col_str].mean(axis=1)
                    coefficient[sv_str + " {}{}_Std".format(ax1, ax2)] = coefficient[col_str].std(axis=1)

            list_coord = ['xy', 'xz', 'yx', 'yz', 'zx', 'zy']
            col_str = [sv_str + " {}_Mean".format(coord) for coord in list_coord]
            coefficient["Shear Viscosity_Mean"] = coefficient[col_str].mean(axis=1)
            coefficient["Shear Viscosity_Std"] = coefficient[col_str].std(axis=1)

            coefficient.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in coefficient.columns])
            coefficient.to_hdf(
                os.path.join(pt.saving_dir, 'Viscosities_' + pt.job_id + '.h5'),
                mode='w',
                key='viscosities',
                index=False)

        if plot or figname:
            # Make the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            ax3 = ax1.twiny()
            ax4 = ax2.twiny()
            # extra space for the second axis at the bottom
            # fig.subplots_adjust(bottom=0.2)
            for ipq, pq in enumerate(plot_quantities):
                if pq[:4].lower() == "bulk":
                    acf_str = "Delta Pressure ACF"
                    acf_avg = pt.dataframe[("Delta Pressure ACF", "Mean")]
                    acf_std = pt.dataframe[("Delta Pressure ACF", "Std")]
                else:
                    # The axis are the last two elements in the string
                    acf_str = "Pressure Tensor ACF " + pq[-2:]
                    acf_avg = pt.dataframe[("Pressure Tensor ACF " + pq[-2:], "Mean")]
                    acf_std = pt.dataframe[("Pressure Tensor ACF " + pq[-2:], "Std")]

                d_avg = coefficient[(pq, "Mean")]
                d_std = coefficient[(pq, "Std")]

                # Calculate axis multipliers and labels
                xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, d_avg, "Time", "Viscosity", pt.units)

                # ACF
                ax1.plot(xmul * time, acf_avg / acf_avg.iloc[0], label=acf_str)
                ax1.fill_between(
                    xmul * time,
                    (acf_avg + acf_std) / (acf_avg.iloc[0] + acf_std.iloc[0]),
                    (acf_avg - acf_std) / (acf_avg.iloc[0] - acf_std.iloc[0]), alpha=0.2)

                # Coefficient
                ax2.plot(xmul * time, ymul * d_avg, label=r'$\eta$')
                ax2.fill_between(xmul * time, ymul * (d_avg + d_std), ymul * (d_avg - d_std), alpha=0.2)

                xlims = (xmul * time[1], xmul * time[-1] * 1.5)

                ax1.set(xlim=xlims, xscale='log', ylabel=acf_str, xlabel=r"Time difference" + xlbl)
                ax2.set(xlim=xlims, xscale='log', ylabel=r'Viscosity' + ylbl, xlabel=r"$\tau$" + xlbl)

                ax1.legend(loc='best')
                ax2.legend(loc='best')
                # Finish the index axes
                for axi in [ax3, ax4]:
                    axi.grid(alpha=0.1)
                    axi.set(xlim=(1, pt.slice_steps * 1.5), xscale='log', xlabel='Index')

                fig.tight_layout()
                if figname:
                    fig.savefig(os.path.join(pt.saving_dir, figname))
                else:
                    fig.savefig(os.path.join(pt.saving_dir, 'Plot_' + pq.replace(" ", "") + '_' + pt.job_id + '.png'))

                if show:
                    fig.show()

        return coefficient
