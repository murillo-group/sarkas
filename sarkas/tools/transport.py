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
        print('Data saved in: \n', os.path.join(jc_acf.saving_dir, 'ElectricalConductivity_' + jc_acf.job_id + '.h5'))

        print('\nNo. of slices = {}'.format(jc_acf.no_slices))
        print('No. dumps per slice = {}'.format( int(jc_acf.slice_steps/jc_acf.dump_step) ))

        print('Time interval of autocorrelation function = {:.4e} [s] ~ {} w_p T'.format(
            jc_acf.dt * jc_acf.slice_steps,
            int(jc_acf.dt * jc_acf.slice_steps * jc_acf.total_plasma_frequency)))

        no_int = jc_acf.slice_steps

        # to_numpy creates a 2d-array, hence the [:,0]
        time = jc_acf.dataframe[("Time")].to_numpy()[:, 0]

        coefficient["Time"] = time
        jc_str = "Electric Current ACF"
        sigma_str = "Electrical Conductivity"
        for isl in range(jc_acf.no_slices):
            sigma_ij = np.zeros(jc_acf.slice_steps)

            integrand = np.array(jc_acf.dataframe[(jc_str, "Total", "slice {}".format(isl))])

            for it in range(1, no_int):
                sigma_ij[it] = np.trapz(integrand[:it]/integrand[0], x=time[:it])

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

        print('Data saved in: \n', os.path.join(vacf.saving_dir, 'Diffusion_' + vacf.job_id + '.h5'))

        print('\nNo. of slices = {}'.format(vacf.no_slices))
        print('No. dumps per slice = {}'.format(int(vacf.slice_steps/vacf.dump_step)))

        print('Time interval of autocorrelation function = {:.4e} [s] ~ {} w_p T'.format(
            vacf.dt * vacf.slice_steps,
            int(vacf.dt * vacf.slice_steps * vacf.total_plasma_frequency)))

        vacf.compute()
        time = vacf.dataframe["Time"].to_numpy()[:,0]
        coefficient["Time"] = time
        vacf_str = 'VACF'
        const = 1.0 / 3.0
        if not params.magnetized:
            # Loop over time slices
            for isl in range(vacf.no_slices):
                # Initialize the temporary diffusion container
                D = np.zeros((params.num_species, vacf.slice_steps))
                # Iterate over the number of species
                for i, sp in enumerate(params.species_names):
                    sp_vacf_str = "{} ".format(sp) + vacf_str
                    # Grab vacf data of each slice
                    integrand = np.array(vacf.dataframe[(sp_vacf_str, "Total","slice {}".format(isl) )])
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
            for isl in range(vacf.no_slices):
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

                    coefficient["{} Parallel Diffusion_slice {}".format(sp, isl)] = D[i, 0, :]
                    coefficient["{} Perpendicular Diffusion_slice {}".format(sp, isl)] = D[i, 1, :]

            # Add the average and std of perp and par VACF to its dataframe
            for isp, sp in enumerate(params.species_names):
                par_col_str = ["{} Z Velocity ACF slice {}".format(sp, isl) for isl in range(vacf.no_slices)]

                vacf.dataframe["{} Parallel Velocity ACF avg".format(sp)] = vacf.dataframe[par_col_str].mean(axis=1)
                vacf.dataframe["{} Parallel Velocity ACF std".format(sp)] = vacf.dataframe[par_col_str].std(axis=1)

                x_col_str = ["{} X Velocity ACF slice {}".format(sp, isl) for isl in range(vacf.no_slices)]
                y_col_str = ["{} Y Velocity ACF slice {}".format(sp, isl) for isl in range(vacf.no_slices)]

                perp_vacf = 0.5 * (np.array(vacf.dataframe[x_col_str]) + np.array(vacf.dataframe[y_col_str]))
                vacf.dataframe["{} Perpendicular Velocity ACF avg".format(sp)] = perp_vacf.mean(axis=1)
                vacf.dataframe["{} Perpendicular Velocity ACF std".format(sp)] = perp_vacf.std(axis=1)

                # Average and std of each diffusion coefficient.
                par_col_str = ["{} Parallel Diffusion slice {}".format(sp, isl) for isl in range(vacf.no_slices)]
                perp_col_str = ["{} Perpendicular Diffusion slice {}".format(sp, isl) for isl in range(vacf.no_slices)]

                coefficient["{} Parallel Diffusion avg".format(sp)] = coefficient[par_col_str].mean(axis=1)
                coefficient["{} Parallel Diffusion std".format(sp)] = coefficient[par_col_str].std(axis=1)

                coefficient["{} Perpendicular Diffusion avg".format(sp)] = coefficient[perp_col_str].mean(axis=1)
                coefficient["{} Perpendicular Diffusion std".format(sp)] = coefficient[perp_col_str].std(axis=1)

            # Save the updated dataframe
            vacf.dataframe.to_csv(vacf.filename_csv, index=False, encoding='utf-8')

        # Endif magnetized.
        coefficient.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in coefficient.columns])
        # Save the coefficient's data
        coefficient.to_hdf(
            os.path.join(vacf.saving_dir, 'Diffusion_' + vacf.job_id + '.h5'),
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

                    d_avg = coefficient[("{} Diffusion".format(sp),"Mean")]
                    d_std = coefficient[("{} Diffusion".format(sp),"Std")]

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
                    par_acf_avg = vacf.dataframe["{} Parallel Velocity ACF avg".format(sp)]
                    par_acf_std = vacf.dataframe["{} Parallel Velocity ACF std".format(sp)]

                    par_d_avg = coefficient["{} Parallel Diffusion avg".format(sp)]
                    par_d_std = coefficient["{} Parallel Diffusion std".format(sp)]

                    perp_acf_avg = vacf.dataframe["{} Perpendicular Velocity ACF avg".format(sp)]
                    perp_acf_std = vacf.dataframe["{} Perpendicular Velocity ACF std".format(sp)]

                    perp_d_avg = coefficient["{} Perpendicular Diffusion avg".format(sp)]
                    perp_d_std = coefficient["{} Perpendicular Diffusion std".format(sp)]

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
        print('Data saved in: \n', os.path.join(jc_acf.saving_dir, 'InterDiffusion_' + jc_acf.job_id + '.h5'))

        print('\nNo. of slices = {}'.format(jc_acf.no_slices))
        print('No. dumps per slice = {}'.format(int(jc_acf.slice_steps/jc_acf.dump_step) ))

        print('Time interval of autocorrelation function = {:.4e} [s] ~ {} w_p T'.format(
            jc_acf.dt * jc_acf.slice_steps * jc_acf.dump_step,
            int(jc_acf.dt * jc_acf.slice_steps * jc_acf.dump_step * jc_acf.total_plasma_frequency)))

        no_int = jc_acf.slice_steps
        no_fluxes_acf = jc_acf.no_fluxes_acf
        # Normalization constant
        const = 1. / (3.0 * jc_acf.total_num_ptcls * jc_acf.species_concentrations.prod())

        # to_numpy creates a 2d-array, hence the [:,0]
        time = jc_acf.dataframe[("Time")].to_numpy()[:, 0]

        coefficient["Time"] = time
        df_str = "Diffusion Flux ACF"
        id_str = "Inter Diffusion Flux"
        for isl in range(jc_acf.no_slices):
            D_ij = np.zeros((no_fluxes_acf, jc_acf.slice_steps))

            for ij in range(no_fluxes_acf):
                integrand = np.array(jc_acf.dataframe[(df_str + " {}".format(ij), "Total", "slice {}".format(isl))])

                for it in range(1, no_int):
                    D_ij[ij, it] = const * np.trapz(integrand[:it], x=time[:it])

                coefficient[id_str + " {}_slice {}".format(ij, isl)] = D_ij[ij, :]

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
    def viscosity(params, phase: str = 'production', show=False):
        """
        TODO: Calculate bulk and shear viscosity from pressure auto-correlation function.

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
