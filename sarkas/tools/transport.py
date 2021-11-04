"""
Transport Module.
"""
from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sarkas Modules
import sarkas.tools.observables as obs
from sarkas.utilities.maths import fast_integral_loop


class TransportCoefficients:
    """
    Transport Coefficients class.
    """

    def __init__(self, params, phase: str = "production", no_slices: int = 1):
        self.__name__ = "TC"
        self.__long_name__ = "Transport Coefficients"
        self.tc_names = ["ElectricalConductivity", "Diffusion", "Interdiffusion", "Viscosities"]
        # Parameters copies
        self.postprocessing_dir = params.postprocessing_dir
        self.units = params.units
        self.job_id = params.job_id
        self.verbose = params.verbose
        self.dt = params.dt
        self.total_plasma_frequency = params.total_plasma_frequency
        self.box_volume = params.box_volume
        self.pbox_volume = params.pbox_volume
        #
        self.saving_dir = None
        self.phase = phase
        self.no_slices = no_slices
        # Calculate the average temperature from the csv data
        self.kB = params.kB
        energy = obs.Thermodynamics()
        energy.setup(params, self.phase)
        energy.parse()
        self.T_avg = energy.dataframe["Temperature"].mean()
        self.beta = 1.0 / (self.kB * self.T_avg)

        self.diffusion_df = None
        self.diffusion_df_slices = None
        self.interdiffusion_df = None
        self.interdiffusion_df_slices = None
        self.viscosity_df = None
        self.viscosity_df_slices = None
        self.conductivity_df = None
        self.conductivity_df_slices = None

        self.create_dir()

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Transport( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def create_dir(self):
        """Create directory where to save the transport coefficients."""

        saving_dir = os.path.join(self.postprocessing_dir, self.__long_name__.replace(" ", ""))
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

    def save_hdf(self, df: pd.DataFrame = None, df_slices: pd.DataFrame = None, tc_name: str = None):
        """
        Save the HDF dataframes to disk in the TransportCoefficient folder.

        Parameters
        ----------
        df: pandas.DataFrame()
            Multi-index dataframe containing the mean and std of the transport coefficient.

        df_slices: pandas.DataFrame()
            Multi-index dataframe containing the transport coefficient of each slice.

        tc_name: str
            Name of the transport coefficient. See :attr:`~.tc_names` for options.

        Returns
        -------
        df: pandas.DataFrame()
            Sorted multi-index dataframe containing the mean and std of the transport coefficient.

        df_slices: pandas.DataFrame()
            Sorted multi-index dataframe containing the transport coefficient of each slice.

        """

        df_slices.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in df_slices.columns])
        df_slices = df_slices.sort_index()
        df_slices.to_hdf(
            os.path.join(self.saving_dir, tc_name + "_slices_" + self.job_id + ".h5"), mode="w", key=tc_name, index=False
        )

        df.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in df.columns])
        df = df.sort_index()
        df.to_hdf(os.path.join(self.saving_dir, tc_name + "_" + self.job_id + ".h5"), mode="w", key=tc_name, index=False)

        return df, df_slices

    def parse(self, observable, tc_name):
        """Read the HDF files containing the transport coefficients.

        Parameters
        ----------
        observable: sarkas.tools.observables.ElectricCurrent
            Observable object containing the ACF whose time integral leads to the electrical conductivity.

        tc_name: str
            Transport Coefficient name.

        Raises
        ------
        ValueError
            Wrong ``tc_name``.

        """
        if tc_name in self.tc_names:

            if tc_name == "ElectricalConductivity":
                self.conductivity_df = pd.read_hdf(
                    os.path.join(self.saving_dir, tc_name + "_" + self.job_id + ".h5"), mode="r", index_col=False
                )

                self.conductivity_df_slices = pd.read_hdf(
                    os.path.join(self.saving_dir, tc_name + "_slices_" + self.job_id + ".h5"), mode="r", index_col=False
                )

            elif tc_name == "Diffusion":
                self.diffusion_df = pd.read_hdf(
                    os.path.join(self.saving_dir, tc_name + "_" + self.job_id + ".h5"), mode="r", index_col=False
                )

                self.diffusion_df_slices = pd.read_hdf(
                    os.path.join(self.saving_dir, tc_name + "_slices_" + self.job_id + ".h5"), mode="r", index_col=False
                )

            elif tc_name == "Interdiffusion":

                self.interdiffusion_df = pd.read_hdf(
                    os.path.join(self.saving_dir, tc_name + "_" + self.job_id + ".h5"), mode="r", index_col=False
                )

                self.interdiffusion_df_slices = pd.read_hdf(
                    os.path.join(self.saving_dir, tc_name + "_slices_" + self.job_id + ".h5"), mode="r", index_col=False
                )

            elif tc_name == "Viscosities":

                self.viscosity_df = pd.read_hdf(
                    os.path.join(self.saving_dir, tc_name + "_" + self.job_id + ".h5"), mode="r", index_col=False
                )

                self.viscosity_df_slices = pd.read_hdf(
                    os.path.join(self.saving_dir, tc_name + "_slices_" + self.job_id + ".h5"), mode="r", index_col=False
                )
        else:
            raise ValueError("Wrong tc_name. Please choose amongst ", self.tc_names)

        self.phase = observable.phase
        self.no_slices = observable.no_slices
        self.slice_steps = observable.slice_steps
        self.dump_step = observable.dump_step
        # Print some info
        self.pretty_print(tc_name=tc_name)

    def plot_tc(self, time, acf_data, tc_data, acf_name, tc_name, figname, show: bool = False):
        """
        Make dual plots with ACF and transport coefficient.

        Parameters
        ----------
        acf_data: numpy.ndarray
            Mean and Std of the ACF. \n
            Shape = (:attr:`sarkas.tools.observables.Observable.slice_steps`, 2).

        tc_data: numpy.ndarray
            Mean and Std of the transport coefficient. \n
            Shape = (:attr:`sarkas.tools.observables.Observable.slice_steps`, 2).

        acf_name: str
            y-Label of the ACF plot.

        tc_name: str
            y-label of the transport coefficient plot.

        figname: str
            Name with which to save the plot.

        show: bool
            Flag for displaying the plot if using IPython or terminal.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object.

        (ax1, ax2, ax3, ax4) : tuple
            Tuple with the axes handles. \n
            ax1 = ACF axes, ax2 = transport coefficient axes, ax3 = ax1.twiny(), ax4 = ax2.twiny()

        """
        # Make the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        ax3 = ax1.twiny()
        ax4 = ax2.twiny()

        # Calculate axis multipliers and labels
        xmul, ymul, _, _, xlbl, ylbl = obs.plot_labels(time, tc_data[:, 0], "Time", tc_name, self.units)

        # ACF
        ax1.plot(xmul * time, acf_data[:, 0] / acf_data[0, 0])
        ax1.fill_between(
            xmul * time,
            (acf_data[:, 0] + acf_data[:, 1]) / (acf_data[0, 0] + acf_data[0, 1]),
            (acf_data[:, 0] - acf_data[:, 1]) / (acf_data[0, 0] - acf_data[0, 1]),
            alpha=0.2,
        )

        # Coefficient
        ax2.plot(xmul * time, ymul * tc_data[:, 0])
        ax2.fill_between(
            xmul * time, ymul * (tc_data[:, 0] + tc_data[:, 1]), ymul * (tc_data[:, 0] - tc_data[:, 1]), alpha=0.2
        )

        xlims = (xmul * time[1], xmul * time[-1] * 1.5)

        ax1.set(xlim=xlims, xscale="log", ylabel=acf_name, xlabel=r"Time difference" + xlbl)
        ax2.set(xlim=xlims, xscale="log", ylabel=tc_name + ylbl, xlabel=r"$\tau$" + xlbl)

        # ax1.legend(loc='best')
        # ax2.legend(loc='best')
        # Finish the index axes
        for axi in [ax3, ax4]:
            axi.grid(alpha=0.1)
            axi.set(xlim=(1, self.slice_steps * 1.5), xscale="log", xlabel="Index")

        fig.tight_layout()
        fig.savefig(os.path.join(self.saving_dir, figname))

        if show:
            fig.show()

        return fig, (ax1, ax2, ax3, ax4)

    def pretty_print(self, tc_name):
        """
        Print to screen the location where data is stored and other relevant information.

        Parameters
        ----------
        observable: sarkas.tools.observables.Observable
            Physical quantity of the ACF.

        tc_name: str
            Name of Transport coefficient to calculate.
        """

        print("Data saved in: \n", os.path.join(self.saving_dir, tc_name + "_" + self.job_id + ".h5"))
        print(os.path.join(self.saving_dir, tc_name + "_slices_" + self.job_id + ".h5"))
        print("\nNo. of slices = {}".format(self.no_slices))
        print("No. dumps per slice = {}".format(int(self.slice_steps / self.dump_step)))

        print(
            "Time interval of autocorrelation function = {:.4e} [s] ~ {} w_p T".format(
                self.dt * self.slice_steps * self.dump_step,
                int(self.dt * self.slice_steps * self.dump_step * self.total_plasma_frequency),
            )
        )

    def electrical_conductivity(self, observable, plot: bool = True, display_plot: bool = False):
        """
        Calculate the transport coefficient from the Green-Kubo formula

        .. math::

            \\sigma = \\frac{\\beta}{V} \\int_0^{\\tau} dt J(t).

        where :math:`\\beta = 1/k_B T` and :math:`V` is the volum of the simulation box.

        Data is retrievable at :attr:`~.conductivity_df` and :attr:`~.conductivity_df_slices`.

        Parameters
        ----------
        observable: sarkas.tools.observables.ElectricCurrent
            Observable object containing the ACF whose time integral leads to the electrical conductivity.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. \n
            Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.

        """

        print("\n\n{:=^70} \n".format(" Electrical Conductivity "))
        self.conductivity_df = pd.DataFrame()
        self.conductivity_df_slices = pd.DataFrame()

        # Check that the phase and no_slices is the same from the one computed in the Observable
        observable.parse()

        self.phase = observable.phase
        self.no_slices = observable.no_slices
        self.slice_steps = observable.slice_steps
        self.dump_step = observable.dump_step
        # Print some info
        self.pretty_print(tc_name="ElectricalConductivity")

        # to_numpy creates a 2d-array, hence the [:,0]
        time = observable.dataframe_acf["Time"].iloc[:, 0].to_numpy()

        self.conductivity_df["Time"] = np.copy(time)
        self.conductivity_df_slices["Time"] = np.copy(time)

        jc_str = "Electric Current ACF"
        sigma_str = "Electrical Conductivity"
        const = self.beta / observable.box_volume
        if not observable.magnetized:
            for isl in tqdm(range(observable.no_slices), disable=not observable.verbose):
                integrand = np.array(observable.dataframe_acf_slices[(jc_str, "Total", "slice {}".format(isl))])
                self.conductivity_df_slices[sigma_str + "_slice {}".format(isl)] = const * fast_integral_loop(
                    time, integrand
                )

            col_str = [sigma_str + "_slice {}".format(isl) for isl in range(observable.no_slices)]
            self.conductivity_df[sigma_str + "_Mean"] = self.conductivity_df_slices[col_str].mean(axis=1)
            self.conductivity_df[sigma_str + "_Std"] = self.conductivity_df_slices[col_str].std(axis=1)
        else:

            for isl in tqdm(range(observable.no_slices), disable=not observable.verbose):
                # Parallel
                par_str = (jc_str, "Z", "slice {}".format(isl))
                sigma_par_str = sigma_str + "_Parallel_slice {}".format(isl)
                integrand = observable.dataframe_acf_slices[par_str].to_numpy()
                self.conductivity_df_slices[sigma_par_str] = const * fast_integral_loop(time, integrand)
                # Perpendicular
                x_col_str = (jc_str, "X", "slice {}".format(isl))
                y_col_str = (jc_str, "Y", "slice {}".format(isl))
                perp_integrand = 0.5 * (
                    observable.dataframe_acf_slices[x_col_str].to_numpy()
                    + observable.dataframe_acf_slices[y_col_str].to_numpy()
                )
                sigma_perp_str = sigma_str + "_Perpendicular_slice {}".format(isl)
                self.conductivity_df_slices[sigma_perp_str] = const * fast_integral_loop(time, perp_integrand)

            par_col_str = [(jc_str, "Z", "slice {}".format(isl)) for isl in range(self.no_slices)]
            observable.dataframe_acf[(jc_str, "Parallel", "Mean")] = observable.dataframe_acf_slices[par_col_str].mean(
                axis=1
            )
            observable.dataframe_acf[(jc_str, "Parallel", "Std")] = observable.dataframe_acf_slices[par_col_str].std(
                axis=1
            )

            x_col_str = [(jc_str, "X", "slice {}".format(isl)) for isl in range(self.no_slices)]
            y_col_str = [(jc_str, "Y", "slice {}".format(isl)) for isl in range(self.no_slices)]

            perp_jc = 0.5 * (
                observable.dataframe_acf_slices[x_col_str].to_numpy()
                + observable.dataframe_acf_slices[y_col_str].to_numpy()
            )
            observable.dataframe_acf[(jc_str, "Perpendicular", "Mean")] = perp_jc.mean(axis=1)
            observable.dataframe_acf[(jc_str, "Perpendicular", "Std")] = perp_jc.std(axis=1)
            # Save the updated dataframe
            observable.save_hdf()

            # Average and std of transport coefficient.
            col_str = [sigma_str + "_Parallel_slice {}".format(isl) for isl in range(observable.no_slices)]
            self.conductivity_df[sigma_str + "_Parallel_Mean"] = self.conductivity_df_slices[col_str].mean(axis=1)
            self.conductivity_df[sigma_str + "_Parallel_Std"] = self.conductivity_df_slices[col_str].std(axis=1)
            # Perpendicular
            col_str = [sigma_str + "_Perpendicular_slice {}".format(isl) for isl in range(observable.no_slices)]
            self.conductivity_df[sigma_str + "_Perpendicular_Mean"] = self.conductivity_df_slices[col_str].mean(axis=1)
            self.conductivity_df[sigma_str + "_Perpendicular_Std"] = self.conductivity_df_slices[col_str].std(axis=1)

            # Endif magnetized.

        self.conductivity_df, self.conductivity_df_slices = self.save_hdf(
            df=self.conductivity_df, df_slices=self.conductivity_df_slices, tc_name="ElectricalConductivity"
        )

        if plot or display_plot:

            if not observable.magnetized:
                acf_avg = observable.dataframe_acf[(jc_str, "Total", "Mean")].to_numpy()
                acf_std = observable.dataframe_acf[(jc_str, "Total", "Std")].to_numpy()

                tc_avg = self.conductivity_df[(sigma_str, "Mean")].to_numpy()
                tc_std = self.conductivity_df[(sigma_str, "Std")].to_numpy()

                self.plot_tc(
                    time=time,
                    acf_data=np.column_stack((acf_avg, acf_std)),
                    tc_data=np.column_stack((tc_avg, tc_std)),
                    acf_name="Electric Current ACF",
                    tc_name="Electrical Conductivity",
                    figname="ElectricalConductivity_Plot.png",
                    show=display_plot,
                )
            else:

                acf_avg = observable.dataframe_acf[(jc_str, "Parallel", "Mean")].to_numpy()
                acf_std = observable.dataframe_acf[(jc_str, "Parallel", "Std")].to_numpy()

                tc_avg = self.conductivity_df[(sigma_str, "Parallel", "Mean")].to_numpy()
                tc_std = self.conductivity_df[(sigma_str, "Parallel", "Std")].to_numpy()

                self.plot_tc(
                    time=time,
                    acf_data=np.column_stack((acf_avg, acf_std)),
                    tc_data=np.column_stack((tc_avg, tc_std)),
                    acf_name="Electric Current ACF Parallel",
                    tc_name="Electrical Conductivity Parallel",
                    figname="ElectricalConductivity_Parallel_Plot.png",
                    show=display_plot,
                )

                acf_avg = observable.dataframe_acf[(jc_str, "Perpendicular", "Mean")].to_numpy()
                acf_std = observable.dataframe_acf[(jc_str, "Perpendicular", "Std")].to_numpy()

                tc_avg = self.conductivity_df[(sigma_str, "Perpendicular", "Mean")].to_numpy()
                tc_std = self.conductivity_df[(sigma_str, "Perpendicular", "Std")].to_numpy()

                self.plot_tc(
                    time=time,
                    acf_data=np.column_stack((acf_avg, acf_std)),
                    tc_data=np.column_stack((tc_avg, tc_std)),
                    acf_name="Electric Current ACF Perpendicular",
                    tc_name="Electrical Conductivity Perpendicular",
                    figname="ElectricalConductivity_Perpendicular_Plot.png",
                    show=display_plot,
                )

    def diffusion(
        self, observable, plot: bool = True, display_plot: bool = False,
    ):
        """
        Calculate the transport coefficient from the Green-Kubo formula

        .. math::

            D_{\\alpha} = \\frac{1}{3 N_{\\alpha}} \\sum_{i}^{N_{\\alpha}} \\int_0^{\\tau} dt \\,
                \\langle \\mathbf v^{(\\alpha)}_{i}(t) \\cdot  \\mathbf v^{(\\alpha)}_{i}(0) \\rangle.

        where :math:`\\mathbf v_{i}^{(\\alpha)}(t)` is the velocity of particle :math:`i` of species
        :math:`\\alpha`. Notice that the diffusion coefficient is averaged over all :math:`N_{\\alpha}` particles.

        Data is retrievable at :attr:`~.diffusion_df` and
        :attr:`~.diffusion_df_slices`.

        Parameters
        ----------
        observable: sarkas.tools.observables.ElectricCurrent
            Observable object containing the ACF whose time integral leads to the electrical conductivity.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.

        """
        print("\n\n{:=^70} \n".format(" Diffusion Coefficient "))
        self.diffusion_df = pd.DataFrame()
        self.diffusion_df_slices = pd.DataFrame()

        # Check that the phase and no_slices is the same from the one computed in the Observable
        observable.parse()

        self.phase = observable.phase
        self.no_slices = observable.no_slices
        self.slice_steps = observable.slice_steps
        self.dump_step = observable.dump_step
        # Print some info
        self.pretty_print(tc_name="Diffusion")

        # to_numpy creates a 2d-array, hence the [:,0]
        time = observable.dataframe_acf["Time"].iloc[:, 0].to_numpy()

        self.diffusion_df["Time"] = np.copy(time)
        self.diffusion_df_slices["Time"] = np.copy(time)

        vacf_str = "VACF"
        const = 1.0 / 3.0

        if not observable.magnetized:
            # Loop over time slices
            for isl in tqdm(range(self.no_slices), disable=not observable.verbose):

                # Iterate over the number of species
                for i, sp in enumerate(observable.species_names):
                    sp_vacf_str = "{} ".format(sp) + vacf_str
                    # Grab vacf data of each slice
                    integrand = np.array(observable.dataframe_acf_slices[(sp_vacf_str, "Total", "slice {}".format(isl))])
                    df_str = "{} Diffusion_slice {}".format(sp, isl)
                    self.diffusion_df_slices[df_str] = const * fast_integral_loop(time=time, integrand=integrand)

            # Average and std of each diffusion coefficient.
            for isp, sp in enumerate(observable.species_names):
                col_str = ["{} Diffusion_slice {}".format(sp, isl) for isl in range(observable.no_slices)]

                self.diffusion_df["{} Diffusion_Mean".format(sp)] = self.diffusion_df_slices[col_str].mean(axis=1)
                self.diffusion_df["{} Diffusion_Std".format(sp)] = self.diffusion_df_slices[col_str].std(axis=1)

        else:
            # Loop over time slices
            for isl in tqdm(range(observable.no_slices), disable=not observable.verbose):

                # Iterate over the number of species
                for i, sp in enumerate(observable.species_names):
                    sp_vacf_str = "{} ".format(sp) + vacf_str

                    # Parallel
                    par_vacf_str = (sp_vacf_str, "Z", "slice {}".format(isl))
                    integrand_par = observable.dataframe_acf_slices[par_vacf_str].to_numpy()
                    par_slice_str = "{} Diffusion_Parallel_slice {}".format(sp, isl)
                    self.diffusion_df_slices[par_slice_str] = fast_integral_loop(time=time, integrand=integrand_par)
                    # Perpendicular
                    perp_vacf_str = (sp_vacf_str, "X", "slice {}".format(isl))
                    perp_slice_str = "{} Diffusion_Perpendicular_slice {}".format(sp, isl)
                    integrand_perp = (
                        observable.dataframe_acf_slices[perp_vacf_str].to_numpy()
                        + observable.dataframe_acf_slices[perp_vacf_str].to_numpy()
                    )

                    self.diffusion_df_slices[perp_slice_str] = 0.5 * fast_integral_loop(
                        time=time, integrand=integrand_perp
                    )

            # Add the average and std of perp and par VACF to its dataframe
            for isp, sp in enumerate(observable.species_names):
                sp_vacf_str = "{} ".format(sp) + vacf_str
                sp_diff_str = "{} ".format(sp) + "Diffusion"
                par_col_str = [(sp_vacf_str, "Z", "slice {}".format(isl)) for isl in range(self.no_slices)]

                observable.dataframe_acf[(sp_vacf_str, "Parallel", "Mean")] = observable.dataframe_acf_slices[
                    par_col_str
                ].mean(axis=1)
                observable.dataframe_acf[(sp_vacf_str, "Parallel", "Std")] = observable.dataframe_acf_slices[
                    par_col_str
                ].std(axis=1)

                x_col_str = [(sp_vacf_str, "X", "slice {}".format(isl)) for isl in range(self.no_slices)]
                y_col_str = [(sp_vacf_str, "Y", "slice {}".format(isl)) for isl in range(self.no_slices)]

                perp_vacf = 0.5 * (
                    observable.dataframe_acf_slices[x_col_str].to_numpy()
                    + observable.dataframe_acf_slices[y_col_str].to_numpy()
                )
                observable.dataframe_acf[(sp_vacf_str, "Perpendicular", "Mean")] = perp_vacf.mean(axis=1)
                observable.dataframe_acf[(sp_vacf_str, "Perpendicular", "Std")] = perp_vacf.std(axis=1)

                # Average and std of each diffusion coefficient.
                par_col_str = [sp_diff_str + "_Parallel_slice {}".format(isl) for isl in range(self.no_slices)]
                perp_col_str = [sp_diff_str + "_Perpendicular_slice {}".format(isl) for isl in range(self.no_slices)]

                self.diffusion_df[sp_diff_str + "_Parallel_Mean"] = self.diffusion_df_slices[par_col_str].mean(axis=1)
                self.diffusion_df[sp_diff_str + "_Parallel_Std"] = self.diffusion_df_slices[par_col_str].std(axis=1)

                self.diffusion_df[sp_diff_str + "_Perpendicular_Mean"] = self.diffusion_df_slices[perp_col_str].mean(
                    axis=1
                )
                self.diffusion_df[sp_diff_str + "_Perpendicular_Std"] = self.diffusion_df_slices[perp_col_str].std(axis=1)

            # Save the updated dataframe
            observable.save_hdf()
            # Endif magnetized.

        self.diffusion_df, self.diffusion_df_slices = self.save_hdf(
            df=self.diffusion_df, df_slices=self.diffusion_df_slices, tc_name="diffusion"
        )

        if plot or display_plot:

            if observable.magnetized:
                for isp, sp in enumerate(observable.species_names):
                    sp_vacf_str = "{} ".format(sp) + vacf_str
                    sp_diff_str = "{} ".format(sp) + "Diffusion"

                    # Parallel
                    acf_avg = observable.dataframe_acf[(sp_vacf_str, "Parallel", "Mean")].to_numpy()
                    acf_std = observable.dataframe_acf[(sp_vacf_str, "Parallel", "Std")].to_numpy()

                    tc_avg = self.diffusion_df[(sp_diff_str, "Parallel", "Mean")].to_numpy()
                    tc_std = self.diffusion_df[(sp_diff_str, "Parallel", "Std")].to_numpy()

                    self.plot_tc(
                        time=time,
                        acf_data=np.column_stack((acf_avg, acf_std)),
                        tc_data=np.column_stack((tc_avg, tc_std)),
                        acf_name=sp_vacf_str + " Parallel",
                        tc_name=sp_diff_str + " Parallel",
                        figname="{}_Parallel_Diffusion_Plot.png".format(sp),
                        show=display_plot,
                    )

                    # Perpendicular
                    acf_avg = observable.dataframe_acf[(sp_vacf_str, "Perpendicular", "Mean")]
                    acf_std = observable.dataframe_acf[(sp_vacf_str, "Perpendicular", "Std")]

                    tc_avg = self.diffusion_df[(sp_diff_str, "Perpendicular", "Mean")]
                    tc_std = self.diffusion_df[(sp_diff_str, "Perpendicular", "Std")]

                    self.plot_tc(
                        time=time,
                        acf_data=np.column_stack((acf_avg, acf_std)),
                        tc_data=np.column_stack((tc_avg, tc_std)),
                        acf_name=sp_vacf_str + " Perpendicular",
                        tc_name=sp_diff_str + " Perpendicular",
                        figname="{}_Perpendicular_Diffusion_Plot.png".format(sp),
                        show=display_plot,
                    )
            else:
                for isp, sp in enumerate(observable.species_names):
                    sp_vacf_str = "{} ".format(sp) + vacf_str
                    acf_avg = observable.dataframe_acf[(sp_vacf_str, "Total", "Mean")].to_numpy()
                    acf_std = observable.dataframe_acf[(sp_vacf_str, "Total", "Std")].to_numpy()

                    d_str = "{} Diffusion".format(sp)
                    tc_avg = self.diffusion_df[(d_str, "Mean")].to_numpy()
                    tc_std = self.diffusion_df[(d_str, "Std")].to_numpy()

                    self.plot_tc(
                        time=time,
                        acf_data=np.column_stack((acf_avg, acf_std)),
                        tc_data=np.column_stack((tc_avg, tc_std)),
                        acf_name=sp_vacf_str,
                        tc_name=d_str,
                        figname="{}_Diffusion_Plot.png".format(sp),
                        show=display_plot,
                    )

    def interdiffusion(self, observable, plot: bool = True, display_plot: bool = False):
        """
        Calculate the transport coefficient from the Green-Kubo formula

        .. math::

            D_{\\alpha} = \\frac{1}{3Nx_1x_2} \\int_0^\\tau dt
            \\langle \\mathbf {J}_{\\alpha}(0) \\cdot \\mathbf {J}_{\\alpha}(t) \\rangle,

        where :math:`x_{1,2}` are the concentration of the two species and
        :math:`\\mathbf {J}_{\\alpha}(t)` is the diffusion current calculated by the
        :class:`sarkas.tools.observables.DiffusionFlux` class.

        Data is retrievable at :attr:`~.interdiffusion_df` and :attr:`~.interdiffusion_df_slices`.

        Parameters
        ----------
        observable: sarkas.tools.observables.DiffusionFlux
            Observable object containing the ACF whose time integral leads to the interdiffusion coefficient.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False

        """

        print("\n\n{:=^70} \n".format(" Interdiffusion Coefficient "))
        self.interdiffusion_df = pd.DataFrame()
        self.interdiffusion_df_slices = pd.DataFrame()

        # Check that the phase and no_slices is the same from the one computed in the Observable
        observable.parse()

        self.phase = observable.phase
        self.no_slices = observable.no_slices
        self.slice_steps = observable.slice_steps
        self.dump_step = observable.dump_step
        # Print some info
        self.pretty_print(tc_name="Interdiffusion")

        # to_numpy creates a 2d-array, hence the [:,0]
        time = observable.dataframe_acf["Time"].to_numpy()[:, 0]
        self.interdiffusion_df["Time"] = np.copy(time)
        self.interdiffusion_df_slices["Time"] = np.copy(time)

        no_fluxes_acf = observable.no_fluxes_acf
        # Normalization constant
        const = 1.0 / (3.0 * observable.total_num_ptcls * observable.species_concentrations.prod())

        df_str = "Diffusion Flux ACF"
        id_str = "Inter Diffusion Flux"
        for isl in tqdm(range(self.no_slices), disable=not self.verbose):
            # D_ij = np.zeros((no_fluxes_acf, jc_acf.slice_steps))

            for ij in range(no_fluxes_acf):
                acf_df_str = (df_str + " {}".format(ij), "Total", "slice {}".format(isl))
                integrand = observable.dataframe_acf_slices[acf_df_str].to_numpy()

                id_df_str = id_str + " {}_slice {}".format(ij, isl)
                self.interdiffusion_df_slices[id_df_str] = const * fast_integral_loop(time=time, integrand=integrand)

        # Average and Std of slices
        for ij in range(no_fluxes_acf):
            col_str = [id_str + " {}_slice {}".format(ij, isl) for isl in range(self.no_slices)]
            self.interdiffusion_df[id_str + " {}_Mean".format(ij)] = self.interdiffusion_df_slices[col_str].mean(axis=1)
            self.interdiffusion_df[id_str + " {}_Std".format(ij)] = self.interdiffusion_df_slices[col_str].std(axis=1)

        self.interdiffusion_df, self.interdiffusion_df_slices = self.save_hdf(
            df=self.interdiffusion_df, df_slices=self.interdiffusion_df_slices, tc_name="Interdiffusion"
        )

        if plot or display_plot:

            for flux in range(observable.no_fluxes_acf):
                flux_str = df_str + " {}".format(flux)
                acf_avg = observable.dataframe_acf[(flux_str, "Total", "Mean")].to_numpy()
                acf_std = observable.dataframe_acf[(flux_str, "Total", "Std")].to_numpy()

                d_str = id_str + " {}".format(flux)
                tc_avg = self.interdiffusion_df[(d_str, "Mean")].to_numpy()
                tc_std = self.interdiffusion_df[(d_str, "Std")].to_numpy()

                self.plot_tc(
                    time=time,
                    acf_data=np.column_stack((acf_avg, acf_std)),
                    tc_data=np.column_stack((tc_avg, tc_std)),
                    acf_name=flux_str,
                    tc_name=d_str,
                    figname="InterDiffusion_{}_Plot.png".format(flux),
                    show=display_plot,
                )

    def viscosity(self, observable, plot: bool = True, display_plot: bool = False):
        """
        Calculate the transport coefficient from the Green-Kubo formula.

        The shear viscosity is obtained from

        .. math::

            \\eta = \\frac{\\beta V}{6} \\sum_{\\alpha} \\sum_{\\gamma \\neq \\alpha} \\int_0^{\\tau} dt \\,
            \\left \\langle \\mathcal P_{\\alpha\\gamma}(t) \\mathcal P_{\\alpha\\gamma}(0) \\right \\rangle

        where :math:`\\beta = 1/k_B T`, :math:`\\alpha,\\gamma = {x, y, z}` and
        :math:`\\mathcal P_{\\alpha\\gamma}(t)` is the element of the Pressure Tensor calculated with
        :class:`sarkas.tools.observables.PressureTensor`.

        The bulk viscosity is obtained from

        .. math::

            \\eta_V = \\beta V \\int_0^{\\tau}dt \\,
             \\left \\langle \\delta \\mathcal P(t) \\delta \\mathcal P(0) \\right \\rangle,

        where

        .. math::
            \\delta \\mathcal P(t) = \\mathcal P(t) - \\left \\langle \\mathcal P  \\right \\rangle

        is the deviation of the scalar pressure.

        Parameters
        ----------
        observable: sarkas.tools.observables.DiffusionFlux
            Observable object containing the ACF whose time integral leads to the interdiffusion coefficient.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False

        """
        print("\n\n{:=^70} \n".format(" Viscosity Coefficient "))
        self.viscosity_df = pd.DataFrame()
        self.viscosity_df_slices = pd.DataFrame()

        # Check that the phase and no_slices is the same from the one computed in the Observable
        observable.parse()

        self.phase = observable.phase
        self.no_slices = observable.no_slices
        self.slice_steps = observable.slice_steps
        self.dump_step = observable.dump_step

        # Print some info
        self.pretty_print(tc_name="Viscosity")

        # to_numpy creates a 2d-array, hence the [:,0]
        time = observable.dataframe_acf["Time"].iloc[:, 0].to_numpy()
        self.viscosity_df["Time"] = np.copy(time)
        self.viscosity_df_slices["Time"] = np.copy(time)

        dim_lbl = ["x", "y", "z"]

        pt_str_list = [
            "Pressure Tensor Kinetic ACF",
            "Pressure Tensor Potential ACF",
            "Pressure Tensor Kin-Pot ACF",
            "Pressure Tensor Pot-Kin ACF",
            "Pressure Tensor ACF",
        ]
        eta_str_list = [
            "Shear Viscosity Tensor Kinetic",
            "Shear Viscosity Tensor Potential",
            "Shear Viscosity Tensor Kin-Pot",
            "Shear Viscosity Tensor Pot-Kin",
            "Shear Viscosity Tensor Total",
        ]

        start_steps = 0
        end_steps = 0

        for isl in tqdm(range(self.no_slices), disable=not observable.verbose):
            end_steps += observable.slice_steps

            const = observable.box_volume * self.beta
            # Calculate Bulk Viscosity
            # It is is calculated from the fluctuations of the pressure eq. 2.124a Allen & Tilsdeley
            integrand = observable.dataframe_acf_slices[("Delta Pressure ACF", "slice {}".format(isl))].to_numpy()
            self.viscosity_df_slices["Bulk Viscosity_slice {}".format(isl)] = const * fast_integral_loop(time, integrand)

            # Calculate the Shear Viscosity Elements
            for _, ax1 in enumerate(dim_lbl):
                for _, ax2 in enumerate(dim_lbl):
                    for _, (pt_str, eta_str) in enumerate(zip(pt_str_list, eta_str_list)):
                        pt_str_temp = (pt_str + " {}{}{}{}".format(ax1, ax2, ax1, ax2), "slice {}".format(isl))
                        integrand = observable.dataframe_acf_slices[pt_str_temp].to_numpy()
                        eta_str_temp = eta_str + " {}{}_slice {}".format(ax1, ax2, isl)
                        self.viscosity_df_slices[eta_str_temp] = const * fast_integral_loop(time, integrand)

            start_steps += observable.slice_steps

        # Now average the slices
        col_str = ["Bulk Viscosity_slice {}".format(isl) for isl in range(observable.no_slices)]
        self.viscosity_df["Bulk Viscosity_Mean"] = self.viscosity_df_slices[col_str].mean(axis=1)
        self.viscosity_df["Bulk Viscosity_Std"] = self.viscosity_df_slices[col_str].std(axis=1)

        for _, ax1 in enumerate(dim_lbl):
            for _, ax2 in enumerate(dim_lbl):
                for _, eta_str in enumerate(eta_str_list):
                    col_str = [eta_str + " {}{}_slice {}".format(ax1, ax2, isl) for isl in range(observable.no_slices)]
                    self.viscosity_df[eta_str + " {}{}_Mean".format(ax1, ax2)] = self.viscosity_df_slices[col_str].mean(
                        axis=1
                    )
                    self.viscosity_df[eta_str + " {}{}_Std".format(ax1, ax2)] = self.viscosity_df_slices[col_str].std(
                        axis=1
                    )

        list_coord = ["xy", "xz", "yx", "yz", "zx", "zy"]
        col_str = [eta_str + " {}_Mean".format(coord) for coord in list_coord]
        self.viscosity_df["Shear Viscosity_Mean"] = self.viscosity_df[col_str].mean(axis=1)
        self.viscosity_df["Shear Viscosity_Std"] = self.viscosity_df[col_str].std(axis=1)

        self.viscosity_df, self.viscosity_df_slices = self.save_hdf(
            df=self.viscosity_df, df_slices=self.viscosity_df_slices, tc_name="Viscosities"
        )

        plot_quantities: list = ["Bulk Viscosity", "Shear Viscosity"]
        if plot or display_plot:
            # Make the plot
            for ipq, pq in enumerate(plot_quantities):
                if pq == "Bulk Viscosity":
                    acf_str = "Delta Pressure ACF"
                    acf_avg = observable.dataframe_acf[("Delta Pressure ACF", "Mean")]
                    acf_std = observable.dataframe_acf[("Delta Pressure ACF", "Std")]
                elif pq == "Shear Viscosity":
                    # The axis are the last two elements in the string
                    acf_strs = [("Pressure Tensor ACF {}".format(coord), "Mean") for coord in list_coord]
                    acf_avg = observable.dataframe_acf[acf_strs].mean(axis=1)
                    acf_std = observable.dataframe_acf[acf_strs].std(axis=1)

                tc_avg = self.viscosity_df[(pq, "Mean")]
                tc_std = self.viscosity_df[(pq, "Std")]

                self.plot_tc(
                    time=time,
                    acf_data=np.column_stack((acf_avg, acf_std)),
                    tc_data=np.column_stack((tc_avg, tc_std)),
                    acf_name=acf_str,
                    tc_name=pq,
                    figname="{}_Plot.png".format(pq),
                    show=display_plot,
                )
