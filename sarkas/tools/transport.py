"""
Transport Module.
"""

from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import datetime
import sys
from matplotlib.pyplot import subplots
from numpy import array, column_stack, pi
from os import mkdir as os_mkdir
from os.path import exists as os_path_exists
from os.path import join as os_path_join
from pandas import concat, DataFrame, MultiIndex, read_hdf, Series

from ..utilities.maths import fast_integral_loop
from ..utilities.timing import SarkasTimer

# Sarkas Modules
from .observables import plot_labels, Thermodynamics


class TransportCoefficients:
    """
    Transport Coefficients class.

    Parameters
    ----------
    params : :class:`sarkas.core.Parameters`
        Simulation parameters.

    phase : str, optional
        Simulation's phase. `"equilibration"` or `"production"`. Default = `"production"`.

    no_slices : str, optional
        Number of slices in which the phase has been divided. Default = 1.
    """

    def __init__(self, params, phase: str = "production", no_slices: int = 1):
        self.__name__ = "TC"
        self.__long_name__ = "Transport Coefficients"
        self.tc_names = ["electricalconductivity", "diffusion", "interdiffusion", "viscosities"]
        self.tc_dict = {
            self.tc_names[0]: {
                "df_fname": "ElectricalConductivity",
                "screen_str": " Electrical Conductivity ",
            },
            self.tc_names[1]: {"df_fname": "Diffusion", "screen_str": " Diffusion Coefficients"},
            self.tc_names[2]: {"df_fname": "Interdiffusion", "screen_str": " Interdiffusion Coefficients"},
            self.tc_names[3]: {"df_fname": "Viscosities", "screen_str": " Viscosity Coefficients "},
        }
        self.timer = SarkasTimer()

        # Parameters copies
        self.postprocessing_dir = params.postprocessing_dir
        self.units = params.units
        self.job_id = params.job_id
        self.verbose = params.verbose
        self.dt = params.dt
        self.units_dict = params.units_dict
        self.total_plasma_frequency = params.total_plasma_frequency
        self.dimensions = params.dimensions
        self.box_volume = params.box_volume
        self.pbox_volume = params.pbox_volume
        #
        self.saving_dir = None
        self.phase = phase
        self.no_slices = no_slices
        # Calculate the average temperature from the csv data
        self.kB = params.kB
        energy = Thermodynamics()
        energy.setup(params, self.phase)
        energy.parse()
        self.T_avg = energy.dataframe["Temperature"].mean()
        self.beta = 1.0 / (self.kB * self.T_avg)

        self.time_array = None
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

    @staticmethod
    def add_to_df(df, data, column_name):
        """Routine to add a column of data to a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe to which data has to be added.

        data: numpy.ndarray
            Data to be added.

        column_name: str
            Name of the column to be added.

        Returns
        -------
        _ : pandas.DataFrame
            Original `df` concatenated with the `data`.

        Note
        ----
            It creates a `pandas.Series` from `data` using `df.index`. Then it uses `concat` to add the column.

        """
        col_data = Series(data, index=df.index)

        return concat([df, col_data.rename(column_name)], axis=1)

    def compute_init(self, observable, coeff_name: str = None):
        """
        Initialize the dataframes where to store the data.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.Observable`
            Observable object containing the ACF whose time integral leads to the desired transport coefficient.

        coeff_name : str
            Transport coefficient to calculate. Choose amongst `self.tc_names`.

        """
        # Write Log File
        self.log_file = os_path_join(self.saving_dir, self.tc_dict[coeff_name]["df_fname"] + "_logfile.out")
        self.datetime_stamp()

        self.get_acf_data(observable)

        if observable.verbose:
            self.pretty_print(coeff_name)

        # to_numpy creates a 2d-array, hence the [:,0]
        self.time_array = observable.dataframe_acf["Time"].iloc[:, 0].values

        if coeff_name in self.tc_names:

            if coeff_name == self.tc_names[0]:
                self.conductivity_df = DataFrame()
                self.conductivity_df_slices = DataFrame()
                self.conductivity_df["Time"] = self.time_array.copy()
                self.conductivity_df_slices["Time"] = self.time_array.copy()

            elif coeff_name == self.tc_names[1]:
                self.diffusion_df = DataFrame()
                self.diffusion_df_slices = DataFrame()
                self.diffusion_df["Time"] = self.time_array.copy()
                self.diffusion_df_slices["Time"] = self.time_array.copy()

            elif coeff_name == self.tc_names[2]:
                self.interdiffusion_df = DataFrame()
                self.interdiffusion_df_slices = DataFrame()
                self.interdiffusion_df["Time"] = self.time_array.copy()
                self.interdiffusion_df_slices["Time"] = self.time_array.copy()

            elif coeff_name == self.tc_names[3]:
                self.viscosity_df = DataFrame()
                self.viscosity_df_slices = DataFrame()
                self.viscosity_df["Time"] = self.time_array.copy()
                self.viscosity_df_slices["Time"] = self.time_array.copy()
        else:
            raise ValueError("Wrong tc_name. Please choose amongst ", self.tc_names)

    def create_dir(self):
        """Create directory where to save the transport coefficients."""

        saving_dir = os_path_join(self.postprocessing_dir, self.__long_name__.replace(" ", ""))
        if not os_path_exists(saving_dir):
            os_mkdir(saving_dir)

        self.saving_dir = os_path_join(saving_dir, self.phase.capitalize())
        if not os_path_exists(self.saving_dir):
            os_mkdir(self.saving_dir)

    def datetime_stamp(self):
        """Add a Date and Time stamp to log file."""

        if os_path_exists(self.log_file):
            with open(self.log_file, "a+") as f_log:
                # Add some space to better distinguish the new beginning
                print(f"\n\n\n", file=f_log)

        with open(self.log_file, "a+") as f_log:
            ct = datetime.datetime.now()
            print(f"{'':~^80}", file=f_log)
            print(f"Date: {ct.year} - {ct.month} - {ct.day}", file=f_log)
            print(f"Time: {ct.hour}:{ct.minute}:{ct.second}", file=f_log)
            print(f"{'':~^80}\n", file=f_log)

    def diffusion(self, observable, plot: bool = True, display_plot: bool = False):
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
        observable : :class:`sarkas.tools.observables.VelocityAutoCorrelationFunction`
            Observable object containing the ACF whose time integral leads to the self diffusion coefficient.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.

        """

        self.compute_init(observable, coeff_name="diffusion")
        vacf_str = "VACF"
        const = 1.0 / self.dimensions
        t0 = self.timer.current()

        if not observable.magnetized:
            # Loop over time slices
            for isl in tqdm(range(self.no_slices), disable=not observable.verbose):

                # Iterate over the number of species
                for i, sp in enumerate(observable.species_names):
                    sp_vacf_str = f"{sp} " + vacf_str
                    # Grab vacf data of each slice
                    integrand = array(observable.dataframe_acf_slices[(sp_vacf_str, "Total", f"slice {isl}")])
                    df_str = f"{sp} Diffusion_slice {isl}"
                    self.diffusion_df_slices[df_str] = const * fast_integral_loop(
                        time=self.time_array, integrand=integrand
                    )

            # Average and std of each diffusion coefficient.
            for isp, sp in enumerate(observable.species_names):
                col_str = [f"{sp} Diffusion_slice {isl}" for isl in range(observable.no_slices)]
                # Mean
                col_data = self.diffusion_df_slices[col_str].mean(axis=1).values
                col_name = f"{sp} Diffusion_Mean"
                self.diffusion_df = self.add_to_df(self.diffusion_df, col_data, col_name)
                # Std
                col_data = self.diffusion_df_slices[col_str].std(axis=1).values
                col_name = f"{sp} Diffusion_Std"
                self.diffusion_df = self.add_to_df(self.diffusion_df, col_data, col_name)

        else:
            # Loop over time slices
            for isl in tqdm(range(observable.no_slices), disable=not observable.verbose):

                # Iterate over the number of species
                for i, sp in enumerate(observable.species_names):
                    sp_vacf_str = f"{sp} " + vacf_str

                    # Parallel
                    par_vacf_str = (sp_vacf_str, "Z", f"slice {isl}")
                    integrand_par = observable.dataframe_acf_slices[par_vacf_str].to_numpy()

                    col_data = fast_integral_loop(time=self.time_array, integrand=integrand_par)
                    col_name = f"{sp} Diffusion_Parallel_slice {isl}"
                    self.diffusion_df_slices = self.add_to_df(self.diffusion_df_slices, col_data, col_name)

                    # Perpendicular
                    x_vacf_str = (sp_vacf_str, "X", f"slice {isl}")
                    y_vacf_str = (sp_vacf_str, "Y", f"slice {isl}")

                    integrand_perp = 0.5 * (
                        observable.dataframe_acf_slices[x_vacf_str].to_numpy()
                        + observable.dataframe_acf_slices[y_vacf_str].to_numpy()
                    )
                    col_data = fast_integral_loop(time=self.time_array, integrand=integrand_perp)
                    col_name = f"{sp} Diffusion_Perpendicular_slice {isl}"
                    self.diffusion_df_slices = self.add_to_df(self.diffusion_df_slices, col_data, col_name)

            # Add the average and std of perp and par VACF to its dataframe
            for isp, sp in enumerate(observable.species_names):
                sp_vacf_str = f"{sp} " + vacf_str
                sp_diff_str = f"{sp} " + "Diffusion"
                par_col_str = [(sp_vacf_str, "Z", f"slice {isl}") for isl in range(self.no_slices)]

                observable.dataframe_acf[(sp_vacf_str, "Parallel", "Mean")] = observable.dataframe_acf_slices[
                    par_col_str
                ].mean(axis=1)
                observable.dataframe_acf[(sp_vacf_str, "Parallel", "Std")] = observable.dataframe_acf_slices[
                    par_col_str
                ].std(axis=1)

                x_col_str = [(sp_vacf_str, "X", f"slice {isl}") for isl in range(self.no_slices)]
                y_col_str = [(sp_vacf_str, "Y", f"slice {isl}") for isl in range(self.no_slices)]

                perp_vacf = 0.5 * (
                    observable.dataframe_acf_slices[x_col_str].to_numpy()
                    + observable.dataframe_acf_slices[y_col_str].to_numpy()
                )
                observable.dataframe_acf[(sp_vacf_str, "Perpendicular", "Mean")] = perp_vacf.mean(axis=1)
                observable.dataframe_acf[(sp_vacf_str, "Perpendicular", "Std")] = perp_vacf.std(axis=1)

                # Average and std of each diffusion coefficient.
                par_col_str = [sp_diff_str + f"_Parallel_slice {isl}" for isl in range(self.no_slices)]
                perp_col_str = [sp_diff_str + f"_Perpendicular_slice {isl}" for isl in range(self.no_slices)]

                # Mean
                col_data = self.diffusion_df_slices[par_col_str].mean(axis=1).values
                col_name = sp_diff_str + "_Parallel_Mean"
                self.diffusion_df = self.add_to_df(self.diffusion_df, col_data, col_name)
                # Std
                col_data = self.diffusion_df_slices[par_col_str].std(axis=1).values
                col_name = sp_diff_str + "_Parallel_Std"
                self.diffusion_df = self.add_to_df(self.diffusion_df, col_data, col_name)

                # Mean
                col_data = self.diffusion_df_slices[perp_col_str].mean(axis=1).values
                col_name = sp_diff_str + "_Perpendicular_Mean"
                self.diffusion_df = self.add_to_df(self.diffusion_df, col_data, col_name)
                # Std
                col_data = self.diffusion_df_slices[perp_col_str].std(axis=1).values
                col_name = sp_diff_str + "_Perpendicular_Std"
                self.diffusion_df = self.add_to_df(self.diffusion_df, col_data, col_name)

            # Save the updated dataframe
            observable.save_hdf()
            # Endif magnetized.

        # Time stamp
        tend = self.timer.current()
        self.time_stamp("Diffusion Calculation", self.timer.time_division(tend - t0))

        # Save
        self.diffusion_df, self.diffusion_df_slices = self.save_hdf(
            df=self.diffusion_df, df_slices=self.diffusion_df_slices, tc_name="diffusion"
        )

        # Plot
        if plot or display_plot:

            if observable.magnetized:
                for isp, sp in enumerate(observable.species_names):
                    sp_vacf_str = f"{sp} " + vacf_str
                    sp_diff_str = f"{sp} Diffusion"

                    # Parallel
                    acf_avg = observable.dataframe_acf[(sp_vacf_str, "Parallel", "Mean")].to_numpy()
                    acf_std = observable.dataframe_acf[(sp_vacf_str, "Parallel", "Std")].to_numpy()

                    tc_avg = self.diffusion_df[(sp_diff_str, "Parallel", "Mean")].to_numpy()
                    tc_std = self.diffusion_df[(sp_diff_str, "Parallel", "Std")].to_numpy()

                    self.plot_tc(
                        time=self.time_array,
                        acf_data=column_stack((acf_avg, acf_std)),
                        tc_data=column_stack((tc_avg, tc_std)),
                        acf_name=sp_vacf_str + " Parallel",
                        tc_name=sp_diff_str + " Parallel",
                        figname=f"{sp}_Parallel_Diffusion_Plot.png",
                        show=display_plot,
                    )

                    # Perpendicular
                    acf_avg = observable.dataframe_acf[(sp_vacf_str, "Perpendicular", "Mean")]
                    acf_std = observable.dataframe_acf[(sp_vacf_str, "Perpendicular", "Std")]

                    tc_avg = self.diffusion_df[(sp_diff_str, "Perpendicular", "Mean")]
                    tc_std = self.diffusion_df[(sp_diff_str, "Perpendicular", "Std")]

                    self.plot_tc(
                        time=self.time_array,
                        acf_data=column_stack((acf_avg, acf_std)),
                        tc_data=column_stack((tc_avg, tc_std)),
                        acf_name=sp_vacf_str + " Perpendicular",
                        tc_name=sp_diff_str + " Perpendicular",
                        figname=f"{sp}_Perpendicular_Diffusion_Plot.png",
                        show=display_plot,
                    )
            else:
                for isp, sp in enumerate(observable.species_names):
                    sp_vacf_str = f"{sp} " + vacf_str
                    acf_avg = observable.dataframe_acf[(sp_vacf_str, "Total", "Mean")].to_numpy()
                    acf_std = observable.dataframe_acf[(sp_vacf_str, "Total", "Std")].to_numpy()

                    d_str = f"{sp} Diffusion"
                    tc_avg = self.diffusion_df[(d_str, "Mean")].to_numpy()
                    tc_std = self.diffusion_df[(d_str, "Std")].to_numpy()

                    self.plot_tc(
                        time=self.time_array,
                        acf_data=column_stack((acf_avg, acf_std)),
                        tc_data=column_stack((tc_avg, tc_std)),
                        acf_name=sp_vacf_str,
                        tc_name=d_str,
                        figname=f"{sp}_Diffusion_Plot.png",
                        show=display_plot,
                    )

    def electrical_conductivity(
        self,
        observable,
        plot: bool = True,
        display_plot: bool = False,
    ):
        """
        Calculate the transport coefficient from the Green-Kubo formula

        .. math::

            \\sigma = \\frac{\\beta}{V} \\int_0^{\\tau} dt J(t).

        where :math:`\\beta = 1/k_B T` and :math:`V` is the volume of the simulation box.

        Data is retrievable at :attr:`~.conductivity_df` and :attr:`~.conductivity_df_slices`.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.ElectricCurrent`
            Observable object containing the ACF whose time integral leads to the electrical conductivity.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. \n
            Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.

        """

        self.compute_init(observable, "electricalconductivity")

        # Time
        t0 = self.timer.current()

        jc_str = "Electric Current ACF"
        sigma_str = "Electrical Conductivity"
        const = self.beta / observable.box_volume

        if not observable.magnetized:
            for isl in tqdm(range(observable.no_slices), disable=not observable.verbose):
                integrand = array(observable.dataframe_acf_slices[(jc_str, "Total", "slice {}".format(isl))])
                col_name = sigma_str + "_slice {}".format(isl)
                col_data = const * fast_integral_loop(self.time_array, integrand)
                self.conductivity_df_slices = self.add_to_df(self.conductivity_df_slices, col_data, col_name)

            col_str = [sigma_str + "_slice {}".format(isl) for isl in range(observable.no_slices)]
            # Mean
            col_data = self.conductivity_df_slices[col_str].mean(axis=1).values
            col_name = sigma_str + "_Mean"
            self.conductivity_df = self.add_to_df(self.conductivity_df, col_data, col_name)
            # Std
            col_data = self.conductivity_df_slices[col_str].std(axis=1).values
            col_name = sigma_str + "_Std"
            self.conductivity_df = self.add_to_df(self.conductivity_df, col_data, col_name)

        else:

            for isl in tqdm(range(observable.no_slices), disable=not observable.verbose):
                # Parallel
                par_str = (jc_str, "Z", f"slice {isl}")
                integrand = observable.dataframe_acf_slices[par_str].to_numpy()
                col_data = const * fast_integral_loop(self.time_array, integrand)
                col_name = sigma_str + f"_Parallel_slice {isl}"
                self.conductivity_df_slices = self.add_to_df(self.conductivity_df_slices, col_data, col_name)

                # Perpendicular
                x_col_str = (jc_str, "X", f"slice {isl}")
                y_col_str = (jc_str, "Y", f"slice {isl}")
                perp_integrand = 0.5 * (
                    observable.dataframe_acf_slices[x_col_str].to_numpy()
                    + observable.dataframe_acf_slices[y_col_str].to_numpy()
                )
                col_name = sigma_str + f"_Perpendicular_slice {isl}"
                col_data = const * fast_integral_loop(self.time_array, perp_integrand)
                self.conductivity_df_slices = self.add_to_df(self.conductivity_df_slices, col_data, col_name)

            par_col_str = [(jc_str, "Z", f"slice {isl}") for isl in range(self.no_slices)]
            observable.dataframe_acf[(jc_str, "Parallel", "Mean")] = observable.dataframe_acf_slices[par_col_str].mean(
                axis=1
            )
            observable.dataframe_acf[(jc_str, "Parallel", "Std")] = observable.dataframe_acf_slices[par_col_str].std(
                axis=1
            )

            x_col_str = [(jc_str, "X", f"slice {isl}") for isl in range(self.no_slices)]
            y_col_str = [(jc_str, "Y", f"slice {isl}") for isl in range(self.no_slices)]

            perp_jc = 0.5 * (
                observable.dataframe_acf_slices[x_col_str].to_numpy()
                + observable.dataframe_acf_slices[y_col_str].to_numpy()
            )
            observable.dataframe_acf[(jc_str, "Perpendicular", "Mean")] = perp_jc.mean(axis=1)
            observable.dataframe_acf[(jc_str, "Perpendicular", "Std")] = perp_jc.std(axis=1)
            # Save the updated dataframe
            observable.save_hdf()

            # Average and std of transport coefficient.
            col_str = [sigma_str + f"_Parallel_slice {isl}".format(isl) for isl in range(observable.no_slices)]
            # Mean
            col_data = self.conductivity_df_slices[col_str].mean(axis=1).values
            col_name = sigma_str + "_Parallel_Mean"
            self.conductivity_df = self.add_to_df(self.conductivity_df, col_data, col_name)
            # Std
            col_data = self.conductivity_df_slices[col_str].std(axis=1).values
            col_name = sigma_str + "_Parallel_Std"
            self.conductivity_df = self.add_to_df(self.conductivity_df, col_data, col_name)

            # Perpendicular
            col_str = [sigma_str + f"_Perpendicular_slice {isl}" for isl in range(observable.no_slices)]
            # Mean
            col_data = sigma_str + "_Perpendicular_Mean"
            col_name = self.conductivity_df_slices[col_str].mean(axis=1).values
            self.conductivity_df = self.add_to_df(self.conductivity_df, col_data, col_name)
            # Std
            col_data = sigma_str + "_Perpendicular_Std"
            col_name = self.conductivity_df_slices[col_str].std(axis=1).values
            self.conductivity_df = self.add_to_df(self.conductivity_df, col_data, col_name)

            # Endif magnetized.
        # Time stamp
        tend = self.timer.current()
        self.time_stamp("Electrica Conductivity Calculation", self.timer.time_division(tend - t0))

        # Save
        self.conductivity_df, self.conductivity_df_slices = self.save_hdf(
            df=self.conductivity_df, df_slices=self.conductivity_df_slices, tc_name="electricalconductivity"
        )

        if plot or display_plot:

            if not observable.magnetized:
                acf_avg = observable.dataframe_acf[(jc_str, "Total", "Mean")].to_numpy()
                acf_std = observable.dataframe_acf[(jc_str, "Total", "Std")].to_numpy()

                tc_avg = self.conductivity_df[(sigma_str, "Mean")].to_numpy()
                tc_std = self.conductivity_df[(sigma_str, "Std")].to_numpy()

                self.plot_tc(
                    time=self.time_array,
                    acf_data=column_stack((acf_avg, acf_std)),
                    tc_data=column_stack((tc_avg, tc_std)),
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
                    time=self.time_array,
                    acf_data=column_stack((acf_avg, acf_std)),
                    tc_data=column_stack((tc_avg, tc_std)),
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
                    time=self.time_array,
                    acf_data=column_stack((acf_avg, acf_std)),
                    tc_data=column_stack((tc_avg, tc_std)),
                    acf_name="Electric Current ACF Perpendicular",
                    tc_name="Electrical Conductivity Perpendicular",
                    figname="ElectricalConductivity_Perpendicular_Plot.png",
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
        observable : :class:`sarkas.tools.observables.DiffusionFlux`
            Observable object containing the ACF whose time integral leads to the interdiffusion coefficient.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False

        """

        self.compute_init(observable, coeff_name="interdiffusion")
        no_fluxes_acf = observable.no_fluxes_acf
        # Normalization constant
        const = 1.0 / (3.0 * observable.total_num_ptcls * observable.species_concentrations.prod())

        df_str = "Diffusion Flux ACF"
        id_str = "Inter Diffusion Flux"
        # Time
        t0 = self.timer.current()
        for isl in tqdm(range(self.no_slices), disable=not self.verbose):
            # D_ij = zeros((no_fluxes_acf, jc_acf.slice_steps))

            for ij in range(no_fluxes_acf):
                acf_df_str = (df_str + f" {ij}", "Total", f"slice {isl}")
                integrand = observable.dataframe_acf_slices[acf_df_str].to_numpy()

                col_data = const * fast_integral_loop(time=self.time_array, integrand=integrand)
                col_name = id_str + f" {ij}_slice {isl}"
                self.interdiffusion_df_slices = self.add_to_df(observable, col_data, col_name)

        # Average and Std of slices
        for ij in range(no_fluxes_acf):
            col_str = [id_str + f" {ij}_slice {isl}" for isl in range(self.no_slices)]
            # Mean
            col_data = self.interdiffusion_df_slices[col_str].mean(axis=1).values
            col_name = id_str + f" {ij}_Mean"
            self.interdiffusion_df = self.add_to_df(observable, col_data, col_name)
            # Mean
            col_data = self.interdiffusion_df_slices[col_str].std(axis=1).values
            col_name = id_str + f" {ij}_Std"
            self.interdiffusion_df = self.add_to_df(observable, col_data, col_name)

        # Save
        self.interdiffusion_df, self.interdiffusion_df_slices = self.save_hdf(
            df=self.interdiffusion_df, df_slices=self.interdiffusion_df_slices, tc_name="interdiffusion"
        )

        # Time stamp
        tend = self.timer.current()
        self.time_stamp("Interdiffusion Calculation", self.timer.time_division(tend - t0))

        if plot or display_plot:

            for flux in range(observable.no_fluxes_acf):
                flux_str = df_str + " {}".format(flux)
                acf_avg = observable.dataframe_acf[(flux_str, "Total", "Mean")].to_numpy()
                acf_std = observable.dataframe_acf[(flux_str, "Total", "Std")].to_numpy()

                d_str = id_str + " {}".format(flux)
                tc_avg = self.interdiffusion_df[(d_str, "Mean")].to_numpy()
                tc_std = self.interdiffusion_df[(d_str, "Std")].to_numpy()

                self.plot_tc(
                    time=self.time_array,
                    acf_data=column_stack((acf_avg, acf_std)),
                    tc_data=column_stack((tc_avg, tc_std)),
                    acf_name=flux_str,
                    tc_name=d_str,
                    figname="InterDiffusion_{}_Plot.png".format(flux),
                    show=display_plot,
                )

    def get_acf_data(self, observable):
        # Check that the phase and no_slices is the same from the one computed in the Observable
        observable.parse()

        self.phase = observable.phase
        self.no_slices = observable.no_slices
        self.slice_steps = observable.slice_steps
        self.dump_step = observable.dump_step

    def parse(self, observable, coeff_name: str = None):
        """Read the HDF files containing the transport coefficients.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.Observable`
            Observable object containing the ACF whose time integral leads to the electrical conductivity.

        coeff_name : str
            Transport Coefficient name.

        Raises
        ------
        ValueError
            Wrong ``coeff_name``.

        """
        coeff_name = coeff_name.lower()

        if coeff_name in self.coeff_names:
            # Copy relevant info
            self.phase = observable.phase
            self.no_slices = observable.no_slices
            self.slice_steps = observable.slice_steps
            self.dump_step = observable.dump_step

            # Define filenames
            df_h5_fname = self.tc_dict[coeff_name]["df_fname"] + "_" + self.job_id + ".h5"
            df_slices_h5_fname = self.tc_dict[coeff_name]["df_fname"] + "_slices_" + self.job_id + ".h5"

            if coeff_name == "electricalconductivity":
                try:
                    self.conductivity_df = read_hdf(os_path_join(self.saving_dir, df_h5_fname), mode="r", index_col=False)

                    self.conductivity_df_slices = read_hdf(
                        os_path_join(self.saving_dir, df_slices_h5_fname), mode="r", index_col=False
                    )
                except FileNotFoundError:
                    print(f"\nEither of these files {df_h5_fname}, {df_slices_h5_fname} not found! Computing them now")
                    self.electrical_conductivity(observable=observable)

            elif coeff_name == "diffusion":
                try:
                    self.diffusion_df = read_hdf(os_path_join(self.saving_dir, df_h5_fname), mode="r", index_col=False)
                    self.diffusion_df_slices = read_hdf(
                        os_path_join(self.saving_dir, df_slices_h5_fname), mode="r", index_col=False
                    )
                except FileNotFoundError:
                    print(f"\nEither of these files {df_h5_fname}, {df_slices_h5_fname} not found! Computing them now")
                    self.diffusion(observable=observable)

            elif coeff_name == "interdiffusion":

                try:
                    self.interdiffusion_df = read_hdf(
                        os_path_join(self.saving_dir, df_h5_fname), mode="r", index_col=False
                    )
                    self.interdiffusion_df_slices = read_hdf(
                        os_path_join(self.saving_dir, df_slices_h5_fname), mode="r", index_col=False
                    )
                except FileNotFoundError:
                    print(f"\nEither of these files {df_h5_fname}, {df_slices_h5_fname} not found! Computing them now")
                    self.interdiffusion(observable=observable)

            elif coeff_name == "viscosities":

                try:
                    self.viscosity_df = read_hdf(
                        os_path_join(self.saving_dir, df_h5_fname),
                        mode="r",
                        index_col=False,
                    )

                    self.viscosity_df_slices = read_hdf(
                        os_path_join(self.saving_dir, df_slices_h5_fname),
                        mode="r",
                        index_col=False,
                    )
                except FileNotFoundError:
                    print(f"\nEither of these files {df_h5_fname}, {df_slices_h5_fname} not found! Computing them now")
                    self.viscosity(observable=observable)
        else:
            raise ValueError("Wrong coeff_name. Please choose amongst ", self.tc_names)

        # Print some info
        self.pretty_print(coeff_name=coeff_name)

    def plot_tc(self, time, acf_data, tc_data, acf_name, tc_name, figname, show: bool = False):
        """
        Make dual plots with ACF and transport coefficient.

        Parameters
        ----------
        time : numpy.ndarray
            Time array.

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
        fig, (ax1, ax2) = subplots(1, 2, figsize=(16, 7))
        ax3 = ax1.twiny()
        ax4 = ax2.twiny()

        # Calculate axis multipliers and labels
        xmul, ymul, _, _, xlbl, ylbl = plot_labels(time, tc_data[:, 0], "Time", tc_name, self.units)

        # ACF
        ax1.plot(xmul * time, acf_data[:, 0] / acf_data[0, 0])
        ax1.fill_between(
            xmul * time,
            (acf_data[:, 0] - acf_data[:, 1]) / (acf_data[0, 0] - acf_data[0, 1]),
            (acf_data[:, 0] + acf_data[:, 1]) / (acf_data[0, 0] + acf_data[0, 1]),
            alpha=0.2,
        )

        # Coefficient
        ax2.plot(xmul * time, ymul * tc_data[:, 0])
        ax2.fill_between(
            xmul * time, ymul * (tc_data[:, 0] - tc_data[:, 1]), ymul * (tc_data[:, 0] + tc_data[:, 1]), alpha=0.2
        )

        xlims = (xmul * time[1], xmul * time[-1] * 1.5)
        ax1.set(xlim=xlims, xscale="log", ylabel=acf_name, xlabel=r"Time difference" + xlbl)
        xlims = (0, xmul * time[-1] * 1.05)
        ax2.set(xlim=xlims, ylabel=tc_name + ylbl, xlabel=r"$\tau$" + xlbl)

        # ax1.legend(loc='best')
        # ax2.legend(loc='best')
        # Finish the index axes
        ax3.set(xlim=(1, self.slice_steps * 1.5), xscale="log")
        for axi in [ax3, ax4]:
            axi.grid(alpha=0.1)
            axi.set(xlabel="Index")
        ax4.set(xlim=(0, self.slice_steps * 1.05))

        fig.tight_layout()
        fig.savefig(os_path_join(self.saving_dir, figname))

        if show:
            fig.show()

        return fig, (ax1, ax2, ax3, ax4)

    def pretty_print(self, coeff_name: str = None):
        """
        Print to screen the location where data is stored and other relevant information.

        Parameters
        ----------
        coeff_name: str
            Name of Transport coefficient to calculate.
        """

        if coeff_name is None or coeff_name not in self.tc_names:
            raise ValueError("Wrong coeff_name. Choose amongst ", self.tc_names)

        fname = self.tc_dict[coeff_name]["df_fname"]
        tc_name = self.tc_dict[coeff_name]["screen_str"]

        # Create the message to print
        data_loc = os_path_join(self.saving_dir, fname + "_" + self.job_id + ".h5")
        data_slices_loc = os_path_join(self.saving_dir, fname + "_slices_" + self.job_id + ".h5")
        dtau = self.dt * self.dump_step
        tau = dtau * self.slice_steps
        t_wp = 2.0 * pi / self.total_plasma_frequency  # Plasma period
        tau_wp = int(tau / t_wp)
        msg = (
            f"\n\n{tc_name:=^70}\n"
            f"Data saved in: \n {data_loc} \n {data_slices_loc} \n"
            f"No. of slices = {self.no_slices}\n"
            f"No. dumps per slice = {int(self.slice_steps / self.dump_step)}\n"
            f"Total time interval of autocorrelation function: tau = {tau:.4e} {self.units_dict['time']} ~ {tau_wp} plasma periods\n"
            f"Time interval step: dtau = {dtau:.4e} ~ {dtau / t_wp:.4e} plasma period"
        )

        # Print the message to log file and screen
        screen = sys.stdout
        f_log = open(self.log_file, "a+")
        repeat = 2 if self.verbose else 1

        # redirect printing to file
        sys.stdout = f_log
        while repeat > 0:
            print(msg)
            repeat -= 1
            sys.stdout = screen

        f_log.close()

    def save_hdf(self, df: DataFrame = None, df_slices: DataFrame = None, tc_name: str = None):
        """
        Save the HDF dataframes to disk in the TransportCoefficient folder.

        Parameters
        ----------
        df: pandas.DataFrame
            Multi-index dataframe containing the mean and std of the transport coefficient.

        df_slices: pandas.DataFrame
            Multi-index dataframe containing the transport coefficient of each slice.

        tc_name: str
            Name of the transport coefficient. See :attr:`~.tc_names` for options.

        Returns
        -------
        df: pandas.DataFrame
            Sorted multi-index dataframe containing the mean and std of the transport coefficient.

        df_slices: pandas.DataFrame
            Sorted multi-index dataframe containing the transport coefficient of each slice.

        """

        df_slices.columns = MultiIndex.from_tuples([tuple(c.split("_")) for c in df_slices.columns])
        df_slices = df_slices.sort_index()
        df_slices.to_hdf(
            os_path_join(self.saving_dir, self.tc_dict[tc_name]["df_fname"] + "_slices_" + self.job_id + ".h5"),
            mode="w",
            key=tc_name,
            index=False,
        )

        df.columns = MultiIndex.from_tuples([tuple(c.split("_")) for c in df.columns])
        df = df.sort_index()
        df.to_hdf(
            os_path_join(self.saving_dir, self.tc_dict[tc_name]["df_fname"] + "_" + self.job_id + ".h5"),
            mode="w",
            key=tc_name,
            index=False,
        )

        return df, df_slices

    def time_stamp(self, message: str, timing: tuple):
        """
        Print out to screen elapsed times. If verbose output, print to file first and then to screen.

        Parameters
        ----------
        message : str
            Message to print.

        timing : tuple
            Time in hrs, min, sec, msec, usec, nsec.

        """

        screen = sys.stdout
        f_log = open(self.log_file, "a+")
        repeat = 2 if self.verbose else 1
        t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = timing

        # redirect printing to file
        sys.stdout = f_log
        while repeat > 0:

            if t_hrs == 0 and t_min == 0 and t_sec <= 2:
                print(f"\n{message} Time: {int(t_sec)} sec {int(t_msec)} msec {int(t_usec)} usec {int(t_nsec)} nsec")
            else:
                print(f"\n{message} Time: {int(t_hrs)} hrs {int(t_min)} min {int(t_sec)} sec")

            repeat -= 1
            sys.stdout = screen

        f_log.close()

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
        observable : :class:`sarkas.tools.observables.PressureTensor`
            Observable object containing the ACF whose time integral leads to the viscsosity coefficients.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False

        """

        self.compute_init(observable, "viscosities")
        # Initialize Timer
        t0 = self.timer.current()

        # Initialize some labels
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
            # It is calculated from the fluctuations of the pressure eq. 2.124a Allen & Tilsdeley
            integrand = observable.dataframe_acf_slices[(f"Delta Pressure ACF", f"slice {isl}")].to_numpy()

            col_name = f"Bulk Viscosity_slice {isl}"
            col_data = const * fast_integral_loop(self.time_array, integrand)
            self.viscosity_df_slices = self.add_to_df(self.viscosity_df_slices, col_data, col_name)

            # Calculate the Shear Viscosity Elements
            for _, ax1 in enumerate(dim_lbl):
                for _, ax2 in enumerate(dim_lbl):
                    for _, (pt_str, eta_str) in enumerate(zip(pt_str_list, eta_str_list)):
                        pt_str_temp = (pt_str + f" {ax1}{ax2}{ax1}{ax2}", f"slice {isl}")
                        integrand = observable.dataframe_acf_slices[pt_str_temp].to_numpy()
                        col_name = eta_str + f" {ax1}{ax2}_slice {isl}".format(ax1, ax2, isl)
                        col_data = const * fast_integral_loop(self.time_array, integrand)
                        self.viscosity_df_slices = self.add_to_df(self.viscosity_df_slices, col_data, col_name)

            start_steps += observable.slice_steps

        # Now average the slices
        col_str = [f"Bulk Viscosity_slice {isl}" for isl in range(observable.no_slices)]

        col_name = "Bulk Viscosity_Mean"
        col_data = self.viscosity_df_slices[col_str].mean(axis=1).values
        self.viscosity_df = self.add_to_df(self.viscosity_df, col_data, col_name)

        col_name = "Bulk Viscosity_Std"
        col_data = self.viscosity_df_slices[col_str].std(axis=1).values
        self.viscosity_df = self.add_to_df(self.viscosity_df, col_data, col_name)

        for _, ax1 in enumerate(dim_lbl):
            for _, ax2 in enumerate(dim_lbl):
                for _, eta_str in enumerate(eta_str_list):
                    col_str = [eta_str + f" {ax1}{ax2}_slice {isl}" for isl in range(observable.no_slices)]
                    col_name = eta_str + f" {ax1}{ax2}_Mean"
                    col_data = self.viscosity_df_slices[col_str].mean(axis=1).values
                    self.viscosity_df = self.add_to_df(self.viscosity_df, col_data, col_name)

                    col_name = eta_str + f" {ax1}{ax2}_Std"
                    col_data = self.viscosity_df_slices[col_str].std(axis=1).values
                    self.viscosity_df = self.add_to_df(self.viscosity_df, col_data, col_name)

        list_coord = ["xy", "xz", "yx", "yz", "zx", "zy"]
        col_str = [eta_str + f" {coord}_Mean" for coord in list_coord]
        # Mean
        col_data = self.viscosity_df[col_str].mean(axis=1).values
        self.viscosity_df = self.add_to_df(self.viscosity_df, col_data, "Shear Viscosity_Mean")
        # Std
        col_data = self.viscosity_df[col_str].std(axis=1).values
        self.viscosity_df = self.add_to_df(self.viscosity_df, col_data, "Shear Viscosity_Std")

        # Time stamp
        tend = self.timer.current()
        self.time_stamp("Viscosities Calculation", self.timer.time_division(tend - t0))

        # Save
        self.viscosity_df, self.viscosity_df_slices = self.save_hdf(
            df=self.viscosity_df, df_slices=self.viscosity_df_slices, tc_name="viscosities"
        )

        # Plot
        plot_quantities: list = ["Bulk Viscosity", "Shear Viscosity"]
        shear_list_coord = ["xyxy", "xzxz", "yxyx", "yzyz", "zxzx", "zyzy"]
        if plot or display_plot:
            # Make the plot
            for ipq, pq in enumerate(plot_quantities):
                if pq == "Bulk Viscosity":
                    acf_str = "Delta Pressure ACF"
                    acf_avg = observable.dataframe_acf[("Delta Pressure ACF", "Mean")]
                    acf_std = observable.dataframe_acf[("Delta Pressure ACF", "Std")]
                elif pq == "Shear Viscosity":
                    # The axis are the last two elements in the string
                    acf_strs = [(f"Pressure Tensor ACF {coord}", "Mean") for coord in shear_list_coord]
                    acf_avg = observable.dataframe_acf[acf_strs].mean(axis=1)
                    acf_std = observable.dataframe_acf[acf_strs].std(axis=1)

                tc_avg = self.viscosity_df[(pq, "Mean")]
                tc_std = self.viscosity_df[(pq, "Std")]

                self.plot_tc(
                    time=self.time_array,
                    acf_data=column_stack((acf_avg, acf_std)),
                    tc_data=column_stack((tc_avg, tc_std)),
                    acf_name=acf_str,
                    tc_name=pq,
                    figname=f"{pq}_Plot.png",
                    show=display_plot,
                )
