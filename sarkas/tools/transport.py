"""
Transport Module.
"""

import inspect
from copy import deepcopy
from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import sys
from matplotlib.pyplot import subplots
from numpy import array, column_stack, ndarray, pi, rint
from os import mkdir as os_mkdir
from os import remove as os_remove
from os.path import exists as os_path_exists
from os.path import join as os_path_join
from pandas import DataFrame, MultiIndex, read_hdf
from scipy.integrate import cumulative_trapezoid
from warnings import warn

from ..utilities.io import print_to_logger
from ..utilities.maths import fast_integral_loop
from ..utilities.misc import add_col_to_df
from ..utilities.timing import datetime_stamp, SarkasTimer

# Sarkas Modules
from .observables import plot_labels, Thermodynamics


class TransportCoefficients:
    """Transport Coefficients parent."""

    def __init__(self):
        self.time_array = None
        self.dataframe = None
        self.dataframe_slices = None
        self.saving_dir = None
        #
        # To be copied from parameters class
        self.postprocessing_dir = None
        self.units = None
        self.job_id = None
        self.verbose = None
        self.dt = None
        self.units_dict = None
        self.total_plasma_frequency = None
        self.dimensions = None
        self.box_volume = None
        self.pbox_volume = None
        self.phase = None
        self.no_slices = None
        self.dump_step = None

        self.kB = 0.0
        self.beta_slices = 0.0
        #
        self.log_file = None
        self.df_fnames = {}
        self.timer = SarkasTimer()

    def setup(self, params, observable, thermodynamics):
        """
        Set up the necessary parameters and structures for computing transport coefficients.
        This includes copying simulation parameters, creating directories, and initializing dataframes.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            An instance of the Parameters class containing simulation parameters.

        observable : :class:`sarkas.tools.observables.Observable`
            An instance of the Observable class. This class is used for obtaining the autocorrelation
            function datasets necessary for the transport coefficient calculations.

        thermodynamics : :class:`sarkas.tools.observables.Thermodynamics`
            An instance of the Thermodynamics class, which is used for calculating the average temperature
            of each block in the simulation. If not provided, the average temperature is calculated using
            the :meth:`calculate_average_temperature` method.

        Notes
        -----
        - The method initializes various attributes of the class based on the provided `params`,
        `observable`, and `thermodynamics` (if provided).
        - It also sets up logging and prints initial information about the setup process.
        - If `thermodynamics` is not provided, it calculates the average temperature using available data.

        See Also
        --------
        `get_observable_data` : Method to retrieve autocorrelation function datasets.
        `calculate_average_temperature` : Method to calculate the average temperature if `thermodynamics` is not provided.
        `make_directories, create_df_filenames` : Methods for setting up directory structures and filenames for saving data.
        """
        #
        self.copy_params(params=params)
        self.postprocessing_dir = self.directory_tree["postprocessing"]["path"]
        self.get_observable_data(observable)

        thermodynamics.calculate_beta_slices()
        self.beta_slices = thermodynamics.beta_slices.copy()

        self.make_directories()
        self.create_df_filenames()
        self.log_file = os_path_join(self.saving_dir, f"{self.__name__}_logfile.out")

        datetime_stamp(self.log_file)
        self.pretty_print()

    def copy_params(self, params):
        for i, val in params.__dict__.items():
            if not inspect.ismethod(val):
                if isinstance(val, dict):
                    self.__dict__[i] = deepcopy(val)
                elif isinstance(val, ndarray):
                    self.__dict__[i] = val.copy()
                else:
                    self.__dict__[i] = val

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = f"{self.__name__}( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def calculate_average_temperature(self, params):
        """
        Calculate the average temperature from the :class:`sarkas.tools.observables.Thermodynamics` data.
        It updates the :attr:`beta_slices` attribute with the inverse temperature calculated from the thermodynamics data.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            Simulation parameters used for setting up the thermodynamics calculations.

        Notes
        -----
        The method creates an instance of :class:`Thermodynamics`, sets it up with the given parameters,
        parses the data (without ACF data), and then calculates beta slices (inverse temperature).
        The average temperature is computed, but only the beta slices are stored in the `beta_slices` attribute.
        """

        energy = Thermodynamics()
        energy.setup(params, phase=self.phase)
        energy.parse(acf_data=False)
        energy.calculate_beta_slices()
        # self.T_avg = energy.dataframe["Temperature"].mean()
        self.beta_slices = energy.beta_slices.copy()

    def initialize_dataframes(self, observable):
        """
        Grab observables autocorrelation data and initialize the dataframes where to store the data.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.Observable`
            Observable object containing the ACF whose time integral leads to the desired transport coefficient.

        """
        time_col_name = observable.dataframe_acf.columns[0]
        self.time_array = observable.dataframe_acf[time_col_name].values

        self.dataframe = DataFrame()
        self.dataframe_slices = DataFrame()
        self.dataframe["Integration_Interval"] = self.time_array.copy()
        self.dataframe_slices["Integration_Interval"] = self.time_array.copy()

    def create_df_filenames(self):
        """
        Creates and stores the paths for the filenames of the dataframes used in the analysis.
        These filenames are based on the instance's name and job ID.

        Notes
        -----
        The method constructs filenames for :attr:`dataframe_slices` and :attr:`dataframe`, combining the instance's
        name, job ID, and file format. These filenames are stored in the :attr:`df_fnames` dictionary attribute.
        """
        fnames = {}
        fnames["dataframe_slices"] = os_path_join(self.saving_dir, f"{self.__name__}_slices_{self.job_id}.h5")
        fnames["dataframe"] = os_path_join(self.saving_dir, f"{self.__name__}_{self.job_id}.h5")

        self.df_fnames = fnames

    def make_directories(self):
        """
        Creates necessary directories for saving transport coefficients.
        The directories are structured to organize the data by postprocessing type, transport coefficient name,
        and simulation phase.

        Notes
        -----
        This method creates a hierarchy of directories, if they do not already exist, in the following structure:
        'postprocessing_dir/TransportCoefficients/{self.__name__}/{self.phase.capitalize()}'. The final directory path
        is stored in the :attr:`saving_dir` attribute.
        """
        transport_dir = os_path_join(self.postprocessing_dir, "TransportCoefficients")
        if not os_path_exists(transport_dir):
            os_mkdir(transport_dir)

        coeff_dir = os_path_join(transport_dir, self.__name__)
        if not os_path_exists(coeff_dir):
            os_mkdir(coeff_dir)

        self.saving_dir = os_path_join(coeff_dir, self.phase.capitalize())
        if not os_path_exists(self.saving_dir):
            os_mkdir(self.saving_dir)

    def diffusion(self, observable, plot: bool = True, display_plot: bool = False):
        """
        Calculate the transport coefficient from the Green-Kubo formula.

        Raises
        ------
            : DeprecationWarning
        """

        warn(
            "Deprecated feature. It will be removed in a future release.\nEach transport coefficient is now a class. Create an instance of the class Diffusion and then pass the same parameters to `Diffusion.compute()`.",
            category=DeprecationWarning,
        )

    def electrical_conductivity(
        self,
        observable,
        plot: bool = True,
        display_plot: bool = False,
    ):
        """
        Calculate the transport coefficient from the Green-Kubo formula.

        Raises
        ------
            : DeprecationWarning
        """

        warn(
            "Deprecated feature. It will be removed in a future release.\nEach transport coefficient is now a class. Create an instance of the class `ElectricalConductivity` and then pass the same parameters to `ElectricalConductivity.compute()`.",
            category=DeprecationWarning,
        )

    def interdiffusion(self, observable, plot: bool = True, display_plot: bool = False):
        """
        Calculate the transport coefficient from the Green-Kubo formula

        Raises
        ------
            : DeprecationWarning
        """

        warn(
            "Deprecated feature. It will be removed in a future release.\nEach transport coefficient is now a class. Create an instance of the class InterDiffusion and then pass the same parameters to `InterDiffusion.compute()`.",
            category=DeprecationWarning,
        )

    def viscosity(self, observable, plot: bool = True, display_plot: bool = False):
        """
        Calculate the transport coefficient from the Green-Kubo formula

        Raises
        ------
            : DeprecationWarning
        """

        warn(
            "Deprecated feature. It will be removed in a future release.\nEach transport coefficient is now a class. Create an instance of the class Viscosity and then pass the same parameters to `Viscosity.compute()`.",
            category=DeprecationWarning,
        )

    def get_observable_data(self, observable):
        """
        Retrieves autocorrelation function datasets for a given observable.
        This method calls the :meth:`sarkas.tools.observables.Observable.parse_acf` method of the provided observable
        object to obtain relevant data such as the phase, number of slices,
        autocorrelation function slice steps, and the dump step.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.Observable`
            An instance of the Observable class from which the autocorrelation
            function datasets will be obtained.
        """
        # Check that the phase and no_slices is the same from the one computed in the Observable
        observable.parse_acf()

        self.phase = observable.phase
        self.no_slices = observable.no_slices
        self.block_length = observable.block_length
        self.timesteps_per_block = observable.timesteps_per_block
        self.timesteps_per_plasma_period = observable.timesteps_per_plasma_period
        self.plasma_period = observable.plasma_period

        self.dump_step = observable.dump_step

    def parse(self):
        """Reads the HDF files containing the transport coefficients data."""

        self.dataframe = read_hdf(self.df_fnames["dataframe"], mode="r", index_col=False)
        self.dataframe_slices = read_hdf(self.df_fnames["dataframe_slices"], mode="r", index_col=False)

        # Print some info
        self.pretty_print()

    def plot_tc(self, time, acf_data, tc_data, acf_name, tc_name, figname, show: bool = False):
        """
        Create dual plots for Autocorrelation Function (ACF) and transport coefficient.
        The method plots both the mean and standard deviation of the ACF and transport coefficient
        data against lag time and the array index.

        Parameters
        ----------
        time : numpy.ndarray
            Array of time values.

        acf_data : numpy.ndarray
            Mean and Standard Deviation of the ACF.
            Shape = (:attr:`sarkas.tools.observables.Observable.acf_slice_steps`, 2).

        tc_data : numpy.ndarray
            Mean and Standard Deviation of the transport coefficient.
            Shape = (:attr:`sarkas.tools.observables.Observable.acf_slice_steps`, 2).

        acf_name : str
            Y-label for the ACF plot.

        tc_name : str
            Y-label for the transport coefficient plot.

        figname : str
            Filename for saving the plot.

        show : bool
            If True, the plot will be displayed. Useful for IPython or terminal environments.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object of the plot.

        (ax1, ax2, ax3, ax4) : tuple
            A tuple containing the axes objects. ax1 and ax2 are the primary axes for the ACF
            and transport coefficient plots, respectively. ax3 and ax4 are the twin axes (twiny())
            for ax1 and ax2, used for alternative representations or scales.

        Notes
        -----
        The method computes and applies appropriate scaling and labeling for both x and y axes.
        It also sets the axis limits and scales (logarithmic in this case). The method optionally saves
        the figure to the specified directory and displays it based on the 'show' parameter.
        """
        # Make the plot
        fig, (ax1, ax2) = subplots(1, 2, figsize=(16, 7))
        ax3 = ax1.twiny()
        ax4 = ax2.twiny()

        # Calculate axis multipliers and labels
        xmul, ymul, _, _, xlbl, ylbl = plot_labels(time, tc_data[:, 0], "Time", self.__long_name__, self.units)

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
        ax1.set(xlim=xlims, xscale="log", ylim=(-0.5, 1.1), ylabel=acf_name, xlabel=r"Time difference" + xlbl)
        xlims = (xmul * time[1], xmul * time[-1] * 1.05)
        ax2.set(xlim=xlims, ylim=(-0.05, ax2.get_ylim()[1]), ylabel=tc_name + ylbl, xlabel=r"$\tau$" + xlbl, xscale="log")

        # ax1.legend(loc='best')
        # ax2.legend(loc='best')
        # Finish the index axes
        ax3.set(xlim=(1, len(time) * 1.5), xscale="log")
        ax4.set(xlim=(1, len(time) * 1.5), xscale="log")
        for axi in [ax3, ax4]:
            axi.grid(alpha=0.1)
            axi.set(xlabel="Index")

        fig.tight_layout()
        fig.savefig(os_path_join(self.saving_dir, figname))

        if show:
            fig.show()

        return fig, (ax1, ax2, ax3, ax4)

    def pretty_print_msg(self, info: str = None, append_info: str = None):
        """
        Generate and return a formatted message containing information about data storage locations,
        number of slices, dumps per slice, and time intervals related to plasma frequency in a simulation.
        The message includes the long name of the current instance, paths to dataframes, the total time interval of the autocorrelation function, and its step size in both standard units and plasma periods.

        Parameters
        ----------
        info : str, optional
            A string containing information to be included in the print out. If provided, the method will use this
            as the base information otherwise, it will generate default information with detailed simulation data.

        append_info : str, optional
            A string containing additional information to be appended at the end of the default generated information print out. Useful for adding
            extra details or notes to the base information.

        Returns
        -------
        msg : str
            A formatted string containing detailed information about the data storage, time intervals, and other relevant
            details of the simulation.
        """

        tc_name = self.__long_name__

        # Create the message to print
        dtau = self.dt * self.dump_step
        tau = dtau * self.block_length
        # t_wp = 2.0 * pi / self.total_plasma_frequency  # Plasma period
        tau_wp = rint(tau / self.plasma_period).astype(int)
        if info:
            msg = info
        else:
            msg = (
                f"\n\n {tc_name:=^70} \n"
                f"Data saved in: \n {self.df_fnames['dataframe']} \n {self.df_fnames['dataframe_slices']} \n"
                f"No. of slices = {self.no_slices}\n"
                f"No. dumps per block = {self.block_length}\n"
                f"Total time interval of autocorrelation function: tau = {tau:.4e} {self.units_dict['time']} ~ {tau_wp} plasma periods\n"
                f"Time interval step: dtau = {dtau:.4e} ~ {dtau / self.plasma_period:.4e} plasma periods"
            )
            if append_info:
                msg += append_info

        return msg

    def pretty_print(self, info: str = None, append_info: str = None):
        """
        Print a formatted message, containing simulation data and additional information,
        to both a log file and the screen, based on the verbosity settings.
        This method constructs the message using `pretty_print_msg` and then outputs it
        using the `print_to_logger` function.

        Parameters
        ----------
        info : str, optional
            Initial information to be included in the printed message. This can be a
            custom message provided by the user. If `None`, `pretty_print_msg` will
            generate a default message based on the simulation data.

        append_info : str, optional
            Additional information to be appended to the end of the message. This allows
            for extra details or notes to be included in the final printed output.

        Notes
        -----
        - The message is printed to the log file specified in `self.log_file`.
        - The verbosity of the printing to the screen is controlled by `self.verbose`.
        If `self.verbose` is `True`, the message will also be printed to the screen;
        otherwise, it will only be logged to the file.

        """
        msg = self.pretty_print_msg(info, append_info)
        # Print the message to log file and screen
        print_to_logger(message=msg, log_file=self.log_file, print_to_screen=self.verbose)

    def save_hdf(self):
        """
        Save the dataframes `dataframe_slices` and `dataframe` as HDF files to disk.
        This method is designed to store the data in the 'TransportCoefficient' directory within the 'PostProcessing` directory.

        This method first checks if the HDF files already exist in the specified paths
        (contained in `self.df_fnames`). If they exist, the existing files are removed
        before saving the new data. The columns of both dataframes are converted into
        MultiIndex format based on the column names, and then the dataframes are sorted
        by these indices before being saved.

        Note
        ----
        The method currently employs a workaround where existing HDF files are deleted
        before writing new ones. Future improvements may include updating the existing
        HDF files directly without the need to delete them first.

        Warnings
        --------
        - This method will overwrite any existing HDF files with the same names in the
        specified save location.
        - It assumes the column names in the dataframes are structured in a specific
        format suitable for splitting into tuples for MultiIndex.
        """

        # TODO: Fix this hack. We should be able to add data to HDF instead of removing it and rewriting it.
        # Save the data.
        if os_path_exists(self.df_fnames["dataframe_slices"]):
            os_remove(self.df_fnames["dataframe_slices"])

        self.dataframe_slices.columns = MultiIndex.from_tuples(
            [tuple(c.split("_")) for c in self.dataframe_slices.columns]
        )
        self.dataframe_slices = self.dataframe_slices.sort_index()
        self.dataframe_slices.to_hdf(self.df_fnames["dataframe_slices"], mode="w", key=self.__name__, index=False)

        # Save the data.
        if os_path_exists(self.df_fnames["dataframe"]):
            os_remove(self.df_fnames["dataframe"])

        self.dataframe.columns = MultiIndex.from_tuples([tuple(c.split("_")) for c in self.dataframe.columns])
        self.dataframe = self.dataframe.sort_index()
        self.dataframe.to_hdf(self.df_fnames["dataframe"], mode="w", key=self.__name__, index=False)

    def time_stamp(self, message: str, timing: tuple):
        """
        Print a given message along with formatted elapsed time to a log file and,
        optionally, to the screen based on the verbosity setting of the instance.
        The elapsed time is detailed in hours, minutes, seconds, milliseconds,
        microseconds, and nanoseconds.

        If the elapsed time is less than or equal to 2 seconds, the output includes
        the finer resolution of milliseconds, microseconds, and nanoseconds. For
        longer durations, only hours, minutes, and seconds are included.

        Parameters
        ----------
        message : str
            The message to be printed before the elapsed time. This is intended to
            provide context or description for the timing being reported.

        timing : tuple
            A tuple containing the elapsed time details in the order of hours (hrs),
            minutes (min), seconds (sec), milliseconds (msec), microseconds (usec),
            and nanoseconds (nsec).

        Notes
        -----
        - The method writes to the log file specified in `self.log_file`.
        - The verbosity of the output is controlled by `self.verbose`. If `self.verbose`
            is `True`, the message will be printed both to the log file and the screen.
            Otherwise, it will only be logged to the file.
        - The method temporarily redirects the standard output to the log file,
            then restores it after logging is completed.
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


class Diffusion(TransportCoefficients):
    """
    The diffusion coefficient is calculated from the Green-Kubo formula

    .. math::

            D_{\\alpha} = \\frac{1}{3 N_{\\alpha}} \\sum_{i}^{N_{\\alpha}} \\int_0^{\\tau} dt \\,
                \\langle \\mathbf v^{(\\alpha)}_{i}(t) \\cdot  \\mathbf v^{(\\alpha)}_{i}(0) \\rangle.

    where :math:`\\mathbf v_{i}^{(\\alpha)}(t)` is the velocity of particle :math:`i` of species
    :math:`\\alpha`. Notice that the diffusion coefficient is averaged over all :math:`N_{\\alpha}` particles.

    Data is retrievable at :attr:`~.dataframe` and :attr:`~.dataframe_slices`.

    """

    def __init__(self):
        self.__name__ = "Diffusion"
        self.__long_name__ = "Diffusion Coefficients"
        self.required_observable = "Velocity Autocorrelation Function"
        super().__init__()

    def compute(self, observable, plot: bool = True, display_plot: bool = False):
        """
        Calculate the transport coefficient from the Green-Kubo formula.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.VelocityAutoCorrelationFunction`
            Observable object containing the ACF whose time integral leads to the self diffusion coefficient.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.

        """
        # Write Log File
        observable.parse_acf()
        self.initialize_dataframes(observable)

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
                    integrand = observable.dataframe_acf_slices[(sp_vacf_str, "Total", f"slice {isl}")].values
                    df_str = f"{sp} Diffusion_slice {isl}"
                    self.dataframe_slices[df_str] = const * fast_integral_loop(time=self.time_array, integrand=integrand)

            # Average and std of each diffusion coefficient.
            for isp, sp in enumerate(observable.species_names):
                col_str = [f"{sp} Diffusion_slice {isl}" for isl in range(observable.no_slices)]
                # Mean
                col_data = self.dataframe_slices[col_str].mean(axis=1).values
                col_name = f"{sp} Diffusion_Mean"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
                # Std
                col_data = self.dataframe_slices[col_str].std(axis=1).values
                col_name = f"{sp} Diffusion_Std"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

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
                    self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

                    # Perpendicular
                    x_vacf_str = (sp_vacf_str, "X", f"slice {isl}")
                    y_vacf_str = (sp_vacf_str, "Y", f"slice {isl}")

                    integrand_perp = 0.5 * (
                        observable.dataframe_acf_slices[x_vacf_str].to_numpy()
                        + observable.dataframe_acf_slices[y_vacf_str].to_numpy()
                    )
                    col_data = fast_integral_loop(time=self.time_array, integrand=integrand_perp)
                    col_name = f"{sp} Diffusion_Perpendicular_slice {isl}"
                    self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

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
                col_data = self.dataframe_slices[par_col_str].mean(axis=1).values
                col_name = sp_diff_str + "_Parallel_Mean"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
                # Std
                col_data = self.dataframe_slices[par_col_str].std(axis=1).values
                col_name = sp_diff_str + "_Parallel_Std"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

                # Mean
                col_data = self.dataframe_slices[perp_col_str].mean(axis=1).values
                col_name = sp_diff_str + "_Perpendicular_Mean"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
                # Std
                col_data = self.dataframe_slices[perp_col_str].std(axis=1).values
                col_name = sp_diff_str + "_Perpendicular_Std"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

            # Save the updated dataframe
            observable.save_hdf()
            # Endif magnetized.

        # Time stamp
        tend = self.timer.current()
        self.time_stamp("Diffusion Calculation", self.timer.time_division(tend - t0))

        # Save
        self.save_hdf()
        # Plot
        if plot:
            _, _ = self.plot(observable, display_plot=display_plot)

    def plot(self, observable, display_plot: bool = False):
        """Make a dual plot comparing the ACF and the Transport Coefficient by using the :meth:`plot_tc` method.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.VelocityAutoCorrelationFunction`
            Observable object containing the ACF whose time integral leads to the self diffusion coefficient.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.

        Return
        ------
        figs: dict
            Dictionary of `matplotlib` figure handles for each species. If the system is magnetized then it returns a nested dictionary.
            Each `figs[species_name]` is a dictionary with keys `Parallel` and `Perpendicular`.

        axes: dict,
            Dictionary of tuples containing the axes handles for each element of `figs`. Each element of `axes` is a tuple of four axes handles.
            'ax1` and `ax2` are the handles for the left and right plots respectively.
            `ax3` and `ax4` are the handles for the "Index" axes, created from `ax1.twiny()` and `ax2.twiny()` respectively.\n
            If the system is magnetized, then it returns a nested dictionary.
            Each `axes[species_name]` is a dictionary with keys `Parallel` and `Perpendicular` each containing a tuple of four axes handles.

        """
        vacf_str = "VACF"

        figs = {}
        axes = {}

        if observable.magnetized:
            for isp, sp in enumerate(observable.species_names):
                sp_vacf_str = f"{sp} " + vacf_str
                sp_diff_str = f"{sp} Diffusion"

                # Parallel
                acf_avg = observable.dataframe_acf[(sp_vacf_str, "Parallel", "Mean")].to_numpy()
                acf_std = observable.dataframe_acf[(sp_vacf_str, "Parallel", "Std")].to_numpy()

                tc_avg = self.dataframe[(sp_diff_str, "Parallel", "Mean")].to_numpy()
                tc_std = self.dataframe[(sp_diff_str, "Parallel", "Std")].to_numpy()

                fig, (ax1, ax2, ax3, ax4) = self.plot_tc(
                    time=self.time_array,
                    acf_data=column_stack((acf_avg, acf_std)),
                    tc_data=column_stack((tc_avg, tc_std)),
                    acf_name=sp_vacf_str + " Parallel",
                    tc_name=sp_diff_str + " Parallel",
                    figname=f"{sp}_Parallel_Diffusion_Plot.png",
                    show=display_plot,
                )
                figs[sp] = {"Parallel": fig}
                axes[sp] = {"Parallel": (ax1, ax2, ax3, ax4)}
                # Perpendicular
                acf_avg = observable.dataframe_acf[(sp_vacf_str, "Perpendicular", "Mean")]
                acf_std = observable.dataframe_acf[(sp_vacf_str, "Perpendicular", "Std")]

                tc_avg = self.dataframe[(sp_diff_str, "Perpendicular", "Mean")]
                tc_std = self.dataframe[(sp_diff_str, "Perpendicular", "Std")]

                fig, (ax1, ax2, ax3, ax4) = self.plot_tc(
                    time=self.time_array,
                    acf_data=column_stack((acf_avg, acf_std)),
                    tc_data=column_stack((tc_avg, tc_std)),
                    acf_name=sp_vacf_str + " Perpendicular",
                    tc_name=sp_diff_str + " Perpendicular",
                    figname=f"{sp}_Perpendicular_Diffusion_Plot.png",
                    show=display_plot,
                )
                figs[sp]["Perpendicular"] = fig
                axes[sp]["Perpendicular"] = (ax1, ax2, ax3, ax4)
        else:
            for isp, sp in enumerate(observable.species_names):
                sp_vacf_str = f"{sp} " + vacf_str
                acf_avg = observable.dataframe_acf[(sp_vacf_str, "Total", "Mean")].to_numpy()
                acf_std = observable.dataframe_acf[(sp_vacf_str, "Total", "Std")].to_numpy()

                d_str = f"{sp} Diffusion"
                tc_avg = self.dataframe[(d_str, "Mean")].to_numpy()
                tc_std = self.dataframe[(d_str, "Std")].to_numpy()

                fig, (ax1, ax2, ax3, ax4) = self.plot_tc(
                    time=self.time_array,
                    acf_data=column_stack((acf_avg, acf_std)),
                    tc_data=column_stack((tc_avg, tc_std)),
                    acf_name=sp_vacf_str,
                    tc_name=d_str,
                    figname=f"{sp}_Diffusion_Plot.png",
                    show=display_plot,
                )
                figs[sp] = fig
                axes[sp] = (ax1, ax2, ax3, ax4)

        return figs, axes


class InterDiffusion(TransportCoefficients):
    """
    The interdiffusion coefficient is calculated from the Green-Kubo formula

    .. math::

            D_{\\alpha} = \\frac{1}{3Nx_1x_2} \\int_0^\\tau dt
            \\langle \\mathbf {J}_{\\alpha}(0) \\cdot \\mathbf {J}_{\\alpha}(t) \\rangle,

    where :math:`x_{1,2}` are the concentration of the two species and
    :math:`\\mathbf {J}_{\\alpha}(t)` is the diffusion current calculated by the
    :class:`sarkas.tools.observables.DiffusionFlux` class.

    Data is retrievable at :attr:`~.dataframe` and :attr:`~.dataframe_slices`.

    """

    def __init__(self):
        self.__name__ = "InterDiffusion"
        self.__long_name__ = "InterDiffusion Coefficients"
        self.required_observable = "Diffusion Flux"
        super().__init__()

    def compute(self, observable, plot: bool = True, display_plot: bool = False):
        """
        Calculate the transport coefficient from the Green-Kubo formula

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.DiffusionFlux`
            Observable object containing the ACF whose time integral leads to the interdiffusion coefficient.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False

        """
        observable.parse_acf()
        self.initialize_dataframes(observable)

        no_fluxes_acf = observable.no_fluxes_acf
        # Normalization constant
        const = 1.0 / (3.0 * observable.total_num_ptcls * observable.species_concentrations.prod())

        df_str = "Diffusion Flux ACF"
        id_str = "Inter Diffusion Flux"
        # Time
        t0 = self.timer.current()
        for isl in tqdm(range(self.no_slices), disable=not self.verbose):
            # D_ij = zeros((no_fluxes_acf, jc_acf.acf_slice_steps))

            for ij in range(no_fluxes_acf):
                acf_df_str = (df_str + f" {ij}", "Total", f"slice {isl}")
                integrand = observable.dataframe_acf_slices[acf_df_str].to_numpy()

                col_data = const * fast_integral_loop(time=self.time_array, integrand=integrand)
                col_name = id_str + f" {ij}_slice {isl}"
                self.dataframe_slices = add_col_to_df(observable, col_data, col_name)

        # Average and Std of slices
        for ij in range(no_fluxes_acf):
            col_str = [id_str + f" {ij}_slice {isl}" for isl in range(self.no_slices)]
            # Mean
            col_data = self.dataframe_slices[col_str].mean(axis=1).values
            col_name = id_str + f" {ij}_Mean"
            self.dataframe = add_col_to_df(observable, col_data, col_name)
            # Mean
            col_data = self.dataframe_slices[col_str].std(axis=1).values
            col_name = id_str + f" {ij}_Std"
            self.dataframe = add_col_to_df(observable, col_data, col_name)

        # Time stamp
        tend = self.timer.current()
        self.time_stamp("Interdiffusion Calculation", self.timer.time_division(tend - t0))

        # Save
        self.save_hdf()
        # Plot
        if plot:
            _, _ = self.plot(observable, display_plot=display_plot)

    def plot(self, observable, display_plot: bool = False):
        """Make a dual plot comparing the ACF and the Transport Coefficient by using the :meth:`plot_tc` method.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.DiffusionFlux`
            Observable object containing the ACF whose time integral leads to the self diffusion coefficient.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.

        Return
        ------
        figs: dict
            Dictionary of `matplotlib` figure handles for each flux. Keys are `flux_0`, `flux_1`, etc..

        axes: dict
            Dictionary of tuples containing the axes handles for each element of `figs`. Keys are the same as in `figs`. Each element of `axes` is a tuple of four axes handles. 'ax1` and `ax2` are the handles for the left and right plots respectively.
            `ax3` and `ax4` are the handles for the "Index" axes, created from `ax1.twiny()` and `ax2.twiny()` respectively.\n
        """

        df_str = "Diffusion Flux ACF"
        id_str = "Inter Diffusion Coefficient Flux"

        figs = {}
        axes = {}

        for flux in range(observable.no_fluxes_acf):
            flux_str = f"{df_str} {flux}"
            acf_avg = observable.dataframe_acf[(flux_str, "Total", "Mean")].to_numpy()
            acf_std = observable.dataframe_acf[(flux_str, "Total", "Std")].to_numpy()

            d_str = f"{id_str} {flux}"
            tc_avg = self.dataframe[(d_str, "Mean")].to_numpy()
            tc_std = self.dataframe[(d_str, "Std")].to_numpy()

            fig, (ax1, ax2, ax3, ax4) = self.plot_tc(
                time=self.time_array,
                acf_data=column_stack((acf_avg, acf_std)),
                tc_data=column_stack((tc_avg, tc_std)),
                acf_name=flux_str,
                tc_name=d_str,
                figname=f"InterDiffusion_Flux{flux}_Plot.png",
                show=display_plot,
            )
            figs[f"Flux_{flux}"] = fig
            axes[f"Flux_{flux}"] = (ax1, ax2, ax3, ax4)

        return figs, axes


class Viscosity(TransportCoefficients):
    """Viscosisty coefficients class.

    The shear viscosity is obtained from the Green-Kubo formula

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

    Data is retrievable at :attr:`~.dataframe` and :attr:`~.dataframe_slices`.

    """

    def __init__(self):
        self.__name__ = "Viscosities"
        self.__long_name__ = "Viscosity"
        self.required_observable = "Pressure Tensor"
        super().__init__()

    def compute(self, observable, plot: bool = True, display_plot: bool = False):
        """
        Calculate the transport coefficient from the Green-Kubo formula.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.PressureTensor`
            Observable object containing the ACF whose time integral leads to the viscsosity coefficients.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False

        """
        observable.parse_acf()
        self.initialize_dataframes(observable)
        # Initialize Timer
        t0 = self.timer.current()

        if observable.kinetic_potential_division:
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
                "Shear Viscosity Tensor",
            ]
        else:
            pt_str_list = ["Pressure Tensor ACF"]
            eta_str_list = ["Shear Viscosity Tensor"]

        start_steps = 0
        end_steps = 0
        time_steps = len(self.time_array)
        for isl in tqdm(range(self.no_slices), disable=not observable.verbose):
            end_steps += time_steps

            const = observable.box_volume * self.beta_slices[isl]
            # Calculate Bulk Viscosity
            # It is calculated from the fluctuations of the pressure eq. 2.124a Allen & Tilsdeley
            integrand = observable.dataframe_acf_slices[(f"Pressure Bulk ACF", f"slice {isl}")].to_numpy()

            col_name = f"Bulk Viscosity_slice {isl}"
            col_data = const * cumulative_trapezoid(integrand, x=self.time_array, initial=0.0)
            self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

            # Calculate the Shear Viscosity Elements
            for iax, ax1 in enumerate(observable.dim_labels):
                for _, ax2 in enumerate(observable.dim_labels[iax:], iax):
                    for _, (pt_str, eta_str) in enumerate(zip(pt_str_list, eta_str_list)):
                        pt_str_temp = (pt_str + f" {ax1}{ax2}{ax1}{ax2}", f"slice {isl}")
                        integrand = observable.dataframe_acf_slices[pt_str_temp].to_numpy()
                        col_name = eta_str + f" {ax1}{ax2}_slice {isl}"
                        col_data = const * cumulative_trapezoid(
                            integrand, x=self.time_array, initial=0.0
                        )  # fast_integral_loop(self.time_array, integrand)
                        self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

            start_steps += time_steps

        # Now average the slices
        col_str = [f"Bulk Viscosity_slice {isl}" for isl in range(observable.no_slices)]

        col_name = "Bulk Viscosity_Mean"
        col_data = self.dataframe_slices[col_str].mean(axis=1).values
        self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

        col_name = "Bulk Viscosity_Std"
        col_data = self.dataframe_slices[col_str].std(axis=1).values
        self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

        for iax, ax1 in enumerate(observable.dim_labels):
            for _, ax2 in enumerate(observable.dim_labels[iax + 1 :], iax + 1):
                for _, eta_str in enumerate(eta_str_list):
                    col_str = [eta_str + f" {ax1}{ax2}_slice {isl}" for isl in range(observable.no_slices)]
                    col_name = eta_str + f" {ax1}{ax2}_Mean"
                    col_data = self.dataframe_slices[col_str].mean(axis=1).values
                    self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

                    col_name = eta_str + f" {ax1}{ax2}_Std"
                    col_data = self.dataframe_slices[col_str].std(axis=1).values
                    self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

        list_coord = ["XY", "XZ", "YZ"]
        col_str = [eta_str + f" {coord}_Mean" for coord in list_coord]
        # Mean
        col_data = self.dataframe[col_str].mean(axis=1).values
        self.dataframe = add_col_to_df(self.dataframe, col_data, "Shear Viscosity_Mean")
        # Std
        col_data = self.dataframe[col_str].std(axis=1).values
        self.dataframe = add_col_to_df(self.dataframe, col_data, "Shear Viscosity_Std")

        # Time stamp
        tend = self.timer.current()
        self.time_stamp("Viscosities Calculation", self.timer.time_division(tend - t0))

        # Save
        self.save_hdf()
        # Plot
        if plot:
            _, _ = self.plot(observable, display_plot=display_plot)

    def plot(self, observable, display_plot: bool = False):
        """Make a dual plot comparing the ACF and the Transport Coefficient by using the :meth:`plot_tc` method.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.PressureTensor`
            Observable object containing the ACF whose time integral leads to the self diffusion coefficient.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.

        Return
        ------
        figs: list, :class:`matplotlib.pyplot.Figure`
            List of `matplotlib` figure handles for the bulk and shear viscosity respectively.

        axes: list,
            List of tuples containing the axes handles for each element of `figs`. Each element of `axes` is a tuple of four axes handles. 'ax1` and `ax2` are the handles for the left and right plots respectively.
            `ax3` and `ax4` are the handles for the "Index" axes, created from `ax1.twiny()` and `ax2.twiny()` respectively.\n
        """

        # Plot
        plot_quantities = ["Bulk Viscosity", "Shear Viscosity"]
        shear_list_coord = ["XYXY", "XZXZ", "YZYZ"]

        figs = []
        axes = []
        # Make the plot
        for _, pq in enumerate(plot_quantities):
            if pq == "Bulk Viscosity":
                acf_str = "Pressure Bulk ACF"
                acf_avg = observable.dataframe_acf[("Pressure Bulk ACF", "Mean")]
                acf_std = observable.dataframe_acf[("Pressure Bulk ACF", "Std")]
            elif pq == "Shear Viscosity":
                # The axis are the last two elements in the string
                acf_str = "Stress Tensors ACF"
                acf_strs = [(f"Pressure Tensor ACF {coord}", "Mean") for coord in shear_list_coord]
                acf_avg = observable.dataframe_acf[acf_strs].mean(axis=1)
                acf_std = observable.dataframe_acf[acf_strs].std(axis=1)

            tc_avg = self.dataframe[(pq, "Mean")]
            tc_std = self.dataframe[(pq, "Std")]

            fig, (ax1, ax2, ax3, ax4) = self.plot_tc(
                time=self.time_array,
                acf_data=column_stack((acf_avg, acf_std)),
                tc_data=column_stack((tc_avg, tc_std)),
                acf_name=acf_str,
                tc_name=pq,
                figname=f"{pq}_Plot.png",
                show=display_plot,
            )
            figs.append(fig)
            axes.append((ax1, ax2, ax3, ax4))

        return figs, axes


class ElectricalConductivity(TransportCoefficients):
    """The electrical conductivity is calculated from the Green-Kubo formula

    .. math::

            \\sigma = \\frac{\\beta}{V} \\int_0^{\\tau} dt J(t).

    where :math:`\\beta = 1/k_B T` and :math:`V` is the volume of the simulation box.

    Data is retrievable at :attr:`~.dataframe` and :attr:`~.dataframe_slices`.

    """

    def __init__(self):
        self.__name__ = "ElectricalConductivity"
        self.__long_name__ = "Electrical Conductivity"
        self.required_observable = "Electric Current"
        super().__init__()

    def compute(
        self,
        observable,
        plot: bool = True,
        display_plot: bool = False,
    ):
        """
        Calculate the transport coefficient from the Green-Kubo formula


        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.ElectricCurrent`
            Observable object containing the ACF whose time integral leads to the electrical conductivity.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient.\n
            Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.
        """
        observable.parse_acf()
        self.initialize_dataframes(observable)

        # Time
        t0 = self.timer.current()

        jc_str = "Electric Current ACF"
        sigma_str = "Electrical Conductivity"
        const = self.beta_slices / observable.box_volume

        if not observable.magnetized:
            for isl in tqdm(range(observable.no_slices), disable=not observable.verbose):
                integrand = array(observable.dataframe_acf_slices[(jc_str, "Total", "slice {}".format(isl))])
                col_name = sigma_str + "_slice {}".format(isl)
                col_data = const[isl] * fast_integral_loop(self.time_array, integrand)
                self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

            col_str = [sigma_str + "_slice {}".format(isl) for isl in range(observable.no_slices)]
            # Mean
            col_data = self.dataframe_slices[col_str].mean(axis=1).values
            col_name = sigma_str + "_Mean"
            self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
            # Std
            col_data = self.dataframe_slices[col_str].std(axis=1).values
            col_name = sigma_str + "_Std"
            self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

        else:
            for isl in tqdm(range(observable.no_slices), disable=not observable.verbose):
                # Parallel
                par_str = (jc_str, "Z", f"slice {isl}")
                integrand = observable.dataframe_acf_slices[par_str].to_numpy()
                col_data = const * fast_integral_loop(self.time_array, integrand)
                col_name = sigma_str + f"_Parallel_slice {isl}"
                self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

                # Perpendicular
                x_col_str = (jc_str, "X", f"slice {isl}")
                y_col_str = (jc_str, "Y", f"slice {isl}")
                perp_integrand = 0.5 * (
                    observable.dataframe_acf_slices[x_col_str].to_numpy()
                    + observable.dataframe_acf_slices[y_col_str].to_numpy()
                )
                col_name = sigma_str + f"_Perpendicular_slice {isl}"
                col_data = const * fast_integral_loop(self.time_array, perp_integrand)
                self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

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
            col_data = self.dataframe_slices[col_str].mean(axis=1).values
            col_name = sigma_str + "_Parallel_Mean"
            self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
            # Std
            col_data = self.dataframe_slices[col_str].std(axis=1).values
            col_name = sigma_str + "_Parallel_Std"
            self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

            # Perpendicular
            col_str = [sigma_str + f"_Perpendicular_slice {isl}" for isl in range(observable.no_slices)]
            # Mean
            col_data = sigma_str + "_Perpendicular_Mean"
            col_name = self.dataframe_slices[col_str].mean(axis=1).values
            self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
            # Std
            col_data = sigma_str + "_Perpendicular_Std"
            col_name = self.dataframe_slices[col_str].std(axis=1).values
            self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

            # Endif magnetized.
        # Time stamp
        tend = self.timer.current()
        self.time_stamp(f"{self.__long_name__} Calculation", self.timer.time_division(tend - t0))

        # Save
        self.save_hdf()
        # Plot
        if plot:
            _, _ = self.plot(observable, display_plot=display_plot)

    def plot(self, observable, display_plot: bool = False):
        """Make a dual plot comparing the ACF and the Transport Coefficient by using the :meth:`plot_tc` method.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.VelocityAutoCorrelationFunction`
            Observable object containing the ACF whose time integral leads to the self diffusion coefficient.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.

        Return
        ------
        fig (fig_par, fig_perp) : :class:`matplotlib.pyplot.Figure`, tuple
            Matplotlib figure handle. If the system is magnetized then it return a tuple with the handles for the parallel (`fig_par`) and perpendicular (`fig_perp`) figures.

        (ax1, ax2, ax3, ax4), ((ax1_par, ax2_par, ax3_par, ax4_par), (ax1_perp, ax2_perp, ax3_perp, ax4_perp)): tuple, :class:`matplotlib.axes.Axes`
            Tuple containing the axes handles for `fig`. 'ax1` and `ax2` are the handles for the left and right plots respectively.
            `ax3` and `ax4` are the handles for the "Index" axes, created from `ax1.twiny()` and `ax2.twiny()` respectively.\n
            If the system is magnetized then it returns a tuple of tuples whose elements are the axes handles of each figure.

        """
        jc_str = "Electric Current ACF"
        sigma_str = "Electrical Conductivity"
        figs = []
        axes = []

        if not observable.magnetized:
            acf_avg = observable.dataframe_acf[(jc_str, "Total", "Mean")].to_numpy()
            acf_std = observable.dataframe_acf[(jc_str, "Total", "Std")].to_numpy()

            tc_avg = self.dataframe[(sigma_str, "Mean")].to_numpy()
            tc_std = self.dataframe[(sigma_str, "Std")].to_numpy()

            fig, (ax1, ax2, ax3, ax4) = self.plot_tc(
                time=self.time_array,
                acf_data=column_stack((acf_avg, acf_std)),
                tc_data=column_stack((tc_avg, tc_std)),
                acf_name="Electric Current ACF",
                tc_name="Electrical Conductivity",
                figname="ElectricalConductivity_Plot.png",
                show=display_plot,
            )

            figs.append[fig]
            axes.append[(ax1, ax2, ax3, ax4)]
        else:
            acf_avg = observable.dataframe_acf[(jc_str, "Parallel", "Mean")].to_numpy()
            acf_std = observable.dataframe_acf[(jc_str, "Parallel", "Std")].to_numpy()

            tc_avg = self.dataframe[(sigma_str, "Parallel", "Mean")].to_numpy()
            tc_std = self.dataframe[(sigma_str, "Parallel", "Std")].to_numpy()

            fig, (ax1, ax2, ax3, ax4) = self.plot_tc(
                time=self.time_array,
                acf_data=column_stack((acf_avg, acf_std)),
                tc_data=column_stack((tc_avg, tc_std)),
                acf_name="Electric Current ACF Parallel",
                tc_name="Electrical Conductivity Parallel",
                figname="ElectricalConductivity_Parallel_Plot.png",
                show=display_plot,
            )

            figs.append[fig]
            axes.append[(ax1, ax2, ax3, ax4)]

            acf_avg = observable.dataframe_acf[(jc_str, "Perpendicular", "Mean")].to_numpy()
            acf_std = observable.dataframe_acf[(jc_str, "Perpendicular", "Std")].to_numpy()

            tc_avg = self.dataframe[(sigma_str, "Perpendicular", "Mean")].to_numpy()
            tc_std = self.dataframe[(sigma_str, "Perpendicular", "Std")].to_numpy()

            fig, (ax1, ax2, ax3, ax4) = self.plot_tc(
                time=self.time_array,
                acf_data=column_stack((acf_avg, acf_std)),
                tc_data=column_stack((tc_avg, tc_std)),
                acf_name="Electric Current ACF Perpendicular",
                tc_name="Electrical Conductivity Perpendicular",
                figname="ElectricalConductivity_Perpendicular_Plot.png",
                show=display_plot,
            )
            figs.append[fig]
            axes.append[(ax1, ax2, ax3, ax4)]

        return figs, axes


class ThermalConductivity(TransportCoefficients):
    def __init__(self):
        self.__name__ = "ThermalConductivity"
        self.__long_name__ = "Thermal Conductivity"
        self.required_observable = "Heat Flux"
        super().__init__()

    def compute(
        self,
        observable,
        plot: bool = True,
        display_plot: bool = False,
    ):
        """
        Calculate the transport coefficient from the Green-Kubo formula.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.PressureTensor`
            Observable object containing the ACF whose time integral leads to the viscsosity coefficients.

        plot : bool, optional
            Flag for making the dual plot of the ACF and transport coefficient. Default = True.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False

        """
        observable.load_hdf_acf()
        self.initialize_dataframes(observable)
        # Initialize Timer
        t0 = self.timer.current()

        const = self.kB * self.beta_slices**2 / self.box_volume
        sp_vacf_str = f"{observable.__long_name__} ACF"

        if self.num_species > 1:
            species_list = [*observable.species_names, "all"]
        else:
            species_list = observable.species_names

        # Loop over time slices
        for isl in tqdm(range(self.no_slices), disable=not observable.verbose):
            # Iterate over the number of species
            for isp, sp1 in enumerate(species_list):
                for _, sp2 in enumerate(species_list[isp:], isp):
                    # Grab vacf data of each slice
                    integrand = observable.dataframe_acf_slices[
                        (sp_vacf_str, f"{sp1}-{sp2}", "Total", f"slice {isl}")
                    ].values
                    df_str = f"{self.__long_name__}_{sp1}-{sp2}_slice {isl}"
                    # self.dataframe_slices[df_str] = const[isl] * fast_integral_loop(
                    #     time=self.time_array, integrand=integrand
                    # )
                    data = const[isl] * cumulative_trapezoid(integrand, self.time_array, initial=0)
                    self.dataframe_slices = add_col_to_df(self.dataframe_slices, data, df_str)

        # Average and std of each transport coefficient.
        for isp, sp1 in enumerate(species_list):
            for _, sp2 in enumerate(species_list[isp:], isp):
                col_str = [f"{self.__long_name__}_{sp1}-{sp2}_slice {isl}" for isl in range(observable.no_slices)]
                # Mean
                col_data = self.dataframe_slices[col_str].mean(axis=1).values
                col_name = f"{self.__long_name__}_{sp1}-{sp2}_Mean"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
                # Std
                col_data = self.dataframe_slices[col_str].std(axis=1).values
                col_name = f"{self.__long_name__}_{sp1}-{sp2}_Std"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

        # Time stamp
        tend = self.timer.current()
        self.time_stamp(f"{self.__long_name__} Calculation", self.timer.time_division(tend - t0))

        # Save
        self.save_hdf()
        # Plot
        if plot:
            _, _ = self.plot(observable, display_plot=display_plot)

    def plot(self, observable, display_plot: bool = False):
        """
        Make a dual plot comparing the ACF and the Transport Coefficient by using the :meth:`plot_tc` method.

        Parameters
        ----------
        observable: :class:`sarkas.tools.observables.HeatFlux`
            Observable object containing the ACF whose time integral leads to the self diffusion coefficient.

        display_plot : bool, optional
            Flag for displaying the plot if using the IPython. Default = False.

        Return
        ------
        fig : :class:`matplotlib.pyplot.Figure`
            Matplotlib figure handle

        (ax1, ax2, ax3, ax4) : tuple, :class:`matplotlib.axes.Axes`
            Tuple containing the axes handles for `fig`. 'ax1` and `ax2` are the handles for the left and right plots respectively.
            `ax3` and `ax4` are the handles for the "Index" axes, created from `ax1.twiny()` and `ax2.twiny()` respectively.

        """
        sp_vacf_str = f"{observable.__long_name__} ACF"

        if self.num_species > 1:
            species_list = [*observable.species_names, "all"]
        else:
            species_list = observable.species_names

        for isp, sp1 in enumerate(species_list):
            for _, sp2 in enumerate(species_list[isp:], isp):
                acf_avg = observable.dataframe_acf[(sp_vacf_str, f"{sp1}-{sp2}", "Total", "Mean")].to_numpy()
                acf_std = observable.dataframe_acf[(sp_vacf_str, f"{sp1}-{sp2}", "Total", "Std")].to_numpy()

                col_name = (f"{self.__long_name__}", f"{sp1}-{sp2}", "Mean")
                tc_avg = self.dataframe[col_name].to_numpy()
                col_name = (f"{self.__long_name__}", f"{sp1}-{sp2}", "Std")
                tc_std = self.dataframe[col_name].to_numpy()

                fig, (ax1, ax2, ax3, ax4) = self.plot_tc(
                    time=self.time_array,
                    acf_data=column_stack((acf_avg, acf_std)),
                    tc_data=column_stack((tc_avg, tc_std)),
                    acf_name=sp_vacf_str,
                    tc_name=f"{sp1}-{sp2} {self.__long_name__}",
                    figname=f"{self.__name__}_{sp1}-{sp2}_Plot.png",
                    show=display_plot,
                )

        return fig, (ax1, ax2, ax3, ax4)
