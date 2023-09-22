"""
Module for calculating physical quantities from Sarkas checkpoints.
"""
import inspect
from copy import deepcopy
from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import datetime
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scp_stats
import sys
from matplotlib.gridspec import GridSpec
from numba import njit
from numpy import append as np_append
from numpy import (
    argsort,
    array,
    complex128,
    concatenate,
    exp,
    format_float_scientific,
    histogram,
    interp,
    isfinite,
    load,
    log,
    ndarray,
    ones,
    ones_like,
    pi,
    real,
    repeat,
    roll,
    savez,
    sqrt,
    trapz,
    unique,
    unwrap,
    where,
    zeros,
)
from numpy.polynomial import hermite_e
from numpy.random import default_rng
from os import listdir, mkdir
from os import remove as os_remove
from os.path import exists as os_path_exists
from os.path import join as os_path_join
from pandas import (
    concat,
    DataFrame,
    HDFStore,
    MultiIndex,
    read_csv,
    read_hdf,
    Series,
    to_numeric,
)
from pickle import dump
from pickle import load as pickle_load
from scipy.fft import fft, fftfreq, fftshift
from scipy.integrate import quad as scp_quad
from scipy.linalg import norm
from scipy.optimize import curve_fit
from scipy.special import erfc, factorial
from seaborn import histplot as sns_histplot

from ..utilities.io import print_to_logger
from ..utilities.maths import correlationfunction
from ..utilities.misc import add_col_to_df
from ..utilities.timing import datetime_stamp, SarkasTimer, time_stamp
from .fit_functions import exponential, gaussian

UNITS = [
    # MKS Units
    {
        "Energy": "J",
        "Heat Flux": "J/s",
        "Time": "s",
        "Length": "m",
        "Charge": "C",
        "Temperature": "K",
        "ElectronVolt": "eV",
        "Mass": "kg",
        "Magnetic Field": "T",
        "Current": "A",
        "Power": "erg/s",
        "Pressure": "Pa",
        "Electrical Conductivity": "S/m",
        "Diffusion": r"m$^2$/s",
        "Bulk Viscosity": r"kg/m-s",
        "Shear Viscosity": r"kg/m-s",
        "Thermal Conductivity": r"J/m-s-K",
        "none": "",
    },
    # CGS Units
    {
        "Energy": "erg",
        "Heat Flux": "erg/s",
        "Time": "s",
        "Length": "m",
        "Charge": "esu",
        "Temperature": "K",
        "ElectronVolt": "eV",
        "Mass": "g",
        "Magnetic Field": "G",
        "Current": "esu/s",
        "Power": "erg/s",
        "Pressure": "Ba",
        "Electrical Conductivity": "mho/m",
        "Diffusion": r"m$^2$/s",
        "Bulk Viscosity": r"g/cm-s",
        "Shear Viscosity": r"g/cm-s",
        "Thermal Conductivity": r"erg/cm-s-K",
        "none": "",
    },
]

PREFIXES = {
    "y": 1.0e-24,  # yocto
    "z": 1.0e-21,  # zepto
    "a": 1.0e-18,  # atto
    "f": 1.0e-15,  # femto
    "p": 1.0e-12,  # pico
    "n": 1.0e-9,  # nano
    r"$\mu$": 1.0e-6,  # micro
    "m": 1.0e-3,  # milli
    "c": 1.0e-2,  # centi
    "": 1.0,
    "k": 1e3,  # kilo
    "M": 1e6,  # mega
    "G": 1e9,  # giga
    "T": 1e12,  # tera
    "P": 1e15,  # peta
    "E": 1e18,  # exa
    "Z": 1e21,  # zetta
    "Y": 1e24,  # yotta
}


def compute_doc(func):
    func.__doc__ = """
    Routine for computing the observable. See class doc for exact quantities.\n
    The data of each slice is saved in hierarchical dataframes,
    :py:attr:`~.dataframe_slices`. \n
    The sliced averaged data is saved in other hierarchical dataframes,
    :py:attr:`~.dataframe`.

    Parameters
    ----------
    calculate_acf : bool
        Calculate the ACF of the observable. Default =`True` except for :class:`sarkas.tools.observables.Thermodynamics`.

    """
    return func


def compute_acf_doc(func):
    func.__doc__ = """
    Routine for computing the observable's autocorrelation function. See class doc for exact quantities.\n
    The data of each slice is saved in hierarchical dataframes,
    :py:attr:`~.dataframe_acf_slices`. \n

    The sliced averaged data is saved in other hierarchical dataframes,
    :py:attr:`~.dataframe_acf`.
    """
    return func


def calc_slices_doc(func):
    func.__doc__ = """
    Calculate the observable for each slice. See class doc for exact quantities.\n
    The data of each slice is saved in hierarchical dataframes,
    :py:attr:`~.dataframe_slices`.\n
    """
    return func


def calc_acf_slices_doc(func):
    func.__doc__ = """
    Calculate the observable acf for each slice. See class doc for exact quantities.\n
    The data of each slice is saved in hierarchical dataframes,
    :py:attr:`~.dataframe_acf_slices`.\n
    """
    return func


def avg_slices_doc(func):
    func.__doc__ = """
    Calculate the average and standard deviation of the observable from the slices dataframe.
    See class doc for exact quantities. \n
    The data of each slice is saved in hierarchical dataframes,
    :py:attr:`~.dataframe_slices` (:py:attr:`~.dataframe_acf_slices`). \n

    The sliced averaged data is saved in other hierarchical dataframes,
    :py:attr:`~.dataframe` (:py:attr:`~.dataframe_acf_slices`).
    """
    return func


def avg_acf_slices_doc(func):
    func.__doc__ = """
    Calculate the average and standard deviation of the observable autocorrelation function from the slices dataframe.
    See class doc for exact quantities. \n
    The data of each slice is saved in hierarchical dataframes,
    :py:attr:`~.dataframe_acf_slices`. \n

    The sliced averaged data is saved in other hierarchical dataframes,
    :py:attr:`~.dataframe_acf`.
    """
    return func


def setup_doc(func):
    func.__doc__ = """
    Assign attributes from simulation's parameters.

    Parameters
    ----------
    params : :class:`sarkas.core.Parameters`
        Simulation's parameters.

    phase : str, optional
        Phase to compute. Default = 'production'.

    no_slices : int, optional
        Number of independent runs inside a long simulation. Default = 1.

    **kwargs :
        These will overwrite any :class:`sarkas.core.Parameters`
        or default :class:`sarkas.tools.observables.Observable`
        attributes and/or add new ones.

   """
    return func


def arg_update_doc(func):
    func.__doc__ = """Update observable specific attributes and call :meth:`~.update_finish` to save info."""
    return func


# TODO: Divide calculation in case of multirun
# TODO: Check Velocity Distribution class.


class Observable:
    """
    Parent class of all the observables.

    Attributes
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing the observable's data averaged over the number of slices.

    dataframe_acf : pandas.DataFrame
        Dataframe containing the observable's autocorrelation function data averaged over the number of slices.

    dataframe_acf_slices : pandas.DataFrame
        Dataframe containing the observable's autocorrelation data for each slice.

    dataframe_slices : pandas.DataFrame
        Dataframe containing the observable's data for each slice.

    max_k_harmonics : list
        Maximum number of :math:`\\mathbf{k}` harmonics to calculate along each dimension.

    phase : str
        Phase to analyze.

    prod_no_dumps : int
        Number of production phase checkpoints. Calculated from the number of files in the Production directory.

    eq_no_dumps : int
        Number of equilibration phase checkpoints. Calculated from the number of files in the Equilibration directory.

    no_dumps : int
        Number of simulation's checkpoints. Calculated from the number of files in the phase folder.

    dump_dir : str
        Path to correct dump directory.

    dump_step : int
        Correct step interval.
        It is either :py:attr:`sarkas.core.Parameters.prod_dump_step` or :py:attr:`sarkas.core.Parameters.eq_dump_step`.

    no_obs : int
        Number of independent binary observable quantities.
        It is calculated as :math:`N_s (N_s + 1) / 2` where :math:`N_s` is the number of species.

    k_file : str
        Path to the npz file storing the :math:`k` vector values.

    nkt_hdf_file : str
        Path to the npy file containing the Fourier transform of density fluctuations. :math:`n(\\mathbf k, t)`.

    vkt_file : str
        Path to the npz file containing the Fourier transform of velocity fluctuations. :math:`\\mathbf v(\\mathbf k, t)`.

    k_space_dir : str
        Directory where :math:`\\mathbf {k}` data is stored.

    saving_dir : str
        Path to the directory where computed data is stored.

    slice_steps : int
        Number of steps per slice.

    dimensional_average: bool
            Flag for averaging over all dimensions. Default = False.

    runs: int
            Number of independent MD runs. Default = 1.

    multi_run_average: bool
        Flag for averaging over multiple runs. Default = False.
        If True, `runs` needs be specified. It will collect data from all runs and stored them in a large ndarray to
        be averaged over.

    """

    def __init__(self):
        self.postprocessing_dir = None
        self.mag_no_dumps = None
        self.eq_no_dumps = None
        self.prod_no_dumps = None
        self.no_obs = None
        self.filename_hdf_acf = None
        self.species_index_start = None
        self.filename_hdf_acf_slices = None
        self.filename_hdf_slices = None
        self.filename_hdf = None
        self.__long_name__ = None
        self.__name__ = None
        self.saving_dir = None
        self.phase = "production"
        self.multi_run_average = False
        self.dimensional_average = False
        self.runs = 1
        self.no_slices = 1
        self.slice_steps = None
        self.screen_output = True
        self.timer = SarkasTimer()
        # k observable attributes
        self.k_observable = False
        self.max_aa_harmonics = None
        self.angle_averaging = "principal_axis"
        self.max_k_harmonics = None
        self.max_aa_ka_value = None
        self.kw_observable = False
        self.dim_labels = ["X", "Y", "Z"]
        self.acf_observable = False
        self.dataframe = None
        self.dataframe_slices = None
        self.dataframe_acf = None
        self.dataframe_acf_slices = None

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Observable( " + self.__class__.__name__ + "\n"
        exclude_list = ["dataframe", "dataframe_slices", "dataframe_acf", "dataframe_acf_slices"]
        for key, value in sortedDict.items():
            if not key in exclude_list:
                disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def __getstate__(self):
        """Copy the object's state from self.__dict__ which contains all our instance attributes.
        Always use the dict.copy() method to avoid modifying the original state.
        Reference: https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        """

        state = self.__dict__.copy()
        # Remove the data that is stored already
        del state["dataframe"]
        del state["dataframe_slices"]
        del state["dataframe_acf"]
        del state["dataframe_acf_slices"]

        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        # Restore the previously deleted dataframes.
        self.parse()

    def calc_k_data(self):
        """Calculate and save Fourier space data."""

        # Do some checks
        if not isinstance(self.angle_averaging, str):
            raise TypeError("angle_averaging not a string. " "Choose from ['full', 'custom', 'principal_axis']")
        elif self.angle_averaging not in ["full", "custom", "principal_axis"]:
            raise ValueError(
                "Option not available. " "Choose from ['full', 'custom', 'principal_axis']" "Note case sensitivity."
            )
        assert self.max_k_harmonics.all(), "max_k_harmonics not defined."

        # Calculate the k arrays
        self.k_list, self.k_counts, k_unique, self.k_harmonics = kspace_setup(
            self.box_lengths, self.angle_averaging, self.max_k_harmonics, self.max_aa_harmonics
        )
        # Save the ka values
        self.ka_values = 2.0 * pi * k_unique * self.a_ws
        self.k_values = 2.0 * pi * k_unique
        self.no_ka_values = len(self.ka_values)

        # Check if the writing folder exist
        if not (os_path_exists(self.k_space_dir)):
            mkdir(self.k_space_dir)

        # Write the npz file
        savez(
            self.k_file,
            k_list=self.k_list,
            k_harmonics=self.k_harmonics,
            k_counts=self.k_counts,
            k_values=self.k_values,
            ka_values=self.ka_values,
            angle_averaging=self.angle_averaging,
            max_k_harmonics=self.max_k_harmonics,
            max_aa_harmonics=self.max_aa_harmonics,
        )

    def calc_nkt_slices_data(self):
        """Calculate n(k,t) for each slice."""
        start_slice = 0
        end_slice = self.slice_steps * self.dump_step
        self.nkt_dataframe_slices = DataFrame()

        for isl in tqdm(
            range(self.no_slices),
            desc="Calculating n(k,t) for slice ",
            position=0,
            disable=not self.verbose,
            leave=True,
        ):
            nkt = calc_nkt(
                self.dump_dir,
                (start_slice, end_slice, self.slice_steps),
                self.dump_step,
                self.species_num,
                self.k_list,
                self.verbose,
            )
            start_slice += self.slice_steps * self.dump_step
            end_slice += self.slice_steps * self.dump_step
            # n(k,t).shape = [no_species, time, k vectors]

            slc_column = "slice {}".format(isl + 1)
            for isp, sp_name in enumerate(self.species_names):
                df_columns = [
                    slc_column + "_{}_k = [{}, {}, {}]".format(sp_name, *self.k_harmonics[ik, :-2].astype(int))
                    for ik in range(len(self.k_harmonics))
                ]
                # df_columns = [time_column, *k_columns]
                self.nkt_dataframe_slices = concat(
                    [self.nkt_dataframe_slices, DataFrame(nkt[isp, :, :], columns=df_columns)], axis=1
                )

        # Example nkt_dataframe
        # slices slice 1
        # species H
        # harmonics k = [0, 0, 1] | k = [0, 1, 0] | ...
        tuples = [tuple(c.split("_")) for c in self.nkt_dataframe_slices.columns]
        self.nkt_dataframe_slices.columns = MultiIndex.from_tuples(tuples, names=["slices", "species", "harmonics"])

    def calc_vkt_slices_data(self):
        """Calculate v(k,t) for each slice."""

        start_slice = 0
        end_slice = self.slice_steps * self.dump_step

        self.vkt_dataframe_slices = DataFrame()

        for isl in tqdm(
            range(self.no_slices),
            desc="Calculating v(k,t) for slice ",
            position=0,
            disable=not self.verbose,
            leave=True,
        ):

            vkt, vkt_i, vkt_j, vkt_k = calc_vkt(
                self.dump_dir,
                (start_slice, end_slice, self.slice_steps),
                self.dump_step,
                self.species_num,
                self.k_list,
                self.verbose,
            )
            start_slice += self.slice_steps * self.dump_step
            end_slice += self.slice_steps * self.dump_step

            slc_column = "slice {}".format(isl + 1)
            for isp, sp_name in enumerate(self.species_names):
                df_columns = [
                    slc_column
                    + "_Longitudinal_{}_k = [{}, {}, {}]".format(sp_name, *self.k_harmonics[ik, :-2].astype(int))
                    for ik in range(len(self.k_harmonics))
                ]
                # df_columns = [time_column, *k_columns]
                self.vkt_dataframe_slices = concat(
                    [self.vkt_dataframe_slices, DataFrame(vkt[isp, :, :], columns=df_columns)], axis=1
                )
                df_columns = [
                    slc_column
                    + "_Transverse i_{}_k = [{}, {}, {}]".format(sp_name, *self.k_harmonics[ik, :-2].astype(int))
                    for ik in range(len(self.k_harmonics))
                ]
                self.vkt_dataframe_slices = concat(
                    [self.vkt_dataframe_slices, DataFrame(vkt_i[isp, :, :], columns=df_columns)], axis=1
                )
                df_columns = [
                    slc_column
                    + "_Transverse j_{}_k = [{}, {}, {}]".format(sp_name, *self.k_harmonics[ik, :-2].astype(int))
                    for ik in range(len(self.k_harmonics))
                ]
                self.vkt_dataframe_slices = concat(
                    [self.vkt_dataframe_slices, DataFrame(vkt_j[isp, :, :], columns=df_columns)], axis=1
                )

                df_columns = [
                    slc_column
                    + "_Transverse k_{}_k = [{}, {}, {}]".format(sp_name, *self.k_harmonics[ik, :-2].astype(int))
                    for ik in range(len(self.k_harmonics))
                ]
                self.vkt_dataframe_slices = concat(
                    [self.vkt_dataframe_slices, DataFrame(vkt_k[isp, :, :], columns=df_columns)], axis=1
                )

        # Example vkt_dataframe
        # slices slice 1
        # species H
        # direction Longitudinal/Transverse
        # harmonics k = [0, 0, 1] | k = [0, 1, 0] | ...

        # Full string: slice 1_Longitudinal_He_k = [1, 0, 0]
        tuples = [tuple(c.split("_")) for c in self.vkt_dataframe_slices.columns]
        self.vkt_dataframe_slices.columns = MultiIndex.from_tuples(
            tuples, names=["slices", "species", "direction", "harmonics"]
        )

    @staticmethod
    def integrate_normalized_acf_squared(time, data):
        """
        Calculate the normalized correlation time as given by
        """
        data_0 = data[0]
        tau_2 = 2.0 * trapz((data / data_0) ** 2, x=time)
        return tau_2

    def calculate_corr_times(self, slices=False):

        if slices:
            df_acf = self.dataframe_acf_slices
        else:
            df_acf = self.dataframe_acf

        cols = df_acf.columns
        time = df_acf[cols[0]].values
        index_list = [
            "exponential [s]",
            "gaussian [s]",
            "integral [s]",
            "exponential [timestep]",
            "gaussian [timestep]",
            "integral [timestep]",
            "exponential [dump step]",
            "gaussian [dump step]",
            "integral [dump step]",
            "exponential [plasma cycle]",
            "gaussian [plasma cycle]",
            "integral [plasma cycle]",
        ]
        # This creates a df with a bunch of Nan objects
        df = DataFrame(index=index_list, columns=MultiIndex.from_tuples(df_acf.columns[1:]))

        t_wp = (2.0 * pi) / self.total_plasma_frequency

        for col in cols[1:]:
            # Normalize the ACF
            acf_0 = (df_acf[col].values)[0]
            acf = df_acf[col].values / acf_0
            # Find a short time to avoid the noise in the tail.
            # Take all the values of the ACF up to 10% from zero.
            indices = where(acf > 0.1)[0]
            acf_fit = acf[indices]
            time_fit = time[indices]
            # Fit an exponential. Use the plasma cycle as the initial guess.
            popt, _ = curve_fit(exponential, time_fit, acf_fit, p0=t_wp)
            # Calculate correlation time. Recall that the corr time is 2 int_0^inf [C(t)/C(0)]^2

            df.loc["exponential [s]"][col] = popt[0]
            df.loc["exponential [timestep]"][col] = popt[0] / self.dt
            df.loc["exponential [dump step]"][col] = popt[0] / (self.dump_step * self.dt)
            df.loc["exponential [plasma cycle]"][col] = popt[0] / t_wp

            # Fit a gaussian
            popt, _ = curve_fit(gaussian, time_fit, acf_fit, p0=t_wp)
            # Calculate correlation time int (exp(- x^2/t) )^2 from 0 to inf = 0.5 * sqrt(pi/2 * t)
            tau = sqrt(0.5 * pi) * popt[0]  # don't forget the 2 in front of the integral
            df.loc["gaussian [s]"][col] = tau
            df.loc["gaussian [timestep]"][col] = tau / self.dt
            df.loc["gaussian [dump step]"][col] = tau / (self.dump_step * self.dt)
            df.loc["gaussian [plasma cycle]"][col] = tau / t_wp

            # No fit.
            tau = self.integrate_normalized_acf_squared(time, acf)
            df.loc["integral [s]"][col] = tau
            df.loc["integral [timestep]"][col] = tau / self.dt
            df.loc["integral [dump step]"][col] = tau / (self.dump_step * self.dt)
            df.loc["integral [plasma cycle]"][col] = tau / t_wp

        # Need to conver to numbers because they are all objects.
        for col in df.columns:
            df[col] = to_numeric(df[col])

        if slices:
            self.correlation_times_slices = df
        else:
            self.correlation_times = df

    def compute_kt_data(self, nkt_flag: bool = False, vkt_flag: bool = False):
        """Calculate Time dependent Fourier space quantities.

        Parameters
        ----------
        nkt_flag : bool
            Flag for calculating microscopic density Fourier components :math:`n(\\mathbf k, t)`. \n
            Default = False.

        vkt_flag : bool
            Flag for calculating microscopic velocity Fourier components, :math:`v(\\mathbf k, t)`. \n
            Default = False.

        """
        if nkt_flag:
            tinit = self.timer.current()
            self.calc_nkt_slices_data()
            self.save_kt_hdf(nkt_flag=True)
            tend = self.timer.current()
            time_stamp(self.log_file, "n(k,t) Calculation", self.timer.time_division(tend - tinit), self.verbose)

        if vkt_flag:
            tinit = self.timer.current()
            self.calc_vkt_slices_data()
            self.save_kt_hdf(vkt_flag=True)
            tend = self.timer.current()
            time_stamp(self.log_file, "v(k,t) Calculation", self.timer.time_division(tend - tinit), self.verbose)

    def copy_params(self, params):

        for i, val in params.__dict__.items():
            if not inspect.ismethod(val):
                if isinstance(val, dict):
                    self.__dict__[i] = deepcopy(val)
                elif isinstance(val, ndarray):
                    self.__dict__[i] = val.copy()
                else:
                    self.__dict__[i] = val

    def create_dirs_filenames(self):
        """Create the directories and filenames where to save dataframes. It also creates a log file that can be accessed at
        :py:attr:`~.log_file`. Finally, it initializes the dataframes used to store data.
        """
        # Saving Directory
        self.setup_multirun_dirs()

        # If multi_run_average self.postprocessing_dir is Simulations/Postprocessing else Job_dir/Postprocessing
        saving_dir = os_path_join(self.directory_tree["postprocessing"]["path"], self.__long_name__.replace(" ", ""))
        if not os_path_exists(saving_dir):
            mkdir(saving_dir)

        self.saving_dir = os_path_join(saving_dir, self.phase.capitalize())
        if not os_path_exists(self.saving_dir):
            mkdir(self.saving_dir)

        # Create the log file path
        fname = self.__long_name__.replace(" ", "") + "_log_file.out"
        self.log_file = os_path_join(self.saving_dir, fname)

        # Filenames and strings
        fname = self.__long_name__.replace(" ", "") + "_" + self.job_id + ".h5"
        self.filename_hdf = os_path_join(self.saving_dir, fname)

        fname = self.__long_name__.replace(" ", "") + "_slices_" + self.job_id + ".h5"
        self.filename_hdf_slices = os_path_join(self.saving_dir, fname)

        if self.__name__ == "ccf":
            fname = "Longitudinal_" + self.__long_name__.replace(" ", "") + "_" + self.job_id + ".h5"
            self.filename_hdf_longitudinal = os_path_join(self.saving_dir, fname)

            fname = "Longitudinal_" + self.__long_name__.replace(" ", "") + "_slices_" + self.job_id + ".h5"
            self.filename_hdf_longitudinal_slices = os_path_join(self.saving_dir, fname)

            fname = "Transverse_" + self.__long_name__.replace(" ", "") + "_" + self.job_id + ".h5"
            self.filename_hdf_transverse = os_path_join(self.saving_dir, fname)

            fname = "Transverse_" + self.__long_name__.replace(" ", "") + "_slices_" + self.job_id + ".h5"
            self.filename_hdf_transverse_slices = os_path_join(self.saving_dir, fname)

            self.filename_hdf = {
                "Longitudinal": self.filename_hdf_longitudinal,
                "Transverse": self.filename_hdf_transverse,
            }

            self.filename_hdf_slices = {
                "Longitudinal": self.filename_hdf_longitudinal_slices,
                "Transverse": self.filename_hdf_transverse_slices,
            }

        if self.acf_observable:
            fname = self.__long_name__.replace(" ", "") + "ACF_" + self.job_id + ".h5"
            self.filename_hdf_acf = os_path_join(self.saving_dir, fname)

            fname = self.__long_name__.replace(" ", "") + "ACF_slices_" + self.job_id + ".h5"
            self.filename_hdf_acf_slices = os_path_join(self.saving_dir, fname)

        if self.k_observable:
            # Create paths for files
            self.k_space_dir = os_path_join(self.postprocessing_dir, "k_space_data")
            self.k_file = os_path_join(self.k_space_dir, "k_arrays.npz")
            self.nkt_hdf_file = os_path_join(self.k_space_dir, "nkt.h5")
            self.vkt_hdf_file = os_path_join(self.k_space_dir, "vkt.h5")

    def from_dict(self, input_dict: dict):
        """
        Update attributes from input dictionary.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """
        self.__dict__.update(input_dict)

    def grab_sim_data(self, pva: str = "vel"):
        """Read in particles data into one large array.

        Parameters
        ----------
        pva : str
            Key of the data to be collected. Options ["pos", "vel", "acc"], Default = "vel"

        Returns
        -------
        time : numpy.ndarray
            One dimensional array with time data.

        data_all : numpy.ndarray
                Array with shape (:attr:`sarkas.tools.observables.Observable.no_dumps`,
        :attr:`sarkas.tools.observables.Observable.self.dim`, :attr:`sarkas.tools.observables.Observable.runs` *
        :attr:`sarkas.tools.observables.Observable.inv_dim` * :attr:`sarkas.tools.observables.Observable.total_num_ptcls`).
        `.dim` = 1 if :attr:`sarkas.tools.observables.Observable.dimensional_average = True` otherwise equals the number
        of dimensions, (e.g. 3D : 3) `.runs` is the number of runs to be averaged over. Default = 1. `.inv_dim` is
        the else option of `dim`. If `dim = 1` then `.inv_dim = .dimensions` and viceversa.


        """

        # Velocity array for storing simulation data
        data_all = zeros((self.no_dumps, self.dim, self.runs * self.inv_dim * self.total_num_ptcls))
        time = zeros(self.no_dumps)

        print("\nCollecting data from snapshots ...")
        if self.dimensional_average:
            # Loop over the runs
            for r, dump_dir_r in enumerate(tqdm(self.dump_dirs_list, disable=(not self.verbose), desc="Runs Loop")):
                # Loop over the timesteps
                for it in tqdm(range(self.no_dumps), disable=(not self.verbose), desc="Timestep Loop"):
                    # Read data from file
                    dump = int(it * self.dump_step)
                    datap = load_from_restart(dump_dir_r, dump)
                    # Loop over the particles' species
                    for sp_indx, (sp_name, sp_num) in enumerate(zip(self.species_names, self.species_num)):
                        # Calculate the correct start and end index for storage
                        start_indx = self.species_index_start[sp_indx] + self.inv_dim * sp_num * r
                        end_indx = self.species_index_start[sp_indx] + self.inv_dim * sp_num * (r + 1)
                        # Use a mask to grab only the selected species and flatten along the first axis
                        # data = ( v1_x, v1_y, v1_z,
                        #          v2_x, v2_y, v2_z,
                        #          v3_x, v3_y, v3_z,
                        #          ...)
                        # The flatten array would like this
                        # flattened = ( v1_x, v2_x, v3_x, ..., v1_y, v2_y, v3_y, ..., v1_z, v2_z, v3_z, ...)
                        mask = datap["names"] == sp_name
                        data_all[it, 0, start_indx:end_indx] = datap[pva][mask].flatten("F")

                    time[it] = datap["time"]

        else:  # Dimensional Average = False
            # Loop over the runs
            for r, dump_dir_r in enumerate(tqdm(self.dump_dirs_list, disable=(not self.verbose), desc="Runs Loop")):
                # Loop over the timesteps
                for it in tqdm(range(self.no_dumps), disable=(not self.verbose), desc="Timestep Loop"):
                    # Read data from file
                    dump = int(it * self.dump_step)
                    datap = load_from_restart(dump_dir_r, dump)
                    # Loop over the particles' species
                    for sp_indx, (sp_name, sp_num) in enumerate(zip(self.species_names, self.species_num)):
                        # Calculate the correct start and end index for storage
                        start_indx = self.species_index_start[sp_indx] + self.inv_dim * sp_num * r
                        end_indx = self.species_index_start[sp_indx] + self.inv_dim * sp_num * (r + 1)
                        # Use a mask to grab only the selected species and transpose the array to put dimensions first
                        for d in range(self.dimensions):
                            mask = datap["names"] == sp_name
                            data_all[it, d, start_indx:end_indx] = datap[pva][mask][:, d]

                    time[it] = datap["time"]

        return time, data_all

    def parse(self, acf_data: bool = False):
        """
        Grab the pandas dataframe from the saved csv file. If file does not exist call :meth:`compute()`.
        """

        if self.__name__ == "rdf":
            acf_data = False

        if self.k_observable:
            try:
                self.dataframe = read_hdf(self.filename_hdf, mode="r", index_col=False)

                k_data = load(self.k_file)
                self.k_list = k_data["k_list"]
                self.k_counts = k_data["k_counts"]
                self.ka_values = k_data["ka_values"]

            except FileNotFoundError:
                print("\nFile {} not found!".format(self.filename_hdf))
                print("\nComputing Observable now ...")
                self.compute()
        else:
            try:
                if hasattr(self, "filename_csv"):
                    self.dataframe = read_csv(self.filename_csv, index_col=False)
                else:
                    self.dataframe = read_hdf(self.filename_hdf, mode="r", index_col=False)

            except FileNotFoundError:
                if hasattr(self, "filename_csv"):
                    data_file = self.filename_csv
                else:
                    data_file = self.filename_hdf
                print("\nData file not found! \n {}".format(data_file))
                print("\nComputing Observable now ...")
                self.compute()

            if hasattr(self, "dataframe_slices"):
                self.dataframe_slices = read_hdf(self.filename_hdf_slices, mode="r", index_col=False)

            if acf_data:
                self.parse_acf()

    def parse_acf(self):

        try:
            self.dataframe_acf = read_hdf(self.filename_hdf_acf, mode="r", index_col=False)
            self.dataframe_acf_slices = read_hdf(self.filename_hdf_acf_slices, mode="r", index_col=False)
        except FileNotFoundError:
            print(f"\nFiles {self.filename_hdf_acf} not found!")
            print("\nComputing Observable now ...")
            self.compute_acf()

    def parse_k_data(self):
        """Read in the precomputed Fourier space data. Recalculate if not correct."""

        try:
            k_data = load(self.k_file)
            # Check for the correct number of k values
            if self.angle_averaging == k_data["angle_averaging"]:
                # Check for the correct max harmonics
                comp = self.max_k_harmonics == k_data["max_k_harmonics"]
                if comp.all():
                    self.k_list = k_data["k_list"]
                    self.k_harmonics = k_data["k_harmonics"]
                    self.k_counts = k_data["k_counts"]
                    self.k_values = k_data["k_values"]
                    self.ka_values = k_data["ka_values"]
                    self.no_ka_values = len(self.ka_values)
                else:
                    self.calc_k_data()
            else:
                self.calc_k_data()

        except FileNotFoundError:
            self.calc_k_data()

    def parse_kt_data(self, nkt_flag: bool = False, vkt_flag: bool = False):
        """
        Read in the precomputed time dependent Fourier space data. Recalculate if not.

        Parameters
        ----------
        nkt_flag : bool
            Flag for reading microscopic density Fourier components :math:`n(\\mathbf k, t)`. \n
            Default = False.

        vkt_flag : bool
            Flag for reading microscopic velocity Fourier components, :math:`v(\\mathbf k, t)`. \n
            Default = False.

        """
        if nkt_flag:
            try:
                # Check that what was already calculated is correct
                with HDFStore(self.nkt_hdf_file, mode="r") as nkt_hfile:
                    metadata = nkt_hfile.get_storer("nkt").attrs.metadata

                if metadata["no_slices"] == self.no_slices:
                    # Check for the correct number of k values
                    if metadata["angle_averaging"] == self.angle_averaging:
                        # Check for the correct max harmonics
                        comp = self.max_k_harmonics == metadata["max_k_harmonics"]
                        if not comp.all():
                            self.compute_kt_data(nkt_flag=True)
                    else:
                        self.compute_kt_data(nkt_flag=True)
                else:
                    self.compute_kt_data(nkt_flag=True)

                # elif metadata['max_k_harmonics']
                #
                # if self.angle_averaging == nkt_data["angle_averaging"]:
                #
                #     comp = self.max_k_harmonics == nkt_data["max_harmonics"]
                #     if not comp.all():
                #         self.compute_kt_data(nkt_flag=True)
                # else:
                #     self.compute_kt_data(nkt_flag=True)

            except OSError:
                self.compute_kt_data(nkt_flag=True)

        if vkt_flag:

            try:
                # Check that what was already calculated is correct
                with HDFStore(self.vkt_hdf_file, mode="r") as vkt_hfile:
                    metadata = vkt_hfile.get_storer("vkt").attrs.metadata

                if metadata["no_slices"] == self.no_slices:
                    # Check for the correct number of k values
                    if metadata["angle_averaging"] == self.angle_averaging:
                        # Check for the correct max harmonics
                        comp = self.max_k_harmonics == metadata["max_k_harmonics"]
                        if not comp.all():
                            self.compute_kt_data(vkt_flag=True)
                    else:
                        self.compute_kt_data(vkt_flag=True)
                else:
                    self.compute_kt_data(vkt_flag=True)

            except OSError:
                self.compute_kt_data(vkt_flag=True)

    def plot(self, scaling: tuple = None, acf: bool = False, figname: str = None, show: bool = False, **kwargs):
        """
        Plot the observable by calling the pandas.DataFrame.plot() function and save the figure.

        Parameters
        ----------
        scaling : float, tuple
            Factor by which to rescale the x and y axis.

        acf : bool
            Flag for renormalizing the autocorrelation functions. Default= False

        figname : str
            Name with which to save the file. It automatically saves it in the correct directory.

        show : bool
            Flag for prompting the plot to screen. Default=False

        **kwargs :
            Options to pass to matplotlib plotting method.

        Returns
        -------
        axes_handle : matplotlib.axes.Axes
            Axes. See `pandas` documentation for more info

        """

        if acf:
            plot_dataframe = self.dataframe_acf.copy()
            # Autocorrelation function renormalization
            for i, col in enumerate(plot_dataframe.columns[1:], 1):
                plot_dataframe[col] /= plot_dataframe[col].iloc[0]
            kwargs["logx"] = True
            # kwargs['xlabel'] = 'Time difference'

        else:
            plot_dataframe = self.dataframe.copy()
        # This is needed because I don't know a priori what the first column name is

        first_col_name = plot_dataframe.columns[0]

        if scaling:
            if isinstance(scaling, tuple):
                plot_dataframe[first_col_name] /= scaling[0]
                plot_dataframe[kwargs["y"]] /= scaling[1]
            else:
                plot_dataframe[first_col_name] /= scaling

        axes_handle = plot_dataframe.plot(x=plot_dataframe.columns[0], **kwargs)

        fig = axes_handle.figure
        fig.tight_layout()

        # Saving
        if figname:
            fig.savefig(os_path_join(self.saving_dir, figname + "_" + self.job_id + ".png"))
        else:
            fig.savefig(os_path_join(self.saving_dir, "Plot_" + self.__name__ + "_" + self.job_id + ".png"))

        if show:
            fig.show()

        return axes_handle

    def pretty_print_msg(self):
        """Create the message with the basic information of every observable

        Returns
        -------
        msg : str
            Message to print.

        """
        name = " " + self.__long_name__ + " "
        if self.__name__ == "ccf":
            msg = (
                f"\n\n{name:=^70}\n"
                f"Data saved in: \n {self.filename_hdf}\n"
                f"Data accessible via: self.dataframe_longitudinal_slices, self.dataframe_longitudinal\n"
                f"\t\tself.dataframe_transverse_slices, self.dataframe_transverse"
            )
        else:
            msg = (
                f"\n\n{name:=^70}\n"
                f"Data saved in: \n {self.filename_hdf}\n"
                f"Data accessible via: self.dataframe_slices, self.dataframe\n"
            )

        if self.__name__ == "rdf":
            msg += (
                f"No. bins = {self.no_bins}\n"
                f"dr = {self.dr_rdf / self.a_ws:.4f} a_ws = {self.dr_rdf:.4e} {self.units_dict['length']}\n"
                f"Maximum Distance (i.e. potential.rc)= {self.rc / self.a_ws:.4f} a_ws = {self.rc:.4e} {self.units_dict['length']}"
            )

        dtau = self.dt * self.dump_step
        tau = dtau * self.slice_steps
        t_wp = 2.0 * pi / self.total_plasma_frequency  # Plasma period
        tau_wp = int(tau / t_wp)
        msg += (
            f"\nTime Series Data:\n"
            f"No. of slices = {self.no_slices}\n"
            f"No. dumps per slice = {int(self.slice_steps)}\n"
            f"Total time: T = {tau:.4e} {self.units_dict['time']} ~ {tau_wp} plasma periods\n"
            f"Time interval step: dt = {dtau:.4e} ~ {dtau / t_wp:.4e} plasma period"
        )
        if self.acf_observable:

            dtau = self.dt * self.dump_step
            tau = dtau * self.acf_slice_steps
            tau_wp = int(tau / t_wp)

            msg += (
                f"\n\nACF Data:\n"
                f"If you choose to set equal_number_time_samples=True in compute_acf() then the following applies. Otherwise, the above applies."
                f"No. of acf slices = {self.no_slices}\n"
                f"No. dumps per slice = {int(self.acf_slice_steps)}\n"
                f"Largest time lag of the autocorrelation function: tau = {tau:.4e} {self.units_dict['time']} ~ {tau_wp} plasma periods\n\n"
            )

        if self.k_observable:
            if self.__name__ == "ccf":
                kt_file = f"v(k,t) data saved in: \n {self.vkt_hdf_file}\n"
            else:
                kt_file = f"n(k,t) data saved in: \n {self.nkt_hdf_file}\n"
            k_msg = (
                f"k wave vector information saved in:\n {self.k_file}\n"
                f"{kt_file}"
                f"Data saved in: \n {self.filename_hdf}\n"
                f"Data accessible at: self.k_list, self.k_counts, self.ka_values, self.frequencies, self.dataframe\n"
                f"\nWave vector parameters:\n"
                f"Smallest wave vector k_min = 2 pi / L = 3.9 / N^(1/3)\n"
                f"k_min = {self.ka_values[0]:.4f} / a_ws = {self.ka_values[0] / self.a_ws:.4e} {self.units_dict['inverse length']}\n"
                f"\nAngle averaging choice: {self.angle_averaging}\n"
            )
            if self.angle_averaging == "full":
                nx, ny, nz = unwrap(self.max_aa_harmonics)
                aa_k_msg = (
                    f"\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {nx}, {ny}, {nz}\n"
                    f"\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)\n"
                    f"\tk_max = {self.max_aa_ka_value:.4f} / a_ws = {self.max_aa_ka_value / self.a_ws :1.4e} {self.units_dict['inverse length']}\n"
                )

            elif self.angle_averaging == "custom":
                nx, ny, nz = unwrap(self.max_aa_harmonics)
                knx, kny, knz = unwrap(self.max_k_harmonics)

                aa_k_msg = (
                    f"\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {nx}, {ny}, {nz}\n"
                    f"\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)\n"
                    f"\tAA k_max = {self.max_aa_ka_value:.4f} / a_ws = {self.max_aa_ka_value / self.a_ws :1.4e} {self.units_dict['inverse length']}\n"
                    f"\tMaximum k harmonics = n_x, n_y, n_z = {knx}, {kny}, {knz}\n"
                    f"\tLargest wave vector k_max = k_min * n_x\n"
                    f"\tk_max = {self.max_ka_value:.4f} / a_ws = {self.max_ka_value / self.a_ws:.4e} {self.units_dict['inverse length']}\n"
                )
            elif self.angle_averaging == "principal_axis":
                knx, kny, knz = unwrap(self.max_k_harmonics)

                aa_k_msg = (
                    f"\tMaximum k harmonics = n_x, n_y, n_z = {knx}, {kny}, {knz}\n"
                    f"\tLargest wave vector k_max = k_min * n_x\n"
                    f"\tk_max = {self.max_ka_value:.4f} / a_ws = {self.max_ka_value / self.a_ws:.4e} {self.units_dict['inverse length']}\n"
                )

            aa_k_msg += (
                f"\nTotal number of k values to calculate = {len(self.k_list)}\n"
                f"No. of unique ka values to calculate = {len(self.ka_values)}\n"
            )

            k_msg += aa_k_msg
            if self.kw_observable:
                kw_msg = (
                    f"\nFrequency Space Parameters:\n"
                    f"\tNo. of slices = {self.no_slices}\n"
                    f"\tNo. dumps per slice = {self.slice_steps}\n"
                    f"\tFrequency step dw = 2 pi /(slice_steps * dump_step * dt)\n"
                    f"\tdw = {self.w_min / self.total_plasma_frequency:.4f} w_p = {self.w_min:.4e} {self.units_dict['frequency']}\n"
                    f"\tMaximum Frequency w_max = pi /(dump_step * dt)\n"
                    f"\tw_max = {self.w_max / self.total_plasma_frequency:.4f} w_p = {self.w_max:.4e} {self.units_dict['frequency']}\n"
                )
                k_msg += kw_msg

                msg += k_msg

        return msg

    def from_pickle(self):
        """Read the observable's info from the pickle file."""
        self.filename_pickle = os_path_join(self.saving_dir, self.__long_name__.replace(" ", "") + ".pickle")
        with open(self.filename_pickle, "rb") as pkl_data:
            data = pickle_load(pkl_data)
        self.from_dict(data.__dict__)

    def save_hdf(self):

        # Create the columns for the HDF df
        if not self.k_observable:
            if not isinstance(self.dataframe_slices.columns, MultiIndex):
                self.dataframe_slices.columns = MultiIndex.from_tuples(
                    [tuple(c.split("_")) for c in self.dataframe_slices.columns]
                )

        if not isinstance(self.dataframe.columns, MultiIndex):
            self.dataframe.columns = MultiIndex.from_tuples([tuple(c.split("_")) for c in self.dataframe.columns])

        # Sort the index for speed
        # see https://stackoverflow.com/questions/54307300/what-causes-indexing-past-lexsort-depth-warning-in-pandas
        self.dataframe = self.dataframe.sort_index()
        self.dataframe_slices = self.dataframe_slices.sort_index()

        # TODO: Fix this hack. We should be able to add data to HDF instead of removing it and rewriting it.
        # Save the data.
        if os_path_exists(self.filename_hdf_slices):
            os_remove(self.filename_hdf_slices)
        self.dataframe_slices.to_hdf(self.filename_hdf_slices, mode="w", key=self.__name__)

        if os_path_exists(self.filename_hdf):
            os_remove(self.filename_hdf)
        self.dataframe.to_hdf(self.filename_hdf, mode="w", key=self.__name__)

    def save_acf_hdf(self):
   
        if not isinstance(self.dataframe_acf.columns, pd.MultiIndex):
            self.dataframe_acf.columns = MultiIndex.from_tuples([tuple(c.split("_")) for c in self.dataframe_acf.columns])

        if not isinstance(self.dataframe_acf_slices.columns, pd.MultiIndex):
            self.dataframe_acf_slices.columns = MultiIndex.from_tuples(
                [tuple(c.split("_")) for c in self.dataframe_acf_slices.columns]
            )

        self.dataframe_acf = self.dataframe_acf.sort_index()
        self.dataframe_acf_slices = self.dataframe_acf_slices.sort_index()

        if os_path_exists(self.filename_hdf_acf):
            os_remove(self.filename_hdf_acf)
        self.dataframe_acf.to_hdf(self.filename_hdf_acf, mode="w", key=self.__name__)

        if os_path_exists(self.filename_hdf_acf_slices):
            os_remove(self.filename_hdf_acf_slices)
        self.dataframe_acf_slices.to_hdf(self.filename_hdf_acf_slices, mode="w", key=self.__name__)

    def save_kt_hdf(self, nkt_flag: bool = False, vkt_flag: bool = False):
        """
        Save the :math:`n(\\mathbf{k},t)` and/or :math:`\mathbf{v}(\\mathbf{k},t)` data of each slice to disk. \n
        The data is contained in :py:attr:~.nkt_dataframe_slices and :py:attr:~.vkt_dataframe_slices respectively.\n
        Each dataframe is stored as a HDF5 file to disk. The location of the data is at :py:attr:~.nkt_hdf_file and :py:attr:~.vkt_hdf_file.

        Parameters
        ----------
        nkt_flag: bool
            Flag for saving :math:`n(\\mathbf{k},t)` data. Default is False.

        vkt_flag : bool
            Flag for saving :math:`\\mathbf{v} (\\mathbf{k},t)` data. Default is False.

        """
        if nkt_flag:
            # Save the data and append metadata
            if os_path_exists(self.nkt_hdf_file):
                os_remove(self.nkt_hdf_file)

            hfile = HDFStore(self.nkt_hdf_file, mode="w")
            hfile.put("nkt", self.nkt_dataframe_slices)
            # This metadata is needed to check if I need to recalculate
            metadata = {
                "no_slices": self.no_slices,
                "max_k_harmonics": self.max_k_harmonics,
                "angle_averaging": self.angle_averaging,
            }

            hfile.get_storer("nkt").attrs.metadata = metadata
            hfile.close()

        if vkt_flag:
            if os_path_exists(self.vkt_hdf_file):
                os_remove(self.vkt_hdf_file)
            # Save the data and append metadata
            hfile = HDFStore(self.vkt_hdf_file, mode="w")
            hfile.put("vkt", self.vkt_dataframe_slices)
            # This metadata is needed to check if I need to recalculate
            metadata = {
                "no_slices": self.no_slices,
                "max_k_harmonics": self.max_k_harmonics,
                "angle_averaging": self.angle_averaging,
            }

            hfile.get_storer("vkt").attrs.metadata = metadata
            hfile.close()

    def save_pickle(self):
        """Save the observable's info into a pickle file."""
        self.filename_pickle = os_path_join(self.saving_dir, self.__long_name__.replace(" ", "") + ".pickle")
        with open(self.filename_pickle, "wb") as pickle_file:
            dump(self, pickle_file)
            pickle_file.close()

    def setup_init(
        self,
        params,
        phase: str = None,
        no_slices: int = None,
        multi_run_average: bool = None,
        dimensional_average: bool = None,
        runs: int = None,
    ):
        """
        Assign Observables attributes and copy the simulation's parameters.

        Parameters
        ----------
        runs
        dimensional_average
        multi_run_average
        params : sarkas.core.Parameters
            Simulation's parameters.

        phase : str, optional
            Phase to compute. Default = 'production'.

        no_slices : int, optional
            Number of independent runs inside a long simulation. Default = 1.

        """

        if phase:
            self.phase = phase.lower()

        if no_slices:
            self.no_slices = no_slices

        if multi_run_average:
            self.multi_run_average = multi_run_average

        if dimensional_average:
            self.dimensional_average = dimensional_average

        if runs:
            self.runs = runs

        # The dict update could overwrite the names
        name = self.__name__
        long_name = self.__long_name__

        self.copy_params(params)

        # Get the right labels
        if self.dimensions == 3:
            self.dim_labels = ["X", "Y", "Z"]
        elif self.dimensions == 2:
            self.dim_labels = ["X", "Y"]

        # Restore the correct names
        self.__name__ = name
        self.__long_name__ = long_name

        if self.k_observable:
            # Check for k space information.
            if self.angle_averaging in ["full", "principal_axis"]:
                if self.max_k_harmonics is None and self.max_ka_value is None:
                    raise AttributeError("max_ka_value and max_k_harmonics not defined.")
            elif self.angle_averaging == "custom":
                # if "custom" I expect that there is a portion that must be angle averaged. Hence, check for those values.
                if self.max_aa_ka_value is None and self.max_aa_harmonics is None:
                    raise AttributeError("max_aa_harmonics and max_aa_ka_value not defined.")

            # More checks on k attributes and initialization of k vectors
            if self.max_k_harmonics is not None:
                # Update angle averaged attributes depending on the choice of angle_averaging

                # Convert max_k_harmonics to a numpy array. YAML reads in a list.
                if not isinstance(self.max_k_harmonics, ndarray):
                    self.max_k_harmonics = array([self.max_k_harmonics for i in range(3)])
                    if self.dimensions < 3:
                        self.max_k_harmonics[2] = 0

                # Calculate max_aa_harmonics based on the choice of angle averaging and inputs
                if self.angle_averaging == "full":
                    self.max_aa_harmonics = self.max_k_harmonics.copy()

                elif self.angle_averaging == "custom":
                    # Check if the user has defined the max_aa_harmonics
                    if self.max_aa_ka_value:
                        nx = int(self.max_aa_ka_value * self.box_lengths[0] / (2.0 * pi * self.a_ws * sqrt(3.0)))
                        ny = int(self.max_aa_ka_value * self.box_lengths[1] / (2.0 * pi * self.a_ws * sqrt(3.0)))
                        nz = int(self.max_aa_ka_value * self.box_lengths[2] / (2.0 * pi * self.a_ws * sqrt(3.0)))
                        self.max_aa_harmonics = array([nx, ny, nz])
                        if self.dimensions < 3:
                            self.max_aa_harmonics[2] = 0
                    # else max_aa_harmonics is user defined
                elif self.angle_averaging == "principal_axis":
                    self.max_aa_harmonics = array([0, 0, 0])
                # max_ka_value is still None

            elif self.max_ka_value is not None:
                # Update max_k_harmonics and angle average attributes based on the choice of angle_averaging
                # Calculate max_k_harmonics from max_ka_value

                # Check for angle_averaging choice
                if self.angle_averaging == "full":
                    # The maximum value is calculated assuming that max nx = max ny = max nz
                    # ka_max = 2pi a/L sqrt( nx^2 + ny^2 + nz^2) = 2pi a/L nx sqrt(3)
                    nx = int(self.max_ka_value * self.box_lengths[0] / (2.0 * pi * self.a_ws * sqrt(3.0)))
                    ny = int(self.max_ka_value * self.box_lengths[1] / (2.0 * pi * self.a_ws * sqrt(3.0)))
                    nz = int(self.max_ka_value * self.box_lengths[2] / (2.0 * pi * self.a_ws * sqrt(3.0)))
                    self.max_k_harmonics = array([nx, ny, nz])
                    self.max_aa_harmonics = array([nx, ny, nz])
                    if self.dimensions < 3:
                        self.max_aa_harmonics[2] = 0
                        self.max_k_harmonics[2] = 0

                elif self.angle_averaging == "custom":
                    # ka_max = 2pi a/L sqrt( nx^2 + 0 + 0) = 2pi a/L nx
                    nx = int(self.max_ka_value * self.box_lengths[0] / (2.0 * pi * self.a_ws))
                    ny = int(self.max_ka_value * self.box_lengths[1] / (2.0 * pi * self.a_ws))
                    nz = int(self.max_ka_value * self.box_lengths[2] / (2.0 * pi * self.a_ws))
                    self.max_k_harmonics = array([nx, ny, nz])
                    # Check if the user has defined the max_aa_harmonics
                    if self.max_aa_ka_value:
                        nx = int(self.max_aa_ka_value * self.box_lengths[0] / (2.0 * pi * self.a_ws * sqrt(3.0)))
                        ny = int(self.max_aa_ka_value * self.box_lengths[1] / (2.0 * pi * self.a_ws * sqrt(3.0)))
                        nz = int(self.max_aa_ka_value * self.box_lengths[2] / (2.0 * pi * self.a_ws * sqrt(3.0)))
                        self.max_aa_harmonics = array([nx, ny, nz])
                    # else max_aa_harmonics is user defined
                elif self.angle_averaging == "principal_axis":
                    # ka_max = 2pi a/L sqrt( nx^2 + 0 + 0) = 2pi a/L nx
                    nx = int(self.max_ka_value * self.box_lengths[0] / (2.0 * pi * self.a_ws))
                    ny = int(self.max_ka_value * self.box_lengths[1] / (2.0 * pi * self.a_ws))
                    nz = int(self.max_ka_value * self.box_lengths[2] / (2.0 * pi * self.a_ws))
                    self.max_k_harmonics = array([nx, ny, nz])
                    self.max_aa_harmonics = array([0, 0, 0])

            # Calculate the maximum ka value based on user's choice of angle_averaging
            # Dev notes: Make sure max_ka_value, max_aa_ka_value are defined when this if is done
            if self.angle_averaging == "full":
                self.max_ka_value = 2.0 * pi * self.a_ws * norm(self.max_k_harmonics / self.box_lengths)
                self.max_aa_ka_value = 2.0 * pi * self.a_ws * norm(self.max_k_harmonics / self.box_lengths)

            elif self.angle_averaging == "principal_axis":
                self.max_ka_value = 2.0 * pi * self.a_ws * self.max_k_harmonics[0] / self.box_lengths[0]
                self.max_aa_ka_value = 0.0

            elif self.angle_averaging == "custom":
                self.max_aa_ka_value = 2.0 * pi * self.a_ws * norm(self.max_aa_harmonics / self.box_lengths)
                self.max_ka_value = 2.0 * pi * self.a_ws * self.max_k_harmonics[0] / self.box_lengths[0]

        # Get the number of independent observables if multi-species
        self.no_obs = int(self.num_species * (self.num_species + 1) / 2)

        # Get the total number of dumps by looking at the files in the directory
        self.dump_dir = self.directory_tree["postprocessing"][self.phase]["dumps"]["path"]

        self.prod_no_dumps = len(listdir(self.directory_tree["postprocessing"]["production"]["dumps"]["path"]))
        self.eq_no_dumps = len(listdir(self.directory_tree["postprocessing"]["equilibration"]["dumps"]["path"]))

        # Check for magnetized plasma options
        if self.magnetized and self.electrostatic_equilibration:
            self.mag_no_dumps = len(listdir(self.directory_tree["postprocessing"]["magnetization"]["dumps"]["path"]))

        # Assign dumps variables based on the choice of phase
        if self.phase == "equilibration":
            self.no_dumps = self.eq_no_dumps
            self.dump_step = self.eq_dump_step
            self.no_steps = self.equilibration_steps

        elif self.phase == "production":
            self.no_dumps = self.prod_no_dumps
            self.dump_step = self.prod_dump_step
            self.no_steps = self.production_steps

        elif self.phase == "magnetization":
            self.no_dumps = self.mag_no_dumps
            self.dump_step = self.mag_dump_step
            self.no_steps = self.magnetization_steps

        # Needed for preprocessing pretty print
        self.slice_steps = (
            int(self.no_steps / self.dump_step / self.no_slices)
            if self.no_dumps < self.no_slices
            else int(self.no_dumps / self.no_slices)
        )

        self.acf_slice_steps = (
            int(self.no_steps / self.dump_step / (self.no_slices + 1))
            if self.no_dumps < self.no_slices
            else int(self.no_dumps / (self.no_slices + 1))
        )

        # Array containing the start index of each species.
        self.species_index_start = array([0, *self.species_num.cumsum()], dtype=int)

    def setup_multirun_dirs(self):
        """Set the attributes postprocessing_dir and dump_dirs_list.

        The attribute postprocessing_dir refers to the location where to store postprocessing results.
        If the attribute :py:attr:`sarkas.tools.observables.Observable.multi_run_average` is set to `True` then the
        postprocessing data will be saved in the :py:attr:`sarkas.core.Parameters.md_simulations_dir` directory, i.e. where
        all the runs are.
        Otherwise :py:attr:`sarkas.tools.observables.Observable.postprocessing_dir` will be
        in the :py:attr:`sarkas.core.Parameters.job_dir`.

        The attribute :py:attr:`sarkas.tools.observables.Observable.dump_dirs_list` is a list of the refers to the locations
        of the production (or other phases) dumps. If :py:attr:`sarkas.tools.observables.Observable.multi_run_average` is
        `False` then the list will contain only one path, namely :py:attr:`sarkas.tools.observables.Observable.dump_dir`.

        """
        self.dump_dirs_list = []

        if self.multi_run_average:
            for r in range(self.runs):
                # Direct to the correct dumps directory
                dump_dir = os_path_join(
                    f"run{r}", os_path_join("Simulation", os_path_join(self.phase.capitalize(), "dumps"))
                )
                dump_dir = os_path_join(self.md_simulations_dir, dump_dir)
                self.dump_dirs_list.append(dump_dir)

            # Re-path the saving directory.
            # Data is saved in Simulations/PostProcessing/Observable/Phase/
            self.postprocessing_dir = os_path_join(self.md_simulations_dir, "PostProcessing")
            if not os_path_exists(self.postprocessing_dir):
                mkdir(self.postprocessing_dir)
        else:
            self.dump_dirs_list = [self.dump_dir]

    def update_finish(self):
        """Update the :py:attr:`~.slice_steps`, CCF's and DSF's attributes, and save pickle file with observable's info.

        Notes
        -----
        The information is saved without the dataframe(s).

        """
        self.create_dirs_filenames()

        # Needed if no_slices has been passed
        self.slice_steps = (
            int(self.no_steps / self.dump_step / self.no_slices)
            if self.no_dumps < self.no_slices
            else int(self.no_dumps / self.no_slices)
        )

        if self.k_observable:
            self.parse_k_data()

        if self.kw_observable:
            # These calculation are needed for the io.postprocess_info().
            # This is a hack and we need to find a faster way to do it
            dt_r = self.dt * self.dump_step

            self.w_min = 2.0 * pi / (self.slice_steps * dt_r)
            self.w_max = pi / dt_r  # Half because fft calculates negative and positive frequencies
            self.frequencies = 2.0 * pi * fftfreq(self.slice_steps, dt_r)
            self.frequencies = fftshift(self.frequencies)

        self.save_pickle()

        self.initialize_hdf()

        if self.__name__ == "ccf":
            self.dataframe_longitudinal = DataFrame()
            self.dataframe_longitudinal_slices = DataFrame()
            self.dataframe_transverse = DataFrame()
            self.dataframe_transverse_slices = DataFrame()

        if self.acf_observable:
            self.acf_slice_steps = (
                int(self.no_steps / self.dump_step / (self.no_slices + 1))
                if self.no_dumps < self.no_slices
                else int(self.no_dumps / (self.no_slices + 1))
            )

        # Write log file
        datetime_stamp(self.log_file)
        msg = self.pretty_print_msg()
        print_to_logger(msg, self.log_file, self.verbose)

    def initialize_hdf(self):

        self.dataframe = DataFrame()
        self.dataframe_slices = DataFrame()
        if self.acf_observable:
            self.dataframe_acf = DataFrame()
            self.dataframe_acf_slices = DataFrame()


class CurrentCorrelationFunction(Observable):
    """
    Current Correlation Functions. \n

    The species dependent longitudinal ccf :math:`L_{AB}(\\mathbf k, \\omega)` is defined as

    .. math::

        L_{AB}(\\mathbf k,\\omega) = \\int_0^\\infty dt \\,
        \\left \\langle \\left [\\mathbf k \\cdot \\mathbf v_{A} ( \\mathbf k, t) \\right ]
        \\left [ - \\mathbf k \\cdot \\mathbf v_{B} ( -\\mathbf k, t) \\right \\rangle \\right ]
        e^{i \\omega t},

    while the transverse are

    .. math::

        T_{AB}(\\mathbf k,\\omega) = \\int_0^\\infty dt \\,
        \\left \\langle \\left [ \\mathbf k \\times \\mathbf v_{A} ( \\mathbf k, t) \\right ] \\cdot
        \\left [  -\\mathbf k \\times \\mathbf v_{A} ( -\\mathbf k, t) \\right \\rangle \\right ]
        e^{i \\omega t},

    where the microscopic velocity of species :math:`A` with number of particles :math:`N_{A}` is given by

    .. math::
        \\mathbf v_{A}(\\mathbf k,t) = \\sum^{N_{A}}_{j} \\mathbf v_j(t) e^{-i \\mathbf k \\cdot \\mathbf r_j(t)} .


    Attributes
    ----------
    k_list : list
        List of all possible :math:`k` vectors with their corresponding magnitudes and indexes.

    k_counts : numpy.ndarray
        Number of occurrences of each :math:`k` magnitude.

    ka_values : numpy.ndarray
        Magnitude of each allowed :math:`ka` vector.

    no_ka_values: int
        Length of :py:attr:`~.ka_values` array.

    """

    def __init__(self):
        super().__init__()
        self.__name__ = "ccf"
        self.__long_name__ = "Current Correlation Function"
        self.k_observable = True
        self.kw_observable = True

    @avg_slices_doc
    def average_slices_data(self, longitudinal: bool = False, transverse: bool = False):

        # Now the actual dataframe
        if longitudinal:
            self.dataframe_longitudinal[" _ _Frequencies"] = self.frequencies
        if transverse:
            self.dataframe_transverse[" _ _Frequencies"] = self.frequencies

        # Take the mean and std and store them into the dataframe to return
        for sp1, sp1_name in enumerate(self.species_names):
            for sp2, sp2_name in enumerate(self.species_names[sp1:], sp1):
                comp_name = f"{sp1_name}-{sp2_name}"
                # Get the k_harmonics columns over which to average
                ka_columns_mean = [
                    comp_name + "_Mean_ka{} = {:.4f}".format(ik + 1, ka) for ik, ka in enumerate(self.ka_values)
                ]

                # Std
                ka_columns_std = [
                    comp_name + "_Std_ka{} = {:.4f}".format(ik + 1, ka) for ik, ka in enumerate(self.ka_values)
                ]
                if longitudinal:
                    # Mean: level = 1 corresponds to averaging all the k harmonics with the same magnitude
                    df_mean = self.dataframe_longitudinal_slices[comp_name].groupby(level=1, axis="columns").mean()
                    df_mean = df_mean.rename(col_mapper(df_mean.columns, ka_columns_mean), axis=1)

                    df_std = self.dataframe_longitudinal_slices[comp_name].groupby(level=1, axis="columns").std()
                    df_std = df_std.rename(col_mapper(df_std.columns, ka_columns_std), axis=1)

                    self.dataframe_longitudinal = concat([self.dataframe_longitudinal, df_mean, df_std], axis=1)

                if transverse:
                    # Mean: level = 1 corresponds to averaging all the k harmonics with the same magnitude
                    tdf_mean = self.dataframe_transverse_slices[comp_name].groupby(level=1, axis="columns").mean()
                    tdf_mean = tdf_mean.rename(col_mapper(tdf_mean.columns, ka_columns_mean), axis=1)
                    # Std
                    tdf_std = self.dataframe_transverse_slices[comp_name].groupby(level=1, axis="columns").std()
                    tdf_std = tdf_std.rename(col_mapper(tdf_std.columns, ka_columns_std), axis=1)

                    self.dataframe_transverse = concat([self.dataframe_transverse, tdf_mean, tdf_std], axis=1)

    @calc_slices_doc
    def calc_longitudinal_data(self):

        self.parse_kt_data(nkt_flag=False, vkt_flag=True)
        vkt_df = read_hdf(self.vkt_hdf_file, mode="r", key="vkt")
        # Add frequencies
        self.dataframe_longitudinal_slices[" _ _ _Frequencies"] = self.frequencies
        # Containers
        vkt = zeros((self.num_species, self.slice_steps, len(self.k_list)), dtype=complex128)
        for isl in tqdm(range(self.no_slices), desc="Calculating longitudinal CCF for slice", disable=not self.verbose):

            # Put data in the container to pass
            for sp, sp_name in enumerate(self.species_names):
                vkt[sp] = array(vkt_df[f"slice {isl + 1}"]["Longitudinal"][sp_name])

            # Calculate Lkw and Tkw
            Lkw = calc_Skw(vkt, self.k_list, self.species_num, self.slice_steps, self.dt, self.dump_step)

            # Create the dataframe's column names
            slc_column = f"slice {isl + 1}"
            ka_columns = ["ka = {:.6f}".format(ka) for ka in self.ka_values]

            # Save the full Lkw into a Dataframe
            sp_indx = 0
            for i, sp1 in enumerate(self.species_names):
                for j, sp2 in enumerate(self.species_names[i:]):

                    # Create the list of column names
                    # Final string : Longitudinal_H-He_slice 1_ka = 0.123456_k = [0, 0, 1]
                    column_names = []
                    col_sp_slc = f"{sp1}-{sp2}_" + slc_column
                    for ik in range(len(self.k_harmonics)):
                        ka_value = ka_columns[int(self.k_harmonics[ik, -1])]
                        col_name = col_sp_slc + f"_{ka_value}"
                        k_harm_col = f"_k = ["
                        for d in range(self.dimensions):
                            k_harm_col += f"{self.k_harmonics[ik, d].astype(int)}, "

                        # The above loop added ", " at the end. I don't want it
                        k_harm_col = k_harm_col[:-2] + f"]"
                        col_name += k_harm_col
                        column_names.append(col_name)

                    self.dataframe_longitudinal_slices = concat(
                        [self.dataframe_longitudinal_slices, DataFrame(Lkw[sp_indx, :, :].T, columns=column_names)],
                        axis=1,
                    )
                    sp_indx += 1

        # Create the MultiIndex
        tuples = [tuple(c.split("_")) for c in self.dataframe_longitudinal_slices.columns]
        self.dataframe_longitudinal_slices.columns = MultiIndex.from_tuples(
            tuples, names=["species", "slices", "ka_value", "k_harmonics"]
        )

    @calc_slices_doc
    def calc_transverse_data(self):

        self.parse_kt_data(nkt_flag=False, vkt_flag=True)

        # Add frequencies
        self.dataframe_transverse_slices[" _ _ _Frequencies"] = self.frequencies
        vkt_df = read_hdf(self.vkt_hdf_file, mode="r", key="vkt")

        # Containers
        vkt_d = zeros((self.num_species, self.slice_steps, len(self.k_list)), dtype=complex128)
        # number of independent observables
        no_skw = int(len(self.species_num) * (len(self.species_num) + 1) / 2)
        Tkw = zeros((no_skw, len(self.k_list), self.slice_steps))

        for isl in tqdm(
            range(self.no_slices), desc="Calculating transverse CCF for slice", position=0, disable=not self.verbose
        ):

            for d, dim in tqdm(
                zip(range(self.dimensions), ["i", "j", "k"]),
                desc="Dimension",
                position=1,
                disable=not self.verbose,
                leave=False,
            ):
                # Put data in the container to pass
                for sp, sp_name in enumerate(self.species_names):
                    vkt_d[sp] = array(vkt_df[f"slice {isl + 1}"][f"Transverse {dim}"][sp_name])
                    Tkw_d = calc_Skw(vkt_d, self.k_list, self.species_num, self.slice_steps, self.dt, self.dump_step)

                    Tkw += Tkw_d / self.dimensions

            # Create the dataframe's column names
            slc_column = "slice {}".format(isl + 1)
            ka_columns = ["ka = {:.6f}".format(ka) for ik, ka in enumerate(self.ka_values)]

            # Save the full Lkw into a Dataframe
            sp_indx = 0
            for i, sp1 in enumerate(self.species_names):
                for j, sp2 in enumerate(self.species_names[i:]):

                    # Create the list of column names
                    # Final string : H-He_slice 1_ka = 0.123456_k = [0, 0, 1]
                    column_names = []
                    col_sp_slc = f"{sp1}-{sp2}_" + slc_column
                    for ik in range(len(self.k_harmonics)):
                        ka_value = ka_columns[int(self.k_harmonics[ik, -1])]
                        col_name = col_sp_slc + f"_{ka_value}"
                        k_harm_col = f"_k = ["
                        for d in range(self.dimensions):
                            k_harm_col += f"{self.k_harmonics[ik, d].astype(int)}, "

                        # The above loop added ", " at the end. I don't want it
                        k_harm_col = k_harm_col[:-2] + f"]"
                        col_name += k_harm_col
                        column_names.append(col_name)

                    self.dataframe_transverse_slices = concat(
                        [self.dataframe_transverse_slices, DataFrame(Tkw[sp_indx, :, :].T, columns=column_names)], axis=1
                    )

                    sp_indx += 1

        # Create the MultiIndex
        tuples = [tuple(c.split("_")) for c in self.dataframe_transverse_slices.columns]
        self.dataframe_transverse_slices.columns = MultiIndex.from_tuples(
            tuples, names=["species", "slices", "ka_value", "k_harmonics"]
        )

    def compute(self, longitudinal: bool = False, transverse: bool = False):
        """
        Routine for computing the current correlation function. See class doc for exact quantities.\n
        The data of each slice is saved in hierarchical dataframes,
        :py:attr:`~.dataframe_longitudinal_slices` or :py:attr:`~.dataframe_transverse_slices`. \n
        The sliced averaged data is saved in other hierarchical dataframes,
        :py:attr:`~.dataframe_longitudinal` :py:attr:`~.dataframe_transverse`.

        Parameters
        ----------
        longitudinal : bool
            Flag for calculating the longitudinal CCF. Default = False.

        transverse : bool
            Flag for calculating the transverse CCF. Default = False.

        """

        t0 = self.timer.current()
        if longitudinal:
            self.calc_longitudinal_data()
        if transverse:
            self.calc_transverse_data()

        self.average_slices_data(longitudinal, transverse)
        self.save_hdf(longitudinal, transverse)
        tend = self.timer.current()
        time_stamp(self.log_file, self.__long_name__ + " Calculation", self.timer.time_division(tend - t0), self.verbose)

    def parse(self, longitudinal: bool = False, transverse: bool = False):

        if longitudinal and not transverse:
            try:
                self.dataframe_longitudinal = read_hdf(self.filename_hdf_longitudinal, mode="r", index_col=False)

                k_data = load(self.k_file)
                self.k_list = k_data["k_list"]
                self.k_counts = k_data["k_counts"]
                self.ka_values = k_data["ka_values"]

            except FileNotFoundError:
                msg = f"\nFile {self.filename_hdf_longitudinal} not found!\nComputing longitudinal ccf ..."
                print_to_logger(msg, self.log_file, self.verbose)
                self.compute(longitudinal=longitudinal)

        elif transverse and not longitudinal:

            try:
                self.dataframe_transverse = read_hdf(self.filename_hdf_transverse, mode="r", index_col=False)

                k_data = load(self.k_file)
                self.k_list = k_data["k_list"]
                self.k_counts = k_data["k_counts"]
                self.ka_values = k_data["ka_values"]

            except FileNotFoundError:
                msg = f"\nFile {self.filename_hdf_transverse} not found!\nComputing transverse ccf ..."
                print_to_logger(msg, self.log_file, self.verbose)
                self.compute(longitudinal=False, transverse=True)

        elif longitudinal and transverse:

            try:
                self.dataframe_longitudinal = read_hdf(self.filename_hdf_longitudinal, mode="r", index_col=False)
                self.dataframe_transverse = read_hdf(self.filename_hdf_transverse, mode="r", index_col=False)

                k_data = load(self.k_file)
                self.k_list = k_data["k_list"]
                self.k_counts = k_data["k_counts"]
                self.ka_values = k_data["ka_values"]

            except FileNotFoundError:
                msg = f"\nFile {self.filename_hdf_longitudinal} or {self.filename_hdf_transverse} not found!\nComputing all ccf ..."
                print_to_logger(msg, self.log_file, self.verbose)
                self.compute(longitudinal=True, transverse=True)

        else:
            msg = (
                "Direction not defined. Call the method with either option set to true."
                "\nlongitudinal = True/False, transverse = True/False"
            )
            print_to_logger(msg, self.log_file, self.verbose)

    @setup_doc
    def setup(self, params, phase: str = None, no_slices: int = None, **kwargs):

        super().setup_init(params, phase=phase, no_slices=no_slices)

        self.update_args(**kwargs)

    def save_hdf(self, longitudinal: bool = False, transverse: bool = False):

        # Create the columns for the HDF df
        if longitudinal:
            if not isinstance(self.dataframe_longitudinal_slices.columns, MultiIndex):
                self.dataframe_longitudinal_slices.columns = MultiIndex.from_tuples(
                    [tuple(c.split("_")) for c in self.dataframe_longitudinal_slices.columns]
                )

            if not isinstance(self.dataframe_longitudinal.columns, MultiIndex):
                self.dataframe_longitudinal.columns = MultiIndex.from_tuples(
                    [tuple(c.split("_")) for c in self.dataframe_longitudinal.columns]
                )

            # Sort the index for speed
            # see https://stackoverflow.com/questions/54307300/what-causes-indexing-past-lexsort-depth-warning-in-pandas
            self.dataframe_longitudinal = self.dataframe_longitudinal.sort_index()
            self.dataframe_longitudinal_slices = self.dataframe_longitudinal_slices.sort_index()

            # TODO: Fix this hack. We should be able to add data to HDF instead of removing it and rewriting it.
            # Save the data.

            if os_path_exists(self.filename_hdf_longitudinal_slices):
                os_remove(self.filename_hdf_longitudinal_slices)
            self.dataframe_longitudinal_slices.to_hdf(self.filename_hdf_longitudinal_slices, mode="w", key=self.__name__)

            if os_path_exists(self.filename_hdf_longitudinal):
                os_remove(self.filename_hdf_longitudinal)
            self.dataframe_longitudinal.to_hdf(self.filename_hdf_longitudinal, mode="w", key=self.__name__)

        # Create the columns for the HDF df
        if transverse:
            if not isinstance(self.dataframe_transverse_slices.columns, MultiIndex):
                self.dataframe_transverse_slices.columns = MultiIndex.from_tuples(
                    [tuple(c.split("_")) for c in self.dataframe_transverse_slices.columns]
                )

            if not isinstance(self.dataframe_transverse.columns, MultiIndex):
                self.dataframe_transverse.columns = MultiIndex.from_tuples(
                    [tuple(c.split("_")) for c in self.dataframe_transverse.columns]
                )

            # Sort the index for speed
            # see https://stackoverflow.com/questions/54307300/what-causes-indexing-past-lexsort-depth-warning-in-pandas
            self.dataframe_transverse = self.dataframe_transverse.sort_index()
            self.dataframe_transverse_slices = self.dataframe_transverse_slices.sort_index()

            # TODO: Fix this hack. We should be able to add data to HDF instead of removing it and rewriting it.
            # Save the data.

            if os_path_exists(self.filename_hdf_transverse_slices):
                os_remove(self.filename_hdf_transverse_slices)
            self.dataframe_transverse_slices.to_hdf(self.filename_hdf_transverse_slices, mode="w", key=self.__name__)

            if os_path_exists(self.filename_hdf_transverse):
                os_remove(self.filename_hdf_transverse)
            self.dataframe_transverse.to_hdf(self.filename_hdf_transverse, mode="w", key=self.__name__)

    @arg_update_doc
    def update_args(self, **kwargs):

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())
        self.update_finish()


class DiffusionFlux(Observable):
    """Diffusion Fluxes and their Auto-correlation functions.\n

    The :math:`\\alpha` diffusion flux :math:`\\mathbf J_{\\alpha}(t)` is calculated from eq.~(3.5) in :cite:`Zhou1996` which
    reads

    .. math::
        \\mathbf J_{\\alpha}(t) = \\frac{m_{\\alpha}}{m_{\\rm tot} }
            \\sum_{\\beta = 1}^{M} \\left ( m_{\\rm tot} \\delta_{\\alpha\\beta} - x_{\\alpha} m_{\\beta} \\right )
            \\mathbf j_{\\beta} (t)

    where :math:`M` is the total number of species, :math:`m_{\\rm tot} = \\sum_{i}^M m_{i}` is the total mass of the
    system, :math:`x_{\\alpha} = N_{\\alpha} / N_{\\rm tot}` is the concentration of species :math:`\\alpha`, and

    .. math::
        \\mathbf j_{\\alpha}(t) = \\sum_{i = 1}^{N_{\\alpha}} \\mathbf v_{i}(t)

    is the microscopic velocity field of species :math:`\\alpha`.

    """

    def __init__(self):
        super().__init__()
        self.__name__ = "diff_flux"
        self.__long_name__ = "Diffusion Flux"
        self.acf_observable = True

    @setup_doc
    def setup(self, params, phase: str = None, no_slices: int = None, **kwargs):

        super().setup_init(params, phase, no_slices)
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

        self.no_fluxes = self.num_species - 1
        self.no_fluxes_acf = int(self.no_fluxes * self.no_fluxes)

        self.update_finish()

    @compute_doc
    def compute(self):

        t0 = self.timer.current()
        # for run_idx, run_id in enumerate(self.runs):
        #     self.dump_dir = self.dump_dirs_list[run_idx]
        self.calc_slices_data()
        self.average_slices_data()
        self.save_hdf()
        tend = self.timer.current()
        time_stamp(self.log_file, self.__long_name__ + " Calculation", self.timer.time_division(tend - t0), self.verbose)

    @calc_slices_doc
    def calc_slices_data(self):

        start_slice = 0
        end_slice = self.slice_steps * self.dump_step
        time = zeros(self.slice_steps)

        df_str = "Diffusion Flux"
        df_acf_str = "Diffusion Flux ACF"

        for isl in tqdm(
            range(self.no_slices),
            desc=f"\nCalculating {df_str} for slice ",
            disable=not self.verbose,
            position=0,
        ):

            # Parse the particles from the dump files
            vel = zeros((self.dimensions, self.slice_steps, self.total_num_ptcls))
            #
            for it, dump in enumerate(
                tqdm(
                    range(start_slice, end_slice, self.dump_step),
                    desc="Reading data",
                    disable=not self.verbose,
                    position=1,
                    leave=False,
                )
            ):
                datap = load_from_restart(self.dump_dir, dump)
                time[it] = datap["time"]
                vel[0, it, :] = datap["vel"][:, 0]
                vel[1, it, :] = datap["vel"][:, 1]
                vel[2, it, :] = datap["vel"][:, 2]
            #
            if isl == 0:
                self.dataframe["Time"] = time
                self.dataframe_slices["Time"] = time
                self.dataframe_acf["Time"] = time
                self.dataframe_acf_slices["Time"] = time

            # This returns two arrays
            # diff_fluxes = array of shape (no_fluxes, no_dim, no_dumps_per_slice)
            # df_acf = array of shape (no_fluxes_acf, no_dim + 1, no_dumps_per_slice)
            diff_fluxes, df_acf = calc_diff_flux_acf(
                vel, self.species_num, self.species_concentrations, self.species_masses
            )

            # # Store the data
            for i, flux in enumerate(diff_fluxes):
                for d, dim in zip(range(self.dimensions), ["X", "Y", "Z"]):
                    col_name = df_str + f" {i}_{dim}_slice {isl}"
                    col_data = flux[d, :]
                    self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

            for i, flux_acf in enumerate(df_acf):
                for d, dim in zip(range(self.dimensions), ["X", "Y", "Z"]):
                    col_name = df_acf_str + f" {i}_{dim}_slice {isl}"
                    col_data = flux_acf[d, :]
                    self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, col_data, col_name)

                col_name = df_acf_str + f" {i}_Total_slice {isl}"
                col_data = flux_acf[-1, :]
                self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, col_data, col_name)

            start_slice += self.slice_steps * self.dump_step
            end_slice += self.slice_steps * self.dump_step

    @avg_slices_doc
    def average_slices_date(self):

        df_str = "Diffusion Flux"
        df_acf_str = "Diffusion Flux ACF"
        # Average and std over the slices
        for i in range(self.no_fluxes):
            for d, dim in zip(range(self.dimensions), ["X", "Y", "Z"]):
                dim_col_str = [df_str + f" {i}_{dim}_slice {isl}" for isl in range(self.no_slices)]

                col_name = df_str + f" {i}_{dim}_Mean"
                col_data = self.dataframe_slices[dim_col_str].mean(axis=1).values
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

                col_name = df_str + f" {i}_{dim}_Std"
                col_data = self.dataframe_slices[dim_col_str].std(axis=1).values
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

        # Average and std over the slices
        for i in range(self.no_fluxes_acf):
            for d, dim in zip(range(self.dimensions), ["X", "Y", "Z"]):
                dim_col_str = [df_acf_str + f" {i}_{dim}_slice {isl}" for isl in range(self.no_slices)]
                col_name = df_acf_str + f" {i}_{dim}_Mean"
                col_data = self.dataframe_acf_slices[dim_col_str].mean(axis=1).values
                self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

                col_name = df_acf_str + f" {i}_{dim}_Std"
                col_data = self.dataframe_acf_slices[dim_col_str].std(axis=1).values
                self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

            tot_col_str = [df_acf_str + f" {i}_Total_slice {isl}" for isl in range(self.no_slices)]
            # Average
            col_name = df_acf_str + f" {i}_Total_Mean"
            col_data = self.dataframe_acf_slices[tot_col_str].mean(axis=1).values
            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)
            # STD
            col_name = df_acf_str + f" {i}_Total_Std"
            col_data = self.dataframe_acf_slices[tot_col_str].std(axis=1).values
            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)


class DynamicStructureFactor(Observable):
    """Dynamic Structure factor.

    The species dependent DSF :math:`S_{AB}(k,\\omega)` is calculated from

    .. math::
        S_{AB}(k,\\omega) = \\int_0^\\infty dt \\,
        \\left \\langle | n_{A}( \\mathbf k, t)n_{B}( -\\mathbf k, t) \\right \\rangle e^{i \\omega t},

    where the microscopic density of species :math:`A` with number of particles :math:`N_{A}` is given by

    .. math::
        n_{A}(\\mathbf k,t) = \\sum^{N_{A}}_{j} e^{-i \\mathbf k \\cdot \\mathbf r_j(t)} .

    """

    def __init__(self):
        super().__init__()
        self.__name__ = "dsf"
        self.__long_name__ = "Dynamic Structure Factor"
        self.kw_observable = True
        self.k_observable = True

    @setup_doc
    def setup(self, params, phase: str = None, no_slices: int = 1, **kwargs):

        super().setup_init(params, phase, no_slices)
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())
        self.update_finish()

    @compute_doc
    def compute(self):

        t0 = self.timer.current()
        # for run_idx, run_id in enumerate(self.runs):
        #     self.dump_dir = self.dump_dirs_list[run_idx]
        self.calc_slices_data()
        self.average_slices_data()
        self.save_hdf()
        tend = self.timer.current()
        time_stamp(self.log_file, self.__long_name__ + " Calculation", self.timer.time_division(tend - t0), self.verbose)

    @calc_slices_doc
    def calc_slices_data(self):

        # Parse nkt otherwise calculate it
        self.parse_kt_data(nkt_flag=True)

        nkt_df = read_hdf(self.nkt_hdf_file, mode="r", key="nkt")
        for isl in tqdm(range(self.no_slices), desc="Calculating DSF for slice", disable=not self.verbose):
            # Initialize container
            nkt = zeros((self.num_species, self.slice_steps, len(self.k_list)), dtype=complex128)
            for sp, sp_name in enumerate(self.species_names):
                nkt[sp] = array(nkt_df["slice {}".format(isl + 1)][sp_name])

            # Calculate Skw
            Skw_all = calc_Skw(nkt, self.k_list, self.species_num, self.slice_steps, self.dt, self.dump_step)

            # Create the dataframe's column names
            slc_column = f"slice {isl + 1}"
            ka_columns = [f"ka = {ka:.8f}" for ik, ka in enumerate(self.ka_values)]
            # Save the full Skw into a Dataframe
            sp_indx = 0
            for i, sp1 in enumerate(self.species_names):
                for j, sp2 in enumerate(self.species_names[i:]):
                    columns = [
                        "{}-{}_".format(sp1, sp2)
                        + slc_column
                        + "_{}_k = [{}, {}, {}]".format(
                            ka_columns[int(self.k_harmonics[ik, -1])], *self.k_harmonics[ik, :-2].astype(int)
                        )
                        for ik in range(len(self.k_harmonics))
                    ]
                    self.dataframe_slices = concat(
                        [self.dataframe_slices, DataFrame(Skw_all[sp_indx, :, :].T, columns=columns)], axis=1
                    )
                    sp_indx += 1

        # Create the MultiIndex
        tuples = [tuple(c.split("_")) for c in self.dataframe_slices.columns]
        self.dataframe_slices.columns = MultiIndex.from_tuples(
            tuples, names=["species", "slices", "k_index", "k_harmonics"]
        )

    @avg_slices_doc
    def average_slices_data(self):

        # Now for the actual df
        self.dataframe[" _ _Frequencies"] = self.frequencies

        # Take the mean and std and store them into the dataframe to return
        for sp1, sp1_name in enumerate(self.species_names):
            for sp2, sp2_name in enumerate(self.species_names[sp1:], sp1):
                skw_name = f"{sp1_name}-{sp2_name}"
                # Rename the columns with values of ka
                ka_columns = [skw_name + f"_Mean_ka{ik + 1} = {ka:.4f}" for ik, ka in enumerate(self.ka_values)]
                # Mean: level = 1 corresponds to averaging all the k harmonics with the same magnitude
                df_mean = self.dataframe_slices[skw_name].groupby(level=1, axis=1).mean()
                df_mean = df_mean.rename(col_mapper(df_mean.columns, ka_columns), axis=1)
                # Std
                ka_columns = [skw_name + f"_Std_ka{ik + 1} = {ka:.4f}" for ik, ka in enumerate(self.ka_values)]
                df_std = self.dataframe_slices[skw_name].groupby(level=1, axis=1).std()
                df_std = df_std.rename(col_mapper(df_std.columns, ka_columns), axis=1)
                self.dataframe = concat([self.dataframe, df_mean, df_std], axis=1)


class ElectricCurrent(Observable):
    """Electric Current Auto-correlation function."""

    def __init__(self):
        super().__init__()
        self.__name__ = "ec"
        self.__long_name__ = "Electric Current"
        self.acf_observable = True

    @setup_doc
    def setup(self, params, phase: str = None, no_slices: int = None, **kwargs):

        super().setup_init(params, phase, no_slices)
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):

        # Update the attribute with the passed arguments. e.g time_averaging and timesteps_to_skip
        self.__dict__.update(kwargs.copy())
        self.update_finish()

    @compute_doc
    def compute(self):

        # Initialize timer
        t0 = self.timer.current()
        self.calc_slices_data()
        self.average_slices_data()
        self.save_hdf()
        tend = self.timer.current()
        time_stamp(self.log_file, "Electric Current Calculation", self.timer.time_division(tend - t0), self.verbose)

    def calc_slices_data(self):

        start_slice = 0
        end_slice = self.slice_steps * self.dump_step
        time = zeros(self.slice_steps)

        # Loop over the slices of each run
        for isl in tqdm(
            range(self.no_slices),
            desc=f"\nCalculating {self.__long_name__} for slice ",
            disable=not self.verbose,
            position=0,
        ):
            # Parse the particles from the dump files
            species_current = zeros((self.num_species, self.dimensions, self.slice_steps))
            for it, dump in enumerate(range(start_slice, end_slice, self.dump_step)):
                datap = load_from_restart(self.dump_dir, dump)
                time[it] = datap["time"]
                species_current[:, :, it] = datap["species_electric_current"]
            #
            if isl == 0:
                self.dataframe["Time"] = time.copy()
                self.dataframe_acf["Time"] = time.copy()
                self.dataframe_slices["Time"] = time.copy()
                self.dataframe_acf_slices["Time"] = time.copy()

            # species_current, total_current = calc_elec_current(vel, self.species_charges, self.species_num)
            total_current = species_current.sum(axis=0)

            # Store species data
            for i, sp_name in enumerate(self.species_names):
                sp_col_str = f"{sp_name} {self.__long_name__}"
                sp_col_str_acf = f"{sp_name} {self.__long_name__} ACF"
                sp_tot_acf = zeros(total_current.shape[1])
                for d in range(self.dimensions):
                    dl = self.dim_labels[d]
                    col_name = sp_col_str + f"_{dl}_slice {isl}"
                    col_data = species_current[i, d, :]
                    self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)
                    # Calculate ACF
                    col_data = correlationfunction(species_current[i, d, :], species_current[i, d, :])
                    col_name = sp_col_str_acf + f"_{dl}_slice {isl}"
                    self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, col_data, col_name)

                    sp_tot_acf += col_data

                # Store Total ACF of single species
                col_name = sp_col_str_acf + f"_Total_slice {isl}"
                col_data = sp_tot_acf
                self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, col_data, col_name)

            # Total current and its ACF
            tot_acf = zeros(total_current.shape[1])
            for d in range(self.dimensions):
                dl = self.dim_labels[d]
                col_name = f"{self.__long_name__}_{dl}_slice {isl}"
                col_data = total_current[d, :]
                self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

                # Calculate ACF
                col_data = correlationfunction(total_current[d, :], total_current[d, :])
                col_name = f"{self.__long_name__} ACF_{dl}_slice {isl}"
                self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, col_data, col_name)
                tot_acf += col_data

            col_data = tot_acf
            col_name = f"{self.__long_name__} ACF_Total_slice {isl}"
            self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, col_data, col_name)

            start_slice += self.slice_steps * self.dump_step
            end_slice += self.slice_steps * self.dump_step

    def average_slices_data(self):
        """Average and std over the slices."""

        # Species data
        for i, sp_name in enumerate(self.species_names):
            col_str = f"{sp_name} {self.__long_name__}"
            col_str_acf = f"{sp_name} {self.__long_name__} ACF"

            for d in range(self.dimensions):
                dl = self.dim_labels[d]
                dim_col_str = [f"{col_str}_{dl}_slice {isl}" for isl in range(self.no_slices)]

                col_data = self.dataframe_slices[dim_col_str].mean(axis=1).values
                col_name = f"{col_str}_{dl}_Mean"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

                col_data = self.dataframe_slices[dim_col_str].std(axis=1).values
                col_name = f"{col_str}_{dl}_Std"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

                # ACF averages
                dim_col_str_acf = [f"{col_str_acf}_{dl}_slice {isl}" for isl in range(self.no_slices)]
                col_data = self.dataframe_acf_slices[dim_col_str_acf].mean(axis=1).values
                col_name = f"{col_str_acf}_{dl}_Mean"
                self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

                col_data = self.dataframe_acf_slices[dim_col_str_acf].std(axis=1).values
                col_name = f"{col_str_acf}_{dl}_Std"
                self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

            tot_col_str = [f"{col_str_acf}_Total_slice {isl}" for isl in range(self.no_slices)]

            col_data = self.dataframe_acf_slices[tot_col_str].mean(axis=1).values
            col_name = f"{col_str}_Total_Mean"
            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

            col_data = self.dataframe_acf_slices[tot_col_str].std(axis=1).values
            col_name = f"{col_str}_Total_Std"
            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

        # Average and std over the slices
        # Total Current data
        for d in range(self.dimensions):
            dl = self.dim_labels[d]
            dim_col_str = [f"{self.__long_name__}_{dl}_slice {isl}" for isl in range(self.no_slices)]
            dim_col_str_acf = [f"{self.__long_name__} ACF_{dl}_slice {isl}" for isl in range(self.no_slices)]

            self.dataframe[f"{self.__long_name__}_{dl}_Mean"] = self.dataframe_slices[dim_col_str].mean(axis=1)
            self.dataframe[f"{self.__long_name__}_{dl}_Std"] = self.dataframe_slices[dim_col_str].std(axis=1)
            # ACF
            self.dataframe_acf[f"{self.__long_name__}_{dl}_Mean"] = self.dataframe_acf_slices[dim_col_str_acf].mean(
                axis=1
            )
            self.dataframe_acf[f"{self.__long_name__}_{dl}_Std"] = self.dataframe_acf_slices[dim_col_str_acf].std(axis=1)

        # Total ACF
        tot_col_str = [f"{self.__long_name__} ACF_Total_slice {isl}" for isl in range(self.no_slices)]

        col_data = self.dataframe_acf_slices[tot_col_str].mean(axis=1).values
        col_name = self.__long_name__ + f" ACF_Total_Mean"
        self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

        col_data = self.dataframe_acf_slices[tot_col_str].std(axis=1).values
        col_name = self.__long_name__ + f" ACF_Total_Std"
        self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)


class HeatFlux(Observable):
    """Heat Flux."""

    def __init__(self):
        super().__init__()
        self.__name__ = "heat_flux_species_tensor"
        self.__long_name__ = "Heat Flux"
        self.acf_observable = True

    @setup_doc
    def setup(self, params, phase: str = None, no_slices: int = None, **kwargs):

        super().setup_init(params, phase, no_slices)
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):

        # Update the attribute with the passed arguments
        self.__dict__.update(**kwargs)
        self.update_finish()

    @compute_doc
    def compute(self, calculate_acf: bool = False):

        t0 = self.timer.current()
        self.calc_slices_data()
        self.average_slices_data()
        self.save_hdf()
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            self.__long_name__ + " calculation",
            self.timer.time_division(tend - t0),
            self.verbose,
        )
        if calculate_acf:
            self.compute_acf()

    @compute_acf_doc
    def compute_acf(self, equal_number_time_samples: bool = False):

        t0 = self.timer.current()
        self.calc_acf_slices_data(equal_number_time_samples)
        self.average_acf_slices_data()
        self.save_acf_hdf()
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            self.__long_name__ + " ACF calculation",
            self.timer.time_division(tend - t0),
            self.verbose,
        )

    @calc_slices_doc
    def calc_slices_data(self):

        time = zeros(self.slice_steps)

        # Let's compute
        start_slice_step = 0
        end_slice_step = self.slice_steps * self.dump_step
        ### Slices loop
        for isl in tqdm(
            range(self.no_slices),
            desc=f"\nCalculating {self.__long_name__} for slice ",
            disable=not self.verbose,
            position=0,
        ):
            # Parse the particles from the dump files
            heat_flux_species_tensor = zeros((3, self.num_species, self.slice_steps))
            for it, dump in enumerate(range(start_slice_step, end_slice_step, self.dump_step)):
                datap = load_from_restart(self.dump_dir, dump)
                time[it] = datap["time"]
                heat_flux_species_tensor[:, :, it] = datap["species_heat_flux"]

            if isl == 0:
                time_str = f"{self.__long_name__}_Species_Axis_Time"
                self.dataframe[time_str] = time.copy()
                self.dataframe_slices[time_str] = time.copy()

            # Store data into dataframes
            for isp, sp1 in enumerate(self.species_names):
                for iax, ax in enumerate(self.dim_labels):
                    col_data = heat_flux_species_tensor[iax, isp, :]
                    col_name = f"{self.__long_name__}_{sp1}_{ax}_slice {isl}"
                    self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

            start_slice_step += self.slice_steps * self.dump_step
            end_slice_step += self.slice_steps * self.dump_step
            # end of slice loop

    @calc_acf_slices_doc
    def calc_acf_slices_data(self, equal_number_time_samples: bool = False):

        if equal_number_time_samples:
            # In order to have the same number of timesteps products for each time lag of the ACF do the following.
            # Take two slice data
            start_slice_step = 0
            end_slice_step = int(2 * self.acf_slice_steps * self.dump_step)
            time = zeros(self.acf_slice_steps)

            # Temporarily store two consecutive slice data
            heat_flux_species_tensor = zeros((3, self.num_species, 2 * self.acf_slice_steps))

            # Slices loop
            ec_ACF = zeros(self.acf_slice_steps)
            for isl in tqdm(
                range(self.no_slices),
                desc=f"\nCalculating {self.__long_name__} ACF for slice ",
                disable=not self.verbose,
                position=0,
            ):
                # This could be
                for it, dump in enumerate(range(start_slice_step, end_slice_step, self.dump_step)):
                    datap = load_from_restart(self.dump_dir, dump)
                    heat_flux_species_tensor[:, :, it] = datap["species_heat_flux"]
                    time[it] = datap["time"]

                if isl == 0:
                    time_str = f"{self.__long_name__}_Species_Axis_Time"
                    self.dataframe_acf[time_str] = time[: self.acf_slice_steps].copy()
                    self.dataframe_acf_slices[time_str] = time[: self.acf_slice_steps].copy()

                # Calculate the ACF and store into dataframes
                total_heat_flux = zeros((self.no_obs, self.acf_slice_steps))

                # These loops calculate the acf of each pair for each axis. In this order
                # EC_X_11 -> EC_X_12 -> EC_X_22 -> EC_Y_11 -> ... -> EC_Z_11 -> ...
                for iax, ax in enumerate(self.dim_labels):
                    for isp, sp1 in enumerate(self.species_names):
                        for isp2, sp2 in enumerate(self.species_names[isp:], isp):
                            # inter-species multiplier. 1 if both species are equal, 2 if not.
                            const = 1 * (isp == isp2) + 2 * (isp < isp2)

                            # Index of total_ec
                            k = int(self.num_species * isp - (isp - 1) * isp / 2 + (isp2 - isp))

                            # Auto-correlation function with the same number of time steps for each lag.

                            for it in tqdm(
                                range(self.acf_slice_steps),
                                desc=f"{sp1}-{sp2} ACF {ax} time lag",
                                disable=not self.verbose,
                                position=0,
                                leave=False,
                            ):
                                ec_sp1 = heat_flux_species_tensor[iax, isp, : self.acf_slice_steps + it]
                                ec_sp2 = heat_flux_species_tensor[iax, isp2, : self.acf_slice_steps + it]
                                delta_ec_sp1 = ec_sp1 - ec_sp1.mean(axis=-1)
                                delta_ec_sp2 = ec_sp2 - ec_sp2.mean(axis=-1)
                                ec_ACF[it] = correlationfunction(delta_ec_sp1, delta_ec_sp2)[it]
                            else:
                                ec_sp1 = heat_flux_species_tensor[iax, isp, : self.acf_slice_steps]
                                ec_sp2 = heat_flux_species_tensor[iax, isp2, : self.acf_slice_steps]
                                delta_ec_sp1 = ec_sp1 - ec_sp1.mean(axis=-1)
                                delta_ec_sp2 = ec_sp2 - ec_sp2.mean(axis=-1)
                                ec_ACF = correlationfunction(delta_ec_sp1, delta_ec_sp2)
                            # Store in the dataframe
                            col_name = f"{self.__long_name__} ACF_{sp1}-{sp2}_{ax}_slice {isl}"
                            self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, ec_ACF, col_name)
                            # Add to the total ACF of each species pair.
                            total_heat_flux[k] += const * ec_ACF

                # Store the total EC_11, EC_12, EC_13, .. EC_22, EC_23, ...
                for isp, sp1 in enumerate(self.species_names):
                    for isp2, sp2 in enumerate(self.species_names[isp:], isp):
                        # Index of total_ec
                        k = int(self.num_species * isp - (isp - 1) * isp / 2 + (isp2 - isp))

                        col_name = f"{self.__long_name__} ACF_{sp1}-{sp2}_Total_slice {isl}"
                        col_data = total_heat_flux[k, :]

                        self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, col_data, col_name)

                # Sum over the species to get the true EC of the system
                if self.num_species > 1:
                    col_name = f"{self.__long_name__} ACF_all-all_Total_slice {isl}"
                    self.dataframe_acf_slices = add_col_to_df(
                        self.dataframe_acf_slices, total_heat_flux.sum(axis=0) / self.dimensions, col_name
                    )

                # Advance by only one slice at a time.
                start_slice_step += self.acf_slice_steps * self.dump_step
                end_slice_step += self.acf_slice_steps * self.dump_step
                # end of slice loop
        else:
            start_slice_step = 0
            end_slice_step = int(self.slice_steps * self.dump_step)
            time = zeros(self.slice_steps)

            # Temporarily store two consecutive slice data
            heat_flux_species_tensor = zeros((3, self.num_species, self.slice_steps))

            # Slices loop
            ec_ACF = zeros(self.slice_steps)
            for isl in tqdm(
                range(self.no_slices),
                desc=f"\nCalculating {self.__long_name__} ACF for slice ",
                disable=not self.verbose,
                position=0,
            ):
                # This could be
                for it, dump in enumerate(range(start_slice_step, end_slice_step, self.dump_step)):
                    datap = load_from_restart(self.dump_dir, dump)
                    heat_flux_species_tensor[:, :, it] = datap["species_heat_flux"]
                    time[it] = datap["time"]

                if isl == 0:
                    time_str = f"{self.__long_name__}_Species_Axis_Time"
                    self.dataframe_acf[time_str] = time[:].copy()
                    self.dataframe_acf_slices[time_str] = time[:].copy()

                # Calculate the ACF and store into dataframes
                total_heat_flux = zeros((self.no_obs, self.slice_steps))

                # These loops calculate the acf of each pair for each axis. In this order
                # EC_X_11 -> EC_X_12 -> EC_X_22 -> EC_Y_11 -> ... -> EC_Z_11 -> ...
                for iax, ax in enumerate(self.dim_labels):
                    for isp, sp1 in enumerate(self.species_names):
                        for isp2, sp2 in enumerate(self.species_names[isp:], isp):
                            # inter-species multiplier. 1 if both species are equal, 2 if not.
                            const = 1 * (isp == isp2) + 2 * (isp < isp2)

                            # Index of total_ec
                            k = int(self.num_species * isp - (isp - 1) * isp / 2 + (isp2 - isp))

                            # Auto-correlation function with the same number of time steps for each lag.
                            ec_sp1 = heat_flux_species_tensor[iax, isp, :]
                            ec_sp2 = heat_flux_species_tensor[iax, isp2, :]
                            delta_ec_sp1 = ec_sp1 - ec_sp1.mean(axis=-1)
                            delta_ec_sp2 = ec_sp2 - ec_sp2.mean(axis=-1)
                            ec_ACF = correlationfunction(delta_ec_sp1, delta_ec_sp2)

                            # Store in the dataframe
                            col_name = f"{self.__long_name__} ACF_{sp1}-{sp2}_{ax}_slice {isl}"
                            self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, ec_ACF, col_name)
                            # Add to the total ACF of each species pair.
                            total_heat_flux[k] += const * ec_ACF

                # Store the total EC_11, EC_12, EC_13, .. EC_22, EC_23, ...
                for isp, sp1 in enumerate(self.species_names):
                    for isp2, sp2 in enumerate(self.species_names[isp:], isp):
                        # Index of total_ec
                        k = int(self.num_species * isp - (isp - 1) * isp / 2 + (isp2 - isp))

                        col_name = f"{self.__long_name__} ACF_{sp1}-{sp2}_Total_slice {isl}"
                        col_data = total_heat_flux[k, :]

                        self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, col_data, col_name)

                # Sum over the species to get the true EC of the system
                if self.num_species > 1:
                    col_name = f"{self.__long_name__} ACF_all-all_Total_slice {isl}"
                    self.dataframe_acf_slices = add_col_to_df(
                        self.dataframe_acf_slices, total_heat_flux.sum(axis=0) / self.dimensions, col_name
                    )

                # Advance by only one slice at a time.
                start_slice_step += self.slice_steps * self.dump_step
                end_slice_step += self.slice_steps * self.dump_step
                # end of slice loop

    @avg_slices_doc
    def average_slices_data(self):

        for isp, sp1 in enumerate(self.species_names):
            for _, ax in enumerate(self.dim_labels):
                columns = [f"{self.__long_name__}_{sp1}_{ax}_slice {isl}" for isl in range(self.no_slices)]
                col_data = self.dataframe_slices[columns].mean(axis=1)
                col_name = f"{self.__long_name__}_{sp1}_{ax}_Mean"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

                col_data = self.dataframe_slices[columns].std(axis=1)
                col_name = f"{self.__long_name__}_{sp1}_{ax}_Std"
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

    @avg_acf_slices_doc
    def average_acf_slices_data(self):

        # ACF data
        dim_labels = [*self.dim_labels, "Total"]
        if self.num_species > 1:
            species_list = [*self.species_names, "all"]
        else:
            species_list = self.species_names

        for isp, sp1 in enumerate(species_list):
            for isp2, sp2 in enumerate(species_list[isp:], isp):
                for _, ax in enumerate(dim_labels):
                    columns = [f"{self.__long_name__} ACF_{sp1}-{sp2}_{ax}_slice {isl}" for isl in range(self.no_slices)]
                    # Mean
                    col_data = self.dataframe_acf_slices[columns].mean(axis=1)
                    col_name = f"{self.__long_name__} ACF_{sp1}-{sp2}_{ax}_Mean"
                    self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)
                    # Std
                    col_data = self.dataframe_acf_slices[columns].std(axis=1)
                    col_name = f"{self.__long_name__} ACF_{sp1}-{sp2}_{ax}_Std"
                    self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)


class PressureTensor(Observable):
    """Pressure Tensor."""

    def __init__(self):
        super().__init__()
        self.__name__ = "pressure_tensor"
        self.__long_name__ = "Pressure Tensor"
        self.acf_observable = True
        self.kinetic_potential_division = False

    @setup_doc
    def setup(self, params, phase: str = None, no_slices: int = None, **kwargs):

        super().setup_init(params, phase, no_slices)
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):

        # Update the attribute with the passed arguments
        self.__dict__.update(**kwargs)
        self.update_finish()

    def df_column_names(self):

        # Dataframes' columns names
        pt_str_kin = "Pressure Tensor Kinetic"
        pt_str_pot = "Pressure Tensor Potential"
        pt_str = "Pressure Tensor"
        pt_acf_str_kin = "Pressure Tensor Kinetic ACF"
        pt_acf_str_pot = "Pressure Tensor Potential ACF"
        pt_acf_str_kinpot = "Pressure Tensor Kin-Pot ACF"
        pt_acf_str_potkin = "Pressure Tensor Pot-Kin ACF"
        pt_acf_str = "Pressure Tensor ACF"

        # Slice Dataframe
        slice_df_column_names = ["Time"]

        for isl in range(self.no_slices):
            slice_df_column_names.append(f"Pressure_slice {isl}")
            slice_df_column_names.append(f"Delta Pressure_slice {isl}")
            for i, ax1 in enumerate(self.dim_labels):
                for j, ax2 in enumerate(self.dim_labels):
                    slice_df_column_names.append(pt_str_kin + f" {ax1}{ax2}_slice {isl}")
                    slice_df_column_names.append(pt_str_pot + f" {ax1}{ax2}_slice {isl}")
                    slice_df_column_names.append(pt_str + f" {ax1}{ax2}_slice {isl}")

        # ACF Slice Dataframe
        acf_slice_df_column_names = ["Time"]

        for isl in range(self.no_slices):
            acf_slice_df_column_names.append(f"Pressure ACF_slice {isl}")
            acf_slice_df_column_names.append(f"Delta Pressure ACF_slice {isl}")
            for ax1 in self.dim_labels:
                for ax2 in self.dim_labels:
                    for ax3 in self.dim_labels:
                        for ax4 in self.dim_labels:
                            acf_slice_df_column_names.append(pt_acf_str_kin + f" {ax1}{ax2}{ax3}{ax4}_slice {isl}")
                            acf_slice_df_column_names.append(pt_acf_str_pot + f" {ax1}{ax2}{ax3}{ax4}_slice {isl}")
                            acf_slice_df_column_names.append(pt_acf_str_kinpot + f" {ax1}{ax2}{ax3}{ax4}_slice {isl}")
                            acf_slice_df_column_names.append(pt_acf_str_potkin + f" {ax1}{ax2}{ax3}{ax4}_slice {isl}")
                            acf_slice_df_column_names.append(pt_acf_str + f" {ax1}{ax2}{ax3}{ax4}_slice {isl}")
        # Mean and std Dataframe
        df_column_names = ["Time", "Pressure_Mean", "Pressure_Std", "Delta Pressure_Mean", "Delta Pressure_Std"]
        for i, ax1 in enumerate(self.dim_labels):
            for j, ax2 in enumerate(self.dim_labels):
                df_column_names.append(pt_str_kin + f" {ax1}{ax2}_Mean")
                df_column_names.append(pt_str_kin + f" {ax1}{ax2}_Std")
                df_column_names.append(pt_str_pot + f" {ax1}{ax2}_Mean")
                df_column_names.append(pt_str_pot + f" {ax1}{ax2}_Std")
                df_column_names.append(pt_str + f" {ax1}{ax2}_Mean")
                df_column_names.append(pt_str + f" {ax1}{ax2}_Std")

        # Mean and std Dataframe
        acf_df_column_names = [
            "Time",
            "Pressure ACF_Mean",
            "Pressure ACF_Std",
            "Delta Pressure ACF_Mean",
            "Delta Pressure ACF_Std",
        ]
        for ax1 in self.dim_labels:
            for ax2 in self.dim_labels:
                for ax3 in self.dim_labels:
                    for ax4 in self.dim_labels:
                        acf_df_column_names.append(pt_acf_str_kin + f" {ax1}{ax2}{ax3}{ax4}_Mean")
                        acf_df_column_names.append(pt_acf_str_kin + f" {ax1}{ax2}{ax3}{ax4}_Std")
                        acf_df_column_names.append(pt_acf_str_pot + f" {ax1}{ax2}{ax3}{ax4}_Mean")
                        acf_df_column_names.append(pt_acf_str_pot + f" {ax1}{ax2}{ax3}{ax4}_Std")
                        acf_df_column_names.append(pt_acf_str_kinpot + f" {ax1}{ax2}{ax3}{ax4}_Mean")
                        acf_df_column_names.append(pt_acf_str_kinpot + f" {ax1}{ax2}{ax3}{ax4}_Std")
                        acf_df_column_names.append(pt_acf_str_potkin + f" {ax1}{ax2}{ax3}{ax4}_Mean")
                        acf_df_column_names.append(pt_acf_str_potkin + f" {ax1}{ax2}{ax3}{ax4}_Std")
                        acf_df_column_names.append(pt_acf_str + f" {ax1}{ax2}{ax3}{ax4}_Mean")
                        acf_df_column_names.append(pt_acf_str + f" {ax1}{ax2}{ax3}{ax4}_Std")

        self.dataframe = pd.DataFrame(columns=df_column_names)
        self.dataframe_slices = pd.DataFrame(columns=slice_df_column_names)
        self.dataframe_acf = pd.DataFrame(columns=acf_df_column_names)
        self.dataframe_acf_slices = pd.DataFrame(columns=acf_slice_df_column_names)

    @compute_doc
    def compute(self, calculate_acf: bool = False, kin_pot_division: bool = False):

        self.kinetic_potential_division = kin_pot_division
        t0 = self.timer.current()
        self.calc_slices_data()
        self.average_slices_data()
        self.save_hdf()
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            self.__long_name__ + " calculation",
            self.timer.time_division(tend - t0),
            self.verbose,
        )
        if calculate_acf:
            self.compute_acf()

    @compute_acf_doc
    def compute_acf(self, kin_pot_division: bool = False, equal_number_time_samples: bool = False):

        self.kinetic_potential_division = kin_pot_division
        t0 = self.timer.current()
        self.calc_acf_slices_data(equal_number_time_samples)
        self.average_acf_slices_data()
        self.save_acf_hdf()
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            self.__long_name__ + " ACF calculation",
            self.timer.time_division(tend - t0),
            self.verbose,
        )

    @calc_slices_doc
    def calc_slices_data(self):
        # Precompute the number of columns in the dataframe and make the list of names
        # self.df_column_names()

        # Dataframes' columns names
        pt_str_kin = "Pressure Tensor Kinetic"
        pt_str_pot = "Pressure Tensor Potential"
        pt_str = "Pressure Tensor"

        time = zeros(self.slice_steps)

        # Let's compute
        start_slice_step = 0
        end_slice_step = self.slice_steps * self.dump_step
        for isl in tqdm(
            range(self.no_slices),
            desc=f"\nCalculating {self.__long_name__} for slice ",
            disable=not self.verbose,
            position=0,
        ):
            # Parse the particles from the dump files
            pressure = zeros(self.slice_steps)
            pt_kin_temp = zeros((self.dimensions, self.dimensions, self.num_species, self.slice_steps))
            pt_pot_temp = zeros((self.dimensions, self.dimensions, self.num_species, self.slice_steps))
            pt_temp = zeros((self.dimensions, self.dimensions, self.num_species, self.slice_steps))
            species_pressure = zeros((self.num_species, self.slice_steps))
            for it, dump in enumerate(
                tqdm(
                    range(start_slice_step, end_slice_step, self.dump_step),
                    desc="Timestep",
                    position=1,
                    disable=not self.verbose,
                    leave=False,
                )
            ):
                datap = load_from_restart(self.dump_dir, dump)
                time[it] = datap["time"]

                pt_pot_temp[:, :, :, it] = datap["species_pressure_pot_tensor"][:, :, :]
                pt_kin_temp[:, :, :, it] = datap["species_pressure_kin_tensor"][:, :, :]
                pt_temp[:, :, :, it] = pt_pot_temp[:, :, :, it] + pt_kin_temp[:, :, :, it]

                for isp in range(self.num_species):
                    species_pressure[isp, it] = (pt_temp[:, :, isp, it]).trace() / self.dimensions
                # pressure[it], pt_kin_temp[:, :, it], pt_pot_temp[:, :, it], pt_temp[:, :, it] = calc_pressure_tensor(
                #     datap["vel"], datap["virial_species_tensor"], self.species_masses, self.species_num, self.box_volume, self.dimensions
                # )

            if isl == 0:
                time_str = f"Species_Quantity_Time"
                self.dataframe[time_str] = time.copy()
                self.dataframe_slices[time_str] = time.copy()

            # Add total quantities first
            col_data = species_pressure[:, :].sum(axis=0)
            col_name = f"Total_Pressure_slice {isl}"
            self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

            # The reason for dividing these two loops is that I want a specific order in the dataframe.
            # Pressure, Stress Tensor, All
            for i, ax1 in enumerate(self.dim_labels):
                for j, ax2 in enumerate(self.dim_labels):
                    slc_str = f"{ax1}{ax2}_slice {isl}"

                    if self.kinetic_potential_division:
                        col_name = f"Total_{pt_str_kin} {slc_str}"
                        col_data = pt_kin_temp[i, j, :, :].sum(axis=-2)
                        self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

                        col_name = f"Total_{pt_str_pot} {slc_str}"
                        col_data = pt_pot_temp[i, j, :, :].sum(axis=-2)
                        self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

                    # Stress Tensor P_XX, P_XY, P_XZ, P_YX, ...
                    col_name = f"Total_{pt_str} {slc_str}"
                    col_data = pt_temp[i, j, :, :].sum(axis=-2)
                    self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

            # Save data for each species to df
            for isp, sp_name in enumerate(self.species_names):

                col_data = species_pressure[isp, :]
                col_name = f"{sp_name}_Pressure_slice {isl}"
                self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

                # The reason for dividing these two loops is that I want a specific order in the dataframe.
                # Pressure, Stress Tensor, All
                for i, ax1 in enumerate(self.dim_labels):
                    for j, ax2 in enumerate(self.dim_labels):
                        slc_str = f"{ax1}{ax2}_slice {isl}"

                        if self.kinetic_potential_division:
                            col_name = f"{sp_name}_{pt_str_kin} {slc_str}"
                            col_data = pt_kin_temp[i, j, isp, :]
                            self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

                            col_name = f"{sp_name}_{pt_str_pot} {slc_str}"
                            col_data = pt_pot_temp[i, j, isp, :]
                            self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

                        # Stress Tensor P_XX, P_XY, P_XZ, P_YX, ...
                        col_name = f"{sp_name}_{pt_str} {slc_str}"
                        col_data = pt_temp[i, j, isp, :]
                        self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

            start_slice_step += self.slice_steps * self.dump_step
            end_slice_step += self.slice_steps * self.dump_step
            # end of slice loop

    @calc_acf_slices_doc
    def calc_acf_slices_data(self, equal_number_time_samples: bool = False):

        # Dataframes' columns names
        pt_acf_str_kin = "Pressure Tensor Kinetic ACF"
        pt_acf_str_pot = "Pressure Tensor Potential ACF"
        pt_acf_str_kinpot = "Pressure Tensor Kin-Pot ACF"
        pt_acf_str_potkin = "Pressure Tensor Pot-Kin ACF"
        pt_acf_str = "Pressure Tensor ACF"
        # The column names will be:

        # Quantity_Time, Pressure Bulk ACF_slice #,

        # Pressure Tensor ACF XXXX_slice #, Pressure Tensor ACF XXXY_slice #, Pressure Tensor ACF XXXZ_slice #
        # ................................  Pressure Tensor ACF XXYY_slice #, Pressure Tensor ACF XXYZ_slice #
        # ................................  ................................  Pressure Tensor ACF XXZZ_slice #

        # Pressure Tensor ACF XYXX_slice #, Pressure Tensor ACF XYXY_slice #, Pressure Tensor ACF XYXZ_slice #
        # ................................  Pressure Tensor ACF XYYY_slice #, Pressure Tensor ACF XYYZ_slice #
        # ................................  ................................  Pressure Tensor ACF XYZZ_slice #

        # Pressure Tensor ACF XZXX_slice #, Pressure Tensor ACF XZXY_slice #, Pressure Tensor ACF XZXZ_slice #
        # ................................  Pressure Tensor ACF XZYY_slice #, Pressure Tensor ACF XZYZ_slice #
        # ................................  ................................  Pressure Tensor ACF XZZZ_slice #

        # ................................  ................................  ................................
        # ................................  Pressure Tensor ACF YYYY_slice #, Pressure Tensor ACF YYYZ_slice #
        # ................................  ................................  Pressure Tensor ACF YYZZ_slice #

        # ................................  ................................  ................................
        # ................................  Pressure Tensor ACF YZYY_slice #, Pressure Tensor ACF YZYZ_slice #
        # ................................  ................................  Pressure Tensor ACF YZZZ_slice #

        # ................................  ................................  ................................
        # ................................  ................................  ................................
        # ................................  ................................  Pressure Tensor ACF ZZZZ_slice #

        # These are the only calculated axes combination, the .................... indicates that those combination are not calculated.
        if equal_number_time_samples:
            data_storage_size = int(2 * self.acf_slice_steps)
            time = zeros(data_storage_size)

            # Let's compute
            start_slice_step = 0
            end_slice_step = int(2 * self.acf_slice_steps * self.dump_step)
            # Unfortunately I haven't found a better way to grab the data than to re-read it from file.
            for isl in tqdm(
                range(self.no_slices),
                desc=f"\nCalculating {self.__long_name__} ACF for slice ",
                disable=not self.verbose,
                position=0,
            ):
                # Parse the particles from the dump files
                pt_kin_temp = zeros((self.dimensions, self.dimensions, data_storage_size))
                pt_pot_temp = zeros((self.dimensions, self.dimensions, data_storage_size))
                pt_temp = zeros((self.dimensions, self.dimensions, data_storage_size))
                pressure = zeros((data_storage_size))
                for it, dump in enumerate(
                    tqdm(
                        range(start_slice_step, end_slice_step, self.dump_step),
                        desc="Timestep",
                        position=1,
                        disable=not self.verbose,
                        leave=False,
                    )
                ):
                    datap = load_from_restart(self.dump_dir, dump)
                    time[it] = datap["time"]

                    pt_pot_temp[:, :, it] = datap["species_pressure_pot_tensor"][:, :, :].sum(axis=-1)
                    pt_kin_temp[:, :, it] = datap["species_pressure_kin_tensor"][:, :, :].sum(axis=-1)
                    pt_temp[:, :, it] = pt_pot_temp[:, :, it] + pt_kin_temp[:, :, it]
                    pressure[it] = pt_temp[:, :, it].trace()

                if isl == 0:
                    time_str = f"Quantity_Time"
                    self.dataframe_acf[time_str] = time[: self.acf_slice_steps].copy()
                    self.dataframe_acf_slices[time_str] = time[: self.acf_slice_steps].copy()

                # Calculate the ACF for each species to df. I am not calculating the Sp1-Sp2 ACF

                acf_data = zeros(self.acf_slice_steps)

                # Auto-correlation function with the same number of time steps for each lag.
                for it in tqdm(
                    range(self.acf_slice_steps),
                    desc=f"Bulk Pressure Tensor ACF calculation",
                    disable=not self.verbose,
                    position=0,
                    leave=False,
                ):
                    # This is needed for the bulk viscosity
                    col_data = pressure[: self.acf_slice_steps + it]
                    delta_pressure = col_data - col_data.mean()
                    acf_data[it] = correlationfunction(delta_pressure, delta_pressure)[it]

                col_name = f"Pressure Bulk ACF_slice {isl}"
                self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, acf_data, col_name)

                # Calculate the ACF of the thermal fluctuations of the pressure tensor elements
                # Note: C_{abcd} = < sigma_{ab} sigma_{cd} >
                for i, ax1 in enumerate(self.dim_labels):
                    for j, ax2 in enumerate(self.dim_labels[i:], i):
                        for k, ax3 in enumerate(self.dim_labels[i:], i):
                            for l, ax4 in enumerate(self.dim_labels[k:], k):
                                slc_str = f"{ax1}{ax2}{ax3}{ax4}_slice {isl}"

                                if self.kinetic_potential_division:
                                    C_ijkl_kin = correlationfunction(pt_kin_temp[i, j, :], pt_kin_temp[k, l, :])
                                    C_ijkl_pot = correlationfunction(pt_pot_temp[i, j, :], pt_pot_temp[k, l, :])
                                    C_ijkl_kinpot = correlationfunction(pt_kin_temp[i, j, :], pt_pot_temp[k, l, :])
                                    C_ijkl_potkin = correlationfunction(pt_pot_temp[i, j, :], pt_kin_temp[k, l, :])

                                    # Kinetic
                                    col_name = pt_acf_str_kin + slc_str
                                    col_data = C_ijkl_kin
                                    self.dataframe_acf_slices = add_col_to_df(
                                        self.dataframe_acf_slices, col_data, col_name
                                    )

                                    # Potential
                                    col_name = pt_acf_str_pot + slc_str
                                    col_data = C_ijkl_pot
                                    self.dataframe_acf_slices = add_col_to_df(
                                        self.dataframe_acf_slices, col_data, col_name
                                    )

                                    # Kin-Pot
                                    col_name = pt_acf_str_kinpot + slc_str
                                    col_data = C_ijkl_kinpot
                                    self.dataframe_acf_slices = add_col_to_df(
                                        self.dataframe_acf_slices, col_data, col_name
                                    )

                                    # Pot-Kin
                                    col_name = pt_acf_str_potkin + slc_str
                                    col_data = C_ijkl_potkin
                                    self.dataframe_acf_slices = add_col_to_df(
                                        self.dataframe_acf_slices, col_data, col_name
                                    )

                                acf_data = zeros(self.acf_slice_steps)
                                # Auto-correlation function with the same number of time steps for each lag.
                                for it in range(self.acf_slice_steps):
                                    # desc=f"Shear {ax1}{ax2}{ax3}{ax4} ACF calculation",
                                    # disable=not self.verbose,
                                    # position=0,
                                    # leave=False,
                                    # ):
                                    # This is needed for the bulk viscosity
                                    col_data = pt_temp[i, j, : self.acf_slice_steps + it]
                                    acf_data[it] = correlationfunction(col_data, col_data)[it]

                                # Total
                                col_name = f"{pt_acf_str} {slc_str}"
                                self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, acf_data, col_name)

                start_slice_step += self.acf_slice_steps * self.dump_step
                end_slice_step += self.acf_slice_steps * self.dump_step
                # end of slice loop
        else:
            data_storage_size = int(self.slice_steps)
            time = zeros(data_storage_size)

            # Let's compute
            start_slice_step = 0
            end_slice_step = int(self.acf_slice_steps * self.dump_step)
            # Unfortunately I haven't found a better way to grab the data than to re-read it from file.
            for isl in tqdm(
                range(self.no_slices),
                desc=f"\nCalculating {self.__long_name__} for slice ",
                disable=not self.verbose,
                position=0,
            ):
                # Parse the particles from the dump files
                pt_kin_temp = zeros((self.dimensions, self.dimensions, data_storage_size))
                pt_pot_temp = zeros((self.dimensions, self.dimensions, data_storage_size))
                pt_temp = zeros((self.dimensions, self.dimensions, data_storage_size))
                pressure = zeros((data_storage_size))
                for it, dump in enumerate(
                    tqdm(
                        range(start_slice_step, end_slice_step, self.dump_step),
                        desc="Timestep",
                        position=1,
                        disable=not self.verbose,
                        leave=False,
                    )
                ):
                    datap = load_from_restart(self.dump_dir, dump)
                    time[it] = datap["time"]

                    pt_pot_temp[:, :, it] = datap["species_pressure_pot_tensor"][:, :, :].sum(axis=-1)
                    pt_kin_temp[:, :, it] = datap["species_pressure_kin_tensor"][:, :, :].sum(axis=-1)
                    pt_temp[:, :, it] = pt_pot_temp[:, :, it] + pt_kin_temp[:, :, it]
                    pressure[it] = pt_temp[:, :, it].trace()

                if isl == 0:
                    time_str = f"Quantity_Time"
                    self.dataframe_acf[time_str] = time[: self.slice_steps].copy()
                    self.dataframe_acf_slices[time_str] = time[: self.slice_steps].copy()

                # Calculate the ACF for each species to df. I am not calculating the Sp1-Sp2 ACF

                acf_data = zeros(self.slice_steps)

                # This is needed for the bulk viscosity
                col_data = pressure
                delta_pressure = col_data - col_data.mean()
                acf_data = correlationfunction(delta_pressure, delta_pressure)

                col_name = f"Pressure Bulk ACF_slice {isl}"
                self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, acf_data, col_name)

                # Calculate the ACF of the thermal fluctuations of the pressure tensor elements
                # Note: C_{abcd} = < sigma_{ab} sigma_{cd} >
                for i, ax1 in enumerate(self.dim_labels):
                    for j, ax2 in enumerate(self.dim_labels[i:], i):
                        for k, ax3 in enumerate(self.dim_labels[i:], i):
                            for l, ax4 in enumerate(self.dim_labels[k:], k):
                                slc_str = f"{ax1}{ax2}{ax3}{ax4}_slice {isl}"

                                if self.kinetic_potential_division:
                                    C_ijkl_kin = correlationfunction(pt_kin_temp[i, j, :], pt_kin_temp[k, l, :])
                                    C_ijkl_pot = correlationfunction(pt_pot_temp[i, j, :], pt_pot_temp[k, l, :])
                                    C_ijkl_kinpot = correlationfunction(pt_kin_temp[i, j, :], pt_pot_temp[k, l, :])
                                    C_ijkl_potkin = correlationfunction(pt_pot_temp[i, j, :], pt_kin_temp[k, l, :])

                                    # Kinetic
                                    col_name = pt_acf_str_kin + slc_str
                                    col_data = C_ijkl_kin
                                    self.dataframe_acf_slices = add_col_to_df(
                                        self.dataframe_acf_slices, col_data, col_name
                                    )

                                    # Potential
                                    col_name = pt_acf_str_pot + slc_str
                                    col_data = C_ijkl_pot
                                    self.dataframe_acf_slices = add_col_to_df(
                                        self.dataframe_acf_slices, col_data, col_name
                                    )

                                    # Kin-Pot
                                    col_name = pt_acf_str_kinpot + slc_str
                                    col_data = C_ijkl_kinpot
                                    self.dataframe_acf_slices = add_col_to_df(
                                        self.dataframe_acf_slices, col_data, col_name
                                    )

                                    # Pot-Kin
                                    col_name = pt_acf_str_potkin + slc_str
                                    col_data = C_ijkl_potkin
                                    self.dataframe_acf_slices = add_col_to_df(
                                        self.dataframe_acf_slices, col_data, col_name
                                    )

                                # acf_data = zeros(self.acf_slice_steps)
                                col_data = pt_temp[i, j, :]
                                acf_data = correlationfunction(col_data, col_data)

                                # Total
                                col_name = f"{pt_acf_str} {slc_str}"
                                self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, acf_data, col_name)

                start_slice_step += self.slice_steps * self.dump_step
                end_slice_step += self.slice_steps * self.dump_step
                # end of slice loop

    @avg_slices_doc
    def average_slices_data(self):

        # Dataframes' columns names
        pt_str_kin = "Pressure Tensor Kinetic"
        pt_str_pot = "Pressure Tensor Potential"
        pt_str = "Pressure Tensor"

        col_str = [f"Total_Pressure_slice {isl}" for isl in range(self.no_slices)]
        col_name = "Total_Pressure_Mean"
        col_data = self.dataframe_slices[col_str].mean(axis=1).values
        self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
        col_name = "Total_Pressure_Std"
        col_data = self.dataframe_slices[col_str].std(axis=1).values
        self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

        # Save data for each species to df
        for _, sp_name in enumerate(self.species_names):
            col_str = [f"{sp_name}_Pressure_slice {isl}" for isl in range(self.no_slices)]
            col_name = f"{sp_name}_Pressure_Mean"
            col_data = self.dataframe_slices[col_str].mean(axis=1).values
            self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
            # Std
            col_name = f"{sp_name}_Pressure_Std"
            col_data = self.dataframe_slices[col_str].std(axis=1).values
            self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

            for i, ax1 in enumerate(self.dim_labels):
                for j, ax2 in enumerate(self.dim_labels):

                    if self.kinetic_potential_division:
                        # Kinetic Terms
                        ij_col_str = [f"{sp_name}_{pt_str_kin} {ax1}{ax2}_slice {isl}" for isl in range(self.no_slices)]
                        # Mean
                        col_name = f"{sp_name}_{pt_str_kin} {ax1}{ax2}_Mean"
                        col_data = self.dataframe_slices[ij_col_str].mean(axis=1).values
                        self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
                        # Std
                        col_name = f"{sp_name}_{pt_str_kin} {ax1}{ax2}_Std"
                        col_data = self.dataframe_slices[ij_col_str].std(axis=1).values
                        self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

                        # Potential Terms
                        ij_col_str = [f"{sp_name}_{pt_str_pot} {ax1}{ax2}_slice {isl}" for isl in range(self.no_slices)]
                        # Mean
                        col_name = f"{sp_name}_{pt_str_pot} {ax1}{ax2}_Mean"
                        col_data = self.dataframe_slices[ij_col_str].mean(axis=1).values
                        self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
                        # Std
                        col_name = f"{sp_name}_{pt_str_pot} {ax1}{ax2}_Std"
                        col_data = self.dataframe_slices[ij_col_str].std(axis=1).values
                        self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

                    # Full
                    ij_col_str = [f"{sp_name}_{pt_str} {ax1}{ax2}_slice {isl}" for isl in range(self.no_slices)]
                    # Mean
                    col_name = f"{sp_name}_{pt_str} {ax1}{ax2}_Mean"
                    col_data = self.dataframe_slices[ij_col_str].mean(axis=1).values
                    self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)
                    # Std
                    col_name = f"{sp_name}_{pt_str} {ax1}{ax2}_Std"
                    col_data = self.dataframe_slices[ij_col_str].std(axis=1).values
                    self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

    @avg_acf_slices_doc
    def average_acf_slices_data(self):

        pt_acf_str_kin = "Pressure Tensor Kinetic ACF"
        pt_acf_str_pot = "Pressure Tensor Potential ACF"
        pt_acf_str_kinpot = "Pressure Tensor Kin-Pot ACF"
        pt_acf_str_potkin = "Pressure Tensor Pot-Kin ACF"
        pt_acf_str = "Pressure Tensor ACF"
        # The column names will be:
        # Quantity_Time, Pressure Tensor ACF_Mean, Pressure Tensor ACF_Std,

        # Pressure Tensor ACF XXXX_Mean/Std, Pressure Tensor ACF XXXY_Mean/Std, Pressure Tensor ACF XXXZ_Mean/Std
        # ................................   Pressure Tensor ACF XXYY_Mean/Std, Pressure Tensor ACF XXYZ_Mean/Std
        # ................................   ................................   Pressure Tensor ACF XXZZ_Mean/Std

        # Pressure Tensor ACF XYXX_Mean/Std, Pressure Tensor ACF XYXY_Mean/Std, Pressure Tensor ACF XYXZ_Mean/Std
        # ................................   Pressure Tensor ACF XYYY_Mean/Std, Pressure Tensor ACF XYYZ_Mean/Std
        # ................................   ................................   Pressure Tensor ACF XYZZ_Mean/Std

        # Pressure Tensor ACF XZXX_Mean/Std, Pressure Tensor ACF XZXY_Mean/Std, Pressure Tensor ACF XZXZ_Mean/Std
        # ................................   Pressure Tensor ACF XZYY_Mean/Std, Pressure Tensor ACF XZYZ_Mean/Std
        # ................................   ................................   Pressure Tensor ACF XZZZ_Mean/Std

        # ................................   ................................   ................................
        # ................................   Pressure Tensor ACF YYYY_Mean/Std, Pressure Tensor ACF YYYZ_Mean/Std
        # ................................   ................................   Pressure Tensor ACF YYZZ_Mean/Std

        # ................................   ................................   ................................
        # ................................   Pressure Tensor ACF YZYY_Mean/Std, Pressure Tensor ACF YZYZ_Mean/Std
        # ................................   ................................   Pressure Tensor ACF YZZZ_Mean/Std

        # ................................   ................................   ................................
        # ................................   ................................   ................................
        # ................................   ................................   Pressure Tensor ACF ZZZZ_Mean/Std

        # These are the only calculated axes combination, the .................... indicates that those combination are not calculated.

        # Pressure ACF Mean and Std
        col_str = [f"Pressure Bulk ACF_slice {isl}" for isl in range(self.no_slices)]
        col_name = "Pressure Bulk ACF_Mean"
        col_data = self.dataframe_acf_slices[col_str].mean(axis=1).values
        self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)
        col_name = "Pressure Bulk ACF_Std"
        col_data = self.dataframe_acf_slices[col_str].std(axis=1).values
        self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

        # Note: C_{abcd} = < sigma_{ab} sigma_{cd} >
        for i, ax1 in enumerate(self.dim_labels):
            for j, ax2 in enumerate(self.dim_labels[i:], i):
                for k, ax3 in enumerate(self.dim_labels[i:], i):
                    for l, ax4 in enumerate(self.dim_labels[k:], k):

                        if self.kinetic_potential_division:
                            # Kinetic Terms
                            ij_col_acf_str = [
                                pt_acf_str_kin + f" {ax1}{ax2}{ax3}{ax4}_slice {isl}" for isl in range(self.no_slices)
                            ]
                            # Mean
                            col_name = pt_acf_str_kin + f" {ax1}{ax2}{ax3}{ax4}_Mean"
                            col_data = self.dataframe_acf_slices[ij_col_acf_str].mean(axis=1).values
                            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)
                            # Std
                            col_name = pt_acf_str_kin + f" {ax1}{ax2}{ax3}{ax4}_Std"
                            col_data = self.dataframe_acf_slices[ij_col_acf_str].std(axis=1).values
                            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

                            # Potential Terms
                            ij_col_acf_str = [
                                pt_acf_str_pot + f" {ax1}{ax2}{ax3}{ax4}_slice {isl}" for isl in range(self.no_slices)
                            ]
                            # Mean
                            col_name = pt_acf_str_pot + f" {ax1}{ax2}{ax3}{ax4}_Mean"
                            col_data = self.dataframe_acf_slices[ij_col_acf_str].mean(axis=1).values
                            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)
                            # Std
                            col_name = pt_acf_str_pot + f" {ax1}{ax2}{ax3}{ax4}_Std"
                            col_data = self.dataframe_acf_slices[ij_col_acf_str].std(axis=1).values
                            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

                            # Kinetic-Potential Terms
                            ij_col_acf_str = [
                                pt_acf_str_kinpot + f" {ax1}{ax2}{ax3}{ax4}_slice {isl}" for isl in range(self.no_slices)
                            ]
                            # Mean
                            col_name = pt_acf_str_kinpot + f" {ax1}{ax2}{ax3}{ax4}_Mean"
                            col_data = self.dataframe_acf_slices[ij_col_acf_str].mean(axis=1).values
                            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)
                            # Std
                            col_name = pt_acf_str_kinpot + f" {ax1}{ax2}{ax3}{ax4}_Std"
                            col_data = self.dataframe_acf_slices[ij_col_acf_str].std(axis=1).values
                            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

                            # Potential-Kinetic Terms
                            ij_col_acf_str = [
                                pt_acf_str_potkin + f" {ax1}{ax2}{ax3}{ax4}_slice {isl}" for isl in range(self.no_slices)
                            ]
                            # Mean
                            col_name = pt_acf_str_potkin + f" {ax1}{ax2}{ax3}{ax4}_Mean"
                            col_data = self.dataframe_acf_slices[ij_col_acf_str].mean(axis=1).values
                            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)
                            # Std
                            col_name = pt_acf_str_potkin + f" {ax1}{ax2}{ax3}{ax4}_Std"
                            col_data = self.dataframe_acf_slices[ij_col_acf_str].std(axis=1).values
                            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

                        # Full
                        ij_col_acf_str = [
                            f"{pt_acf_str} {ax1}{ax2}{ax3}{ax4}_slice {isl}" for isl in range(self.no_slices)
                        ]
                        # Mean
                        col_name = f"{pt_acf_str} {ax1}{ax2}{ax3}{ax4}_Mean"
                        col_data = self.dataframe_acf_slices[ij_col_acf_str].mean(axis=1).values
                        self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)
                        # Std
                        col_name = f"{pt_acf_str} {ax1}{ax2}{ax3}{ax4}_Std"
                        col_data = self.dataframe_acf_slices[ij_col_acf_str].std(axis=1).values
                        self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

    def sum_rule(self, beta, rdf, potential):
        r"""
        Calculate the sum rule integrals from the rdf.

        .. math::
            :nowrap:

            \begin{eqnarray}
                \sigma_{zzzz} & = &   \frac{n}{\beta^2} \left [ 3 +
                \frac{2\beta}{15} I^{(1)} + \frac{\beta}{5} I^{(2)} \right ] , \\
                \sigma_{zzxx} & =& \frac{n}{\beta^2} \left [ 1
                - \frac{2\beta}{5} I^{(1)} + \frac {\beta}{15} I^{(2)} \right ] , \\
                \sigma_{xyxy} & = & \frac{n}{\beta^2} \left [ 1 +
                \frac{4\beta}{15} I^{(2)} + \frac {\beta}{15} I^{(2)} \right ] ,
            \end{eqnarray}

        where :math:`I^{(k)} = \sum_{A} \sum_{B \geq A}I_{AB}^{(\rm {Hartree}, k)} + I_{AB}^{(\rm {Corr}, k)}`
        calculated from
        :meth:`sarkas.tools.observables.RadialDistributionFunction.compute_sum_rule_integrals`.

        Parameters
        ----------
        beta: float
            Inverse temperature factor. Grab it from :py:attr:`sarkas.tools.observables.Thermodynamics.beta`.

        rdf: sarkas.tools.observables.RadialDistributionFunction
            Radial Distribution function object.

        potential : :class:`sarkas.potentials.core.Potential`
            Potential object.

        Returns
        -------
        sigma_zzzz : float
            See above integral.

        sigma_zzxx : float
            See above integral.

        sigma_xyxy : float
            See above integral.
        """

        hartrees, corrs = rdf.compute_sum_rule_integrals(potential)
        # Eq. 2.3.43 -44 in Boon and Yip
        I_1 = hartrees[:, 1].sum() + corrs[:, 1].sum()
        I_2 = hartrees[:, 2].sum() + corrs[:, 2].sum()
        nkT = self.total_num_density / beta

        sigma_zzzz = 3.0 * nkT + 2.0 / 15.0 * I_1 + I_2 / 5.0
        sigma_zzxx = nkT - 2.0 / 5.0 * I_1 + I_2 / 15.0
        sigma_xyxy = nkT + 4.0 / 15.0 * I_1 + I_2 / 15.0

        return sigma_zzzz, sigma_zzxx, sigma_xyxy


class RadialDistributionFunction(Observable):
    """
    Radial Distribution Function.

    Attributes
    ----------
    no_bins : int
        Number of bins.

    dr_rdf : float
        Size of each bin.

    """

    def __init__(self):
        super().__init__()
        self.__name__ = "rdf"
        self.__long_name__ = "Radial Distribution Function"

    @setup_doc
    def setup(self, params, phase: str = None, no_slices: int = None, **kwargs):

        super().setup_init(params, phase=phase, no_slices=no_slices)
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

        # These definitions are needed for the print out.
        self.rc = self.cutoff_radius
        self.no_bins = self.rdf_nbins
        self.dr_rdf = self.rc / self.no_bins

        self.update_finish()

    @compute_doc
    def compute(self):

        t0 = self.timer.current()
        self.calc_slices_data()
        self.average_slices_data()
        self.save_hdf()
        self.save_pickle()
        tend = self.timer.current()
        time_stamp(self.log_file, self.__long_name__ + " Calculation", self.timer.time_division(tend - t0), self.verbose)

    def calc_slices_data(self):

        # initialize temporary arrays
        r_values = zeros(self.no_bins)
        bin_vol = zeros(self.no_bins)
        pair_density = zeros((self.num_species, self.num_species))

        # This is needed to be certain the number of bins is the same.
        # if not isinstance(rdf_hist, ndarray):
        #     # Find the last dump by looking for the largest number in the checkpoints filenames
        #     dumps_list = listdir(self.dump_dir)
        #     dumps_list.sort(key=num_sort)
        #     name, ext = os.path.splitext(dumps_list[-1])
        #     _, number = name.split('_')

        datap = load_from_restart(self.dump_dir, 0)

        # Make sure you are getting the right number of bins and redefine dr_rdf.
        self.no_bins = datap["rdf_hist"].shape[0]
        self.dr_rdf = self.rc / self.no_bins

        # No. of pairs per volume
        for i, sp1 in enumerate(self.species_num):
            pair_density[i, i] = sp1 * (sp1 - 1) / self.box_volume
            if self.num_species > 1:
                for j, sp2 in enumerate(self.species_num[i + 1 :], i + 1):
                    pair_density[i, j] = sp1 * sp2 / self.box_volume

        # Calculate the volume of each bin
        # The formula for the N-dimensional sphere is
        # pi^{N/2}/( factorial( N/2) )
        # from https://en.wikipedia.org/wiki/N-sphere#:~:text=In%20general%2C%20the-,volume,-%2C%20in%20n%2Ddimensional
        sphere_shell_const = (pi ** (self.dimensions / 2.0)) / factorial(self.dimensions / 2.0)
        bin_vol[0] = sphere_shell_const * self.dr_rdf**self.dimensions
        for ir in range(1, self.no_bins):
            r1 = ir * self.dr_rdf
            r2 = (ir + 1) * self.dr_rdf
            bin_vol[ir] = sphere_shell_const * (r2**self.dimensions - r1**self.dimensions)
            r_values[ir] = (ir + 0.5) * self.dr_rdf

        # Save the ra values for simplicity
        self.ra_values = r_values / self.a_ws

        self.dataframe["Interparticle_Distance"] = r_values
        self.dataframe_slices["Interparticle_Distance"] = r_values

        dump_init = 0

        for isl in tqdm(range(self.no_slices), desc="Calculating RDF for slice", disable=not self.verbose):

            # Grab the data from the dumps. The -1 is for '0'-indexing
            dump_end = (isl + 1) * (self.slice_steps - 1) * self.dump_step

            data_init = load_from_restart(self.dump_dir, int(dump_init))
            data_end = load_from_restart(self.dump_dir, int(dump_end))
            for i, sp1 in enumerate(self.species_names):
                for j, sp2 in enumerate(self.species_names[i:], i):
                    denom_const = pair_density[i, j] * self.slice_steps * self.dump_step
                    # Each slice should be considered as an independent system.
                    # The RDF is calculated from the difference between the last dump of the slice and the initial dump
                    # of the slice
                    rdf_hist_init = data_init["rdf_hist"][:, i, j] + data_init["rdf_hist"][:, j, i]
                    rdf_hist_end = data_end["rdf_hist"][:, i, j] + data_end["rdf_hist"][:, j, i]
                    rdf_hist_slc = rdf_hist_end - rdf_hist_init

                    col_name = f"{sp1}-{sp2} RDF_slice {isl}"
                    col_data = rdf_hist_slc / denom_const / bin_vol
                    self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)

            dump_init = dump_end

    @avg_slices_doc
    def average_slices_data(self):

        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:], i):
                col_str = [f"{sp1}-{sp2} RDF_slice {isl}" for isl in range(self.no_slices)]

                col_name = f"{sp1}-{sp2} RDF_Mean"
                col_data = self.dataframe_slices[col_str].mean(axis=1).values
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

                col_name = f"{sp1}-{sp2} RDF_Std"
                col_data = self.dataframe_slices[col_str].std(axis=1).values
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

    def compute_sum_rule_integrals(self, potential):
        """
        Compute integrals of the RDF used in sum rules. \n

        The species dependent integrals are

        .. math::

            I_{AB}^{\\rm (Hartree, k)} = 2^{D - 2} \\pi  n_{A} n_{B} \\int_0^{\\infty} dr \\,
            r^{D - 1 + k} \\frac{d^k}{dr^k} \\phi_{AB}(r),

        .. math::

            I_{AB}^{\\rm (Corr, k)} = 2^{D - 2} \\pi  n_{A} n_{B} \\int_0^{\\infty} dr \\,
            r^{D - 1 + k} h_{AB} (r) \\frac{d^k}{dr^k} \\phi_{AB}(r),

        where :math:`D` is the number of dimensions, :math:`k = {0, 1, 2}`,
        and :math:`\\phi_{AB}(r)` is the potential between species :math:`A` and :math:`B`. \n
        Only Coulomb and Yukawa potentials are supported at the moment.

        Parameters
        ----------
        potential : :class:`sarkas.potentials.core.Potential`
            Sarkas Potential object. Needed for all its attributes.

        Returns
        -------
        hartrees : numpy.ndarray
            Hartree integrals with :math:`k = {0, 1, 2}`. \n
            Shape = ( :py:attr:`sarkas.tools.observables.Observable.no_obs`, 3).

        corrs : numpy.ndarray
            Correlational integrals with :math:`k = {0, 1, 2}`. \n
            Shape = ( :py:attr:`sarkas.tools.observables.Observable.no_obs`, 3).

        """

        r = self.dataframe[self.dataframe.columns[0]].to_numpy().copy()

        dims = self.dimensions
        dim_const = 2.0 ** (dims - 2) * pi

        if r[0] == 0.0:
            r[0] = 1e-40

        corrs = zeros((self.no_obs, 3))
        hartrees = zeros((self.no_obs, 3))

        obs_indx = 0
        # TODO:Make this calculation for each slice and/or run
        for sp1, sp1_name in enumerate(self.species_names):
            for sp2, sp2_name in enumerate(self.species_names[sp1:], sp1):
                h_r = self.dataframe[(f"{sp1_name}-{sp2_name} RDF", "Mean")].to_numpy() - 1.0

                # Calculate the derivatives of the potential
                u_r, dv_dr, d2v_dr2 = potential.potential_derivatives(r, potential.matrix[sp1, sp2])

                densities = self.species_num_dens[sp1] * self.species_num_dens[sp2]

                hartrees[obs_indx, 0] = dim_const * densities * trapz(u_r * r ** (dims - 1), x=r)
                corrs[obs_indx, 0] = dim_const * densities * trapz(u_r * h_r * r ** (dims - 1), x=r)

                hartrees[obs_indx, 1] = dim_const * densities * trapz(dv_dr * r**dims, x=r)
                corrs[obs_indx, 1] = dim_const * densities * trapz(dv_dr * h_r * r**dims, x=r)

                hartrees[obs_indx, 2] = dim_const * densities * trapz(d2v_dr2 * r ** (dims + 1), x=r)
                corrs[obs_indx, 2] = dim_const * densities * trapz(d2v_dr2 * h_r * r ** (dims + 1), x=r)

                obs_indx += 1

        return hartrees, corrs


class StaticStructureFactor(Observable):
    """
    Static Structure Factors.

    The species dependent SSF :math:`S_{AB}(\\mathbf k)` is calculated from

    .. math::
        S_{AB}(\\mathbf k) = \\int_0^\\infty dt \\,
        \\left \\langle | n_{A}( \\mathbf k, t)n_{B}( -\\mathbf k, t) \\right \\rangle,

    where the microscopic density of species :math:`A` with number of particles :math:`N_{A}` is given by

    .. math::
        n_{A}(\\mathbf k,t) = \\sum^{N_{A}}_{j} e^{-i \\mathbf k \\cdot \\mathbf r_j(t)} .


    Attributes
    ----------
    k_list : list
        List of all possible :math:`k` vectors with their corresponding magnitudes and indexes.

    k_counts : numpy.ndarray
        Number of occurrences of each :math:`k` magnitude.

    ka_values : numpy.ndarray
        Magnitude of each allowed :math:`ka` vector.

    no_ka_values: int
        Length of ``ka_values`` array.

    """

    def __init__(self):
        super().__init__()
        self.__name__ = "ssf"
        self.__long_name__ = "Static Structure Function"
        self.k_observable = True
        self.kw_observable = False

    @setup_doc
    def setup(self, params, phase: str = None, no_slices: int = None, **kwargs):

        super().setup_init(params, phase=phase, no_slices=no_slices)
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):
        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())
        self.update_finish()

    @compute_doc
    def compute(self):

        tinit = self.timer.current()
        self.calc_slices_data()
        self.average_slices_data()
        self.save_hdf()
        tend = self.timer.current()
        time_stamp(
            self.log_file, self.__long_name__ + " Calculation", self.timer.time_division(tend - tinit), self.verbose
        )

    @calc_slices_doc
    def calc_slices_data(self):

        self.parse_kt_data(nkt_flag=True)
        no_dumps_calculated = self.slice_steps * self.no_slices
        Sk_avg = zeros((self.no_obs, len(self.k_counts), no_dumps_calculated))

        k_column = "Wavenumber_k"
        self.dataframe_slices[k_column] = self.k_values
        self.dataframe[k_column] = self.k_values

        nkt_df = read_hdf(self.nkt_hdf_file, mode="r", key="nkt")
        for isl in tqdm(range(self.no_slices), desc="Calculating SSF for slice", disable=not self.verbose):
            # Initialize container
            nkt = zeros((self.num_species, self.slice_steps, len(self.k_list)), dtype=complex128)
            for sp, sp_name in enumerate(self.species_names):
                nkt[sp] = array(nkt_df[f"slice {isl + 1}"][sp_name])

            init = isl * self.slice_steps
            fin = (isl + 1) * self.slice_steps
            Sk_avg[:, :, init:fin] = calc_Sk(nkt, self.k_list, self.k_counts, self.species_num, self.slice_steps)
            sp_indx = 0
            for i, sp1 in enumerate(self.species_names):
                for j, sp2 in enumerate(self.species_names[i:]):
                    col_name = f"{sp1}-{sp2} SSF_slice {isl}"
                    col_data = Sk_avg[sp_indx, :, init:fin].mean(axis=-1)
                    self.dataframe_slices = add_col_to_df(self.dataframe_slices, col_data, col_name)
                    sp_indx += 1

    @avg_slices_doc
    def average_slices_data(self):

        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:]):
                column = [f"{sp1}-{sp2} SSF_slice {isl}" for isl in range(self.no_slices)]
                col_name = f"{sp1}-{sp2} SSF_Mean"
                col_data = self.dataframe_slices[column].mean(axis=1).values
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

                col_name = f"{sp1}-{sp2} SSF_Std"
                col_data = self.dataframe_slices[column].std(axis=1).values
                self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)


class Thermodynamics(Observable):
    """
    Thermodynamic functions.
    """

    def __init__(self):
        super().__init__()
        self.__name__ = "therm"
        self.__long_name__ = "Thermodynamics"
        self.beta_slice = None
        self.specific_heat_volume_slice = None
        self.acf_observable = True

    def setup(self, params, phase: str = None, no_slices: int = None, **kwargs):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        params : sarkas.core.Parameters
            Simulation's parameters.

        phase : str, optional
            Phase to compute. Default = 'production'.

        **kwargs :
            These will overwrite any :class:`sarkas.core.Parameters`
            or default :class:`sarkas.tools.observables.Observable`
            attributes and/or add new ones.

        """

        super().setup_init(params, phase, no_slices)
        if params.load_method[:-7] == "restart":
            self.restart_sim = True
        else:
            self.restart_sim = False

        self.update_args(**kwargs)
        self.grab_sim_data(phase)

    @arg_update_doc
    def update_args(self, **kwargs):

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())
        self.update_finish()

    def calculate_beta_slices(self, ensemble: str = "NVE"):
        """Calculate the inverse temperature by taking the mean of the temperature time series."""
        if ensemble == "NVE":
            self.beta_slice = zeros(self.no_slices)
            for isl in range(self.no_slices):
                self.beta_slice[isl] = 1.0 / (self.kB * self.dataframe_slices[("Temperature", f"slice {isl}")].mean())
        else:
            self.beta_slice = ones(self.no_slices) * 1.0 / (self.kB * self.T_desired)

    def calculate_heat_capacity_slices(self, ensemble: str = "NVE"):
        """Calculate the specific heat capacity from the fluctuations of the energy."""

        self.calculate_beta_slices(ensemble=ensemble)

        if ensemble == "NVE":
            self.specific_heat_volume_slice = zeros(self.no_slices)
            for isl in range(self.no_slices):
                kin_2 = (self.dataframe_slices[("Total Kinetic Energy", f"slice {isl}")].std()) ** 2
                denom = 1 - 2.0 * self.beta_slice[isl] ** 2 * kin_2 / (self.dimensions * self.total_num_ptcls)
                self.specific_heat_volume_slice[isl] = 0.5 * self.dimensions * self.kB * self.total_num_ptcls / denom
        else:
            self.specific_heat_volume_slice = zeros(self.no_slices)
            for isl in range(self.no_slices):
                deltaE_2 = (self.dataframe_slices[("Total Energy", f"slice {isl}")].std()) ** 2
                self.specific_heat_volume_slice[isl] = deltaE_2 * self.beta_slice[isl] ** 2 * self.kB

    def calculate_beta_simulation(self, ensemble: str = "NVE"):
        """Calculate the inverse temperature by taking the mean of the temperature time series."""
        if ensemble == "NVE":
            self.beta = 1.0 / (self.kB * self.simulation_dataframe["Temperature"].mean())
        else:
            self.beta = 1.0 / (self.kB * self.T_desired)

    def calculate_heat_capacity_simulation(self, ensemble: str = "NVE"):
        """Calculate the specific heat capacity from the fluctuations of the energy."""

        self.calculate_beta_simulation(ensemble=ensemble)

        if ensemble == "NVE":
            kin_2 = (self.simulation_dataframe["Total Kinetic Energy"].std()) ** 2
            denom = 1 - 2.0 * self.beta**2 * kin_2 / (self.dimensions * self.total_num_ptcls)
            self.specific_heat_volume = 0.5 * self.dimensions * self.kB * self.total_num_ptcls / denom
        else:
            deltaE_2 = (self.simulation_dataframe["Total Energy"].std()) ** 2
            self.specific_heat_volume = deltaE_2 * self.beta**2 * self.kB

    @calc_slices_doc
    def calc_slices_data(self):

        # data_size
        data_size = int(self.slice_steps)

        ### Slices loop
        for col_name, col_data in self.simulation_dataframe.iloc[:, 1:].items():
            data_start = 0
            data_end = data_size
            for isl in range(self.no_slices):

                if isl == 0:
                    time = self.simulation_dataframe["Time"][: self.slice_steps]
                    time_str = f"Quantity_Time"
                    self.dataframe[time_str] = time.copy()
                    self.dataframe_slices[time_str] = time.copy()

                df_col_name = f"{col_name}_slice {isl}"
                self.dataframe_slices = add_col_to_df(
                    self.dataframe_slices, col_data[data_start:data_end].values, df_col_name
                )

                data_start += self.slice_steps
                data_end += self.slice_steps
            # end of slice loop

    @calc_acf_slices_doc
    def calc_acf_slice_data(self, equal_number_time_samples: bool = False):

        # In order to have the same number of timesteps products for each time lag of the ACF do the following.
        # Temporarily store two consecutive slice data
        data_col = zeros(2 * self.acf_slice_steps)

        # data_size
        data_size = int(2 * self.acf_slice_steps)

        for col_name, col_data in self.simulation_dataframe.iloc[:, 1:].items():

            data_start = 0
            data_end = data_size
            data_acf = zeros(self.acf_slice_steps)

            # Slices loop
            for isl in tqdm(
                range(self.no_slices),
                desc=f"\nCalculating {col_name} ACF for slice ",
                disable=not self.verbose,
                position=0,
            ):
                if isl == 0:
                    time = self.simulation_dataframe["Time"][: self.acf_slice_steps]
                    time_str = f"Quantity_Time"
                    self.dataframe_acf[time_str] = time[: self.acf_slice_steps].copy()
                    self.dataframe_acf_slices[time_str] = time[: self.acf_slice_steps].copy()

                data_col = col_data[data_start:data_end].values

                if equal_number_time_samples:
                    # Auto-correlation function with the same number of time steps for each lag.
                    for it in tqdm(
                        range(self.acf_slice_steps),
                        desc=f"{col_name} ACF time lag",
                        disable=not self.verbose,
                        position=0,
                        leave=False,
                    ):
                        delta_data = data_col[:it + self.acf_slice_steps] - data_col[:it + self.acf_slice_steps].mean()
                        data_acf[it] = correlationfunction(delta_data, delta_data)[it]
                else:
                    delta_data = data_col - data_col.mean()
                    data_acf = correlationfunction(delta_data, delta_data)

                # Store in the dataframe
                col_name = f"{col_name} ACF_slice {isl}"
                self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, data_acf, col_name)

                # Advance by only one slice at a time.
                data_start += self.acf_slice_steps
                data_end += self.acf_slice_steps
            # end of slice loop

    @avg_slices_doc
    def average_slices_data(self):

        ### Slices loop
        for _, df_col_name in enumerate(self.simulation_dataframe.columns[1:]):

            columns = [f"{df_col_name}_slice {isl}" for isl in range(self.no_slices)]
            col_data = self.dataframe_slices[columns].mean(axis=1)
            col_name = f"{df_col_name}_Mean"
            self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

            col_data = self.dataframe_slices[columns].std(axis=1)
            col_name = f"{df_col_name}_Std"
            self.dataframe = add_col_to_df(self.dataframe, col_data, col_name)

    @avg_acf_slices_doc
    def average_acf_slices_data(self):

        ### Slices loop
        for _, df_col_name in enumerate(self.simulation_dataframe.columns[1:]):

            columns = [f"{df_col_name} ACF_slice {isl}" for isl in range(self.no_slices)]
            col_data = self.dataframe_acf_slices[columns].mean(axis=1)
            col_name = f"{df_col_name} ACF_Mean"
            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

            col_data = self.dataframe_acf_slices[columns].std(axis=1)
            col_name = f"{df_col_name} ACF_Std"
            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

    @compute_doc
    def compute(self, calculate_acf: bool = False):

        t0 = self.timer.current()
        self.calc_slices_data()
        self.average_slices_data()
        self.save_hdf()
        self.calculate_beta_slices()
        self.calculate_heat_capacity_slices()

        msg = self.__long_name__
        if calculate_acf:
            msg = (self.__long_name__ + " and its ACF Calculation",)
            self.calc_acf_slice_data()
            self.average_acf_slices_data()
            self.save_acf_hdf()

        tend = self.timer.current()
        time_stamp(
            self.log_file,
            msg,
            self.timer.time_division(tend - t0),
            self.verbose,
        )

    def compute_from_rdf(self, rdf, potential):
        """
        Calculate the correlational energy and correlation pressure using
        :meth:`sarkas.tools.observables.RadialDistributionFunction.compute_sum_rule_integrals` method.

        The Hartree and correlational terms between species :math:`A` and :math:`B` are

        .. math::

            U_{AB}^{\\rm hartree} =  2 \\pi \\frac{N_AN_B}{V} \\int_0^\\infty dr \\, \\phi_{AB}(r) r^2 dr,

        .. math::

            U_{AB}^{\\rm corr} =  2 \\pi \\frac{N_AN_B}{V} \\int_0^\\infty dr \\, \\phi_{AB}(r) h(r) r^2 dr,

        .. math::

            P_{AB}^{\\rm hartree} =  - \\frac{2 \\pi}{3} \\frac{N_AN_B}{V^2} \\int_0^\\infty dr \\, \\frac{d\\phi_{AB}(r)}{dr} r^3 dr,

        .. math::

            P_{AB}^{\\rm corr} =  - \\frac{2 \\pi}{3} \\frac{N_AN_B}{V^2} \\int_0^\\infty dr \\, \\frac{d\\phi_{AB}(r)}{dr} h(r) r^3 dr,


        Parameters
        ----------
        rdf: :class:`sarkas.tools.observables.RadialDistributionFunction`
            Radial Distribution Function object.

        potential : :class:`sarkas.potentials.core.Potential`
            Potential object.

        Returns
        -------
        nkT : float
            Ideal term of the pressure :math:`nk_BT`. Where :math:`n` is the total density.

        u_hartree : numpy.ndarray
            Hartree energy calculated from the above formula for each :math:`g_{ab}(r)`.

        u_corr : numpy.ndarray
            Correlational energy calculated from the above formula for each :math:`g_{ab}(r)`.

        p_hartree : numpy.ndarray
            Hartree pressure calculated from the above formula for each :math:`g_{ab}(r)`.

        p_corr : numpy.ndarray
            Correlational pressure calculated from the above formula for each :math:`g_{ab}(r)`.

        """

        hartrees, corrs = rdf.compute_sum_rule_integrals(potential)

        u_hartree = self.box_volume * hartrees[:, 0]
        u_corr = self.box_volume * corrs[:, 0]

        p_hartree = -hartrees[:, 1] / 3.0
        p_corr = -corrs[:, 1] / 3.0

        nkT = self.total_num_density / self.beta_slice.mean()

        return nkT, u_hartree, u_corr, p_hartree, p_corr

    def grab_sim_data(self, phase=None):
        """
        Grab the pandas dataframe from the saved csv file.
        """
        if phase:
            self.phase = phase.lower()

        self.simulation_dataframe = read_csv(self.filenames_tree["thermodynamics"][self.phase]["path"], index_col=False)
        self.fldr = self.directory_tree["simulation"][self.phase]["path"]

    def temp_energy_plot(
        self,
        process,
        info_list: list = None,
        phase: str = None,
        show: bool = False,
        publication: bool = False,
        figname: str = None,
    ):
        """
        Plot Temperature and Energy as a function of time with their cumulative sum and average.

        Parameters
        ----------
        process : sarkas.processes.Process
            Sarkas Process.

        info_list: list, optional
            List of strings to print next to the plots.

        phase: str, optional
            Phase to plot. "equilibration" or "production".

        show: bool, optional
            Flag for displaying the figure.

        publication: bool, optional
            Flag for publication style plotting.

        figname: str, optional
            Name with which to save the plot.

        """

        if phase:
            phase = phase.lower()
            self.phase = phase
            if self.phase == "equilibration":
                self.no_dumps = self.eq_no_dumps
                self.dump_dir = self.eq_dump_dir
                self.dump_step = self.eq_dump_step
                # self.saving_dir = self.equilibration_dir
                self.no_steps = self.equilibration_steps
                self.grab_sim_data(self.phase)
                # self.simulation_dataframe = self.simulation_dataframe.iloc[1:, :]

            elif self.phase == "production":
                self.no_dumps = self.prod_no_dumps
                self.dump_dir = self.prod_dump_dir
                self.dump_step = self.prod_dump_step
                # self.saving_dir = self.production_dir
                self.no_steps = self.production_steps
                self.grab_sim_data(self.phase)

            elif self.phase == "magnetization":
                self.no_dumps = self.mag_no_dumps
                self.dump_dir = self.mag_dump_dir
                self.dump_step = self.mag_dump_step
                # self.saving_dir = self.magnetization_dir
                self.no_steps = self.magnetization_steps
                self.grab_sim_data(self.phase)

        else:
            self.grab_sim_data()

        if self.phase == "production":
            self.calculate_beta_simulation(ensemble="NVE")
        else:
            self.calculate_beta_simulation(ensemble="NVT")

        completed_steps = self.dump_step * (self.no_dumps - 1)
        fig = plt.figure(figsize=(20, 8))
        fsz = 16
        if publication:

            plt.style.use("PUBstyle")
            gs = GridSpec(3, 7)

            # Temperature plots
            T_delta_plot = fig.add_subplot(gs[0, 0:2])
            T_main_plot = fig.add_subplot(gs[1:3, 0:2])
            T_hist_plot = fig.add_subplot(gs[1:3, 2])
            # Energy plots
            E_delta_plot = fig.add_subplot(gs[0, 4:6])
            E_main_plot = fig.add_subplot(gs[1:3, 4:6])
            E_hist_plot = fig.add_subplot(gs[1:3, 6])

        else:

            gs = GridSpec(3, 8)

            Info_plot = fig.add_subplot(gs[0:4, 0:2])
            # Temperature plots
            T_delta_plot = fig.add_subplot(gs[0, 2:4])
            T_main_plot = fig.add_subplot(gs[1:4, 2:4])
            T_hist_plot = fig.add_subplot(gs[1:4, 4])
            # Energy plots
            E_delta_plot = fig.add_subplot(gs[0, 5:7])
            E_main_plot = fig.add_subplot(gs[1:4, 5:7])
            E_hist_plot = fig.add_subplot(gs[1:4, 7])

        # Grab the current rcParams so that I can restore it later
        current_rcParams = plt.rcParams.copy()
        # Update rcParams with the necessary values
        plt.rc("font", size=fsz)  # controls default text sizes
        plt.rc("axes", titlesize=fsz)  # fontsize of the axes title
        plt.rc("axes", labelsize=fsz)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=fsz - 2)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=fsz - 2)  # fontsize of the tick labels

        # Grab the color line list from the plt cycler. I will use this in the hist plots
        color_from_cycler = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # ------------------------------------------- Temperature -------------------------------------------#
        # Calculate Temperature plot's labels and multipliers
        time_mul, temp_mul, time_prefix, temp_prefix, time_lbl, temp_lbl = plot_labels(
            self.simulation_dataframe["Time"], self.simulation_dataframe["Temperature"], "Time", "Temperature", self.units
        )
        # Rescale quantities
        time = time_mul * self.simulation_dataframe["Time"]
        Temperature = temp_mul * self.simulation_dataframe["Temperature"]
        T_desired = temp_mul * self.T_desired

        # Temperature moving average
        T_cumavg = Temperature.expanding().mean()

        # Temperature deviation and its moving average
        Delta_T = (Temperature - T_desired) * 100 / T_desired
        Delta_T_cum_avg = Delta_T.expanding().mean()

        # Temperature Main plot
        T_main_plot.plot(time, Temperature, alpha=0.7)
        T_main_plot.plot(time, T_cumavg, label="Moving Average")
        T_main_plot.axhline(T_desired, ls="--", c="r", alpha=0.7, label="Desired T")
        T_main_plot.legend(loc="best")
        T_main_plot.set(ylabel="Temperature" + temp_lbl, xlabel="Time" + time_lbl)

        # Temperature Deviation plot
        T_delta_plot.plot(time, Delta_T, alpha=0.5)
        T_delta_plot.plot(time, Delta_T_cum_avg, alpha=0.8)
        T_delta_plot.set(xticks=[], ylabel=r"Deviation [%]")

        if phase == "production":
            # The Temperature fluctuations in an NVE ensemble are
            # < delta T^2> = T_desired^2 * ( 2 /(Np * Dims) ) *( 1 -  Np * Dims/2 * k_B/Cv)
            # where Cv is the heat capacity at constant volume.
            self.calculate_heat_capacity_simulation(ensemble="NVE")
            dN = self.total_num_ptcls * self.dimensions
            factor = 1 - 0.5 * dN * self.kB / self.specific_heat_volume
            T_std = T_desired * sqrt(2.0 / dN * factor)
            # TODO: Review this calculation.
            # T_std *= sqrt(1 - 0.5 *process.parameters.dimensions*process.parameters.total_num_ptcls/heat_capacity_v)
        else:
            # The Temperature fluctuations in an NVT ensemble are
            # < delta T^2> = T_desired^2 *( 2 /(Np * Dims))
            dN = self.total_num_ptcls * self.dimensions
            T_std = T_desired * sqrt(2.0 / dN)

        # Calculate the theoretical distribution of the temperature.
        # T_dist_desired is a Gaussian centered at T_desired with theoretical T_std
        T_dist_desired = scp_stats.norm(loc=T_desired, scale=T_std)
        # T_dist_actual is a Gaussian centered at the actual mean with theoretical T_std
        T_dist_actual = scp_stats.norm(loc=Temperature.mean(), scale=T_std)
        # Histogram plot
        sns_histplot(y=Temperature, bins="fd", stat="density", alpha=0.75, legend="False", ax=T_hist_plot)
        T_hist_plot.set(ylabel=None, xlabel=None, xticks=[], yticks=[], ylim=T_main_plot.get_ylim())
        T_hist_plot.plot(
            T_dist_desired.pdf(Temperature.sort_values()), Temperature.sort_values(), ls="--", color="r", alpha=0.7
        )
        T_hist_plot.plot(
            T_dist_actual.pdf(Temperature.sort_values()), Temperature.sort_values(), color=color_from_cycler[1]
        )
        if self.phase == "equilibration":
            T_main_plot.set(ylim=(T_desired * 0.85, T_desired * 1.15))
            T_hist_plot.set(ylim=(T_desired * 0.85, T_desired * 1.15))

        # ------------------------------------------- Total Energy -------------------------------------------#
        # Calculate Energy plot's labels and multipliers
        factor = 1.0 / process.parameters.J2erg if process.parameters.units == "cgs" else 1.0

        Energy = self.simulation_dataframe["Total Energy"].copy()  # * factor / process.parameters.eV2J  # Energy in [eV]
        time_mul, energy_mul, _, _, time_lbl, energy_lbl = plot_labels(
            self.simulation_dataframe["Time"], Energy, "Time", "Energy", self.units
        )
        Energy *= energy_mul
        # Total Energy moving average
        E_cumavg = Energy.expanding().mean()
        # Total Energy Deviation and its moving average
        Delta_E = (Energy - Energy.iloc[0]) * 100 / Energy.iloc[0]
        Delta_E_cum_avg = Delta_E.expanding().mean()
        # Deviation Plot
        E_delta_plot.plot(time, Delta_E, alpha=0.5)
        E_delta_plot.plot(time, Delta_E_cum_avg, alpha=0.8)
        E_delta_plot.set(xticks=[], ylabel=r"Deviation [%]")

        # Energy main plot
        E_main_plot.plot(time, Energy, alpha=0.7)
        E_main_plot.plot(time, E_cumavg, label="Moving Average")
        E_main_plot.axhline(Energy.mean(), ls="--", c="r", alpha=0.7, label="Avg")
        E_main_plot.legend(loc="best")
        E_main_plot.set(ylabel="Total Energy" + energy_lbl, xlabel="Time" + time_lbl)

        # Histogram plot

        # (Failed) Attempt to calculate the theoretical Energy distribution
        # In an NVT ensemble Energy fluctuation are given by sigma(E) = sqrt( k_B T^2 C_v)
        # where C_v is the isothermal heat capacity
        # Since this requires a lot of prior calculation I skip it and just make a Gaussian
        if phase == "production":
            self.calculate_heat_capacity_simulation(ensemble="NVE")
            NkB = self.total_num_ptcls * self.kB
            beta_desired = 1.0 / (self.kB * self.T_desired)
            delta_E2 = (
                self.dimensions
                * self.total_num_ptcls
                / beta_desired**2
                * (1 - 0.5 * self.dimensions * NkB / self.specific_heat_volume)
            )
        else:
            self.calculate_heat_capacity_simulation(ensemble="NVT")
            delta_E2 = self.specific_heat_volume / self.beta**2 / self.kB

        delta_E = sqrt(delta_E2)
        # Calculate the theoretical distribution of the energy.
        # E_dist_desired is a Gaussian centered at the actual mean with actaul E_std. This is to confirm that we have a Gaussian process.
        E_dist_desired = scp_stats.norm(loc=Energy.mean(), scale=delta_E * energy_mul)
        # E_dist_actual is a Gaussian centered at the actual mean with actaul E_std. This is to confirm that we have a Gaussian process.
        E_dist_actual = scp_stats.norm(loc=Energy.mean(), scale=Energy.std())
        sns_histplot(y=Energy, bins="fd", stat="density", alpha=0.75, legend="False", ax=E_hist_plot)
        # Grab the second color since the first is used for histplot
        E_hist_plot.plot(E_dist_desired.pdf(Energy.sort_values()), Energy.sort_values(), alpha=0.7, ls="--", color="r")
        E_hist_plot.plot(E_dist_actual.pdf(Energy.sort_values()), Energy.sort_values(), color=color_from_cycler[1])

        E_hist_plot.set(ylabel=None, xlabel=None, ylim=E_main_plot.get_ylim(), xticks=[], yticks=[])

        if not publication:
            dt_mul, _, _, _, dt_lbl, _ = plot_labels(
                process.integrator.dt, self.simulation_dataframe["Total Energy"], "Time", "Energy", self.units
            )

            # Information section
            Info_plot.axis([0, 10, 0, 10])
            Info_plot.grid(False)

            Info_plot.text(0.0, 10, "Job ID: {}".format(self.job_id))
            Info_plot.text(0.0, 9.5, "Phase: {}".format(self.phase.capitalize()))
            Info_plot.text(0.0, 9.0, "No. of species = {}".format(len(self.species_num)))
            y_coord = 8.5
            for isp, sp in enumerate(process.species):

                if sp.name != "electron_background":
                    Info_plot.text(0.0, y_coord, "Species {} : {}".format(isp + 1, sp.name))
                    Info_plot.text(0.0, y_coord - 0.5, "  No. of particles = {} ".format(sp.num))
                    Info_plot.text(
                        0.0, y_coord - 1.0, "  Temperature = {:.2f} {}".format(temp_mul * sp.temperature, temp_lbl)
                    )
                    y_coord -= 1.5

            y_coord -= 0.25
            delta_t = dt_mul * process.integrator.dt
            # Plasma Period
            t_wp = 2.0 * pi / process.parameters.total_plasma_frequency
            # Print some info to the left
            if info_list is None:
                integrator_type = {
                    "equilibration": process.integrator.equilibration_type,
                    "magnetization": process.integrator.magnetization_type,
                    "production": process.integrator.production_type,
                }

                info_list = [f"Total $N$ = {process.parameters.total_num_ptcls}"]

                if process.integrator.thermalization:
                    info_list.append(f"Thermostat: {process.integrator.thermostat_type}")
                    info_list.append(f"  Berendsen rate = {process.integrator.thermalization_rate:.2f}")

                eq_cycles = int(process.parameters.equilibration_steps * process.integrator.dt / t_wp)
                # calculate the actual coupling constant
                t_ratio = self.T_desired / self.simulation_dataframe["Temperature"].mean()
                coupling_constant = (
                    self.simulation_dataframe["Total Potential Energy"].mean()
                    / self.simulation_dataframe["Total Kinetic Energy"].mean()
                )
                to_append = [
                    f"Equilibration cycles = {eq_cycles}",
                    f"Potential: {process.potential.type}",
                    f"  Eff Coupl Const = {coupling_constant:.2e}",
                    f"  Tot Force Error = {process.potential.force_error:.2e}",
                    f"Integrator: {integrator_type[self.phase]}",
                ]
                [info_list.append(info) for info in to_append]
                if integrator_type[self.phase] == "langevin":
                    info_list.append(f"langevin gamma = {process.integrator.langevin_gamma:.4e}")

                prod_cycles = int(process.parameters.production_steps * process.integrator.dt / t_wp)

                to_append = [
                    r"  Plasma period $\tau_{\omega_p}$ = " + f"{t_wp*dt_mul:.2f} {dt_lbl}",
                    f"  $\Delta t$ = {delta_t:.2f} {dt_lbl} = {delta_t/t_wp/dt_mul:.2e}" + r"$\tau_{\omega_p}$",
                    # "Step interval = {}".format(self.dump_step),
                    # "Step interval time = {:.2f} {}".format(self.dump_step * delta_t, dt_lbl),
                    f"Completed steps = {completed_steps}",
                    f"Total steps = {self.no_steps}",
                    f"{100 * completed_steps / self.no_steps:.2f} % Completed",
                    # "Completed time = {:.2f} {}".format(completed_steps * delta_t / dt_mul * time_mul, time_lbl),
                    f"Production time = {self.no_steps * delta_t / dt_mul * time_mul:.2f} {time_lbl}",
                    f"Production cycles = {prod_cycles}",
                ]
                [info_list.append(info) for info in to_append]

            for itext, text_str in enumerate(info_list):
                Info_plot.text(0.0, y_coord, text_str)
                y_coord -= 0.5

            Info_plot.axis("off")

            fig.tight_layout()

        # Saving
        if figname:
            fig.savefig(os_path_join(self.saving_dir, figname + "_" + self.job_id + ".png"))
        else:
            fig.savefig(os_path_join(self.saving_dir, "Plot_EnsembleCheck_" + self.job_id + ".png"))

        if show:
            fig.show()

        # Restore the previous rcParams
        plt.rcParams = current_rcParams

    # def gamma_plot(self, phase: str = None, figname: str = None, show: bool = False):

    #     if phase:
    #         phase = phase.lower()
    #         self.parse(phase)
    #     else:
    #         self.parse()

    #     Gamma = self.simulation_dataframe["Total Potential Energy"] / self.simulation_dataframe["Total Kinetic Energy"]
    #     Gamma_T = self.simulation_dataframe["Total Potential Energy"] * self.beta / self.total_num_ptcls
    #     Gamma_a = self.coupling_constant * self.T_desired / (self.simulation_dataframe["Temperature"])
    #     time_mul, energy_mul, _, _, time_lbl, energy_lbl = plot_labels(
    #         self.simulation_dataframe["Time"], self.simulation_dataframe["Total Potential Energy"], "Time", "ElectronVolt", self.units
    #     )

    #     time = self.simulation_dataframe["Time"] * time_mul

    #     fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    #     ax[0].plot(time, Gamma)
    #     ax[0].plot(time, Gamma.expanding().mean(), alpha=0.8, label="Moving Average")
    #     # ax.axhline(self.coupling_constant, c = 'r', ls = '--', alpha = 0.7, label = r"Original $\Gamma$")
    #     ax[0].legend()
    #     ax[0].set(ylabel=r"$\Gamma = \frac{\langle U \rangle}{\langle K \rangle}$", xlabel=f"Time {time_lbl}")

    #     ax[1].plot(time, Gamma_T)
    #     ax[1].plot(time, Gamma_T.expanding().mean(), alpha=0.8, label="Moving Average")
    #     # ax.axhline(self.coupling_constant, c = 'r', ls = '--', alpha = 0.7, label = r"Original $\Gamma$")
    #     ax[1].legend()
    #     ax[1].set(ylabel=r"$\Gamma_T = \frac{\langle U \rangle}{k_B T}$", xlabel=f"Time {time_lbl}")

    #     ax[2].plot(time, Gamma_a)
    #     ax[2].plot(time, Gamma_a.expanding().mean(), alpha=0.8, label="Moving Average")
    #     ax[2].axhline(self.coupling_constant, c="r", ls="--", alpha=0.7, label=r"Original $\Gamma$")
    #     ax[2].legend()
    #     ax[2].set(
    #         ylabel=r"$\Gamma_a = \frac{Q^2}{4\pi \epsilon_0 a_{ws} } \frac{1}{\langle k_B T \rangle }$",
    #         xlabel=f"Time {time_lbl}",
    #     )

    #     # Saving
    #     if figname:
    #         fig.savefig(os_path_join(self.saving_dir, figname + "_" + self.job_id + ".png"))
    #     else:
    #         fig.savefig(os_path_join(self.saving_dir, "Plot_Gamma_" + self.job_id + ".png"))

    #     if show:
    #         fig.show()


class VelocityAutoCorrelationFunction(Observable):
    """Velocity Auto-correlation function."""

    def __init__(self):
        super(VelocityAutoCorrelationFunction, self).__init__()
        self.__name__ = "vacf"
        self.__long_name__ = "Velocity AutoCorrelation Function"
        self.no_ptcls_per_species = [10]
        self.particles_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.acf_observable = True

    @setup_doc
    def setup(self, params, phase: str = None, no_slices: int = None, **kwargs):

        super().setup_init(params, phase=phase, no_slices=no_slices)
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):
        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())
        self.update_finish()

        if "no_ptcls_per_species" in kwargs.keys():
            self.select_random_indices(kwargs["no_ptcls_per_species"])
        else:
            self.select_random_indices()

    @compute_doc
    def compute(self, calculate_acf_data: bool = True, no_ptcls_per_species=None):

        raise DeprecationWarning("VACF does not have a `compute` method anymore. Use `compute_acf()`.")

    @compute_acf_doc
    def compute_acf(self, no_ptcls_per_species=None):

        if no_ptcls_per_species:
            self.select_random_indices(no_ptcls_per_species)

        t0 = self.timer.current()
        self.calc_acf_slices_data()
        self.average_acf_slices_data()
        self.save_acf_hdf()
        tend = self.timer.current()
        time_stamp(self.log_file, f"{self.__long_name__} Calculation", self.timer.time_division(tend - t0), self.verbose)

    @calc_acf_slices_doc
    def calc_acf_slices_data(self):

        start_slice = 0
        end_slice = int(2 * self.acf_slice_steps * self.dump_step)

        time = zeros(2 * self.slice_steps)
        vel = zeros((self.dimensions, sum(self.no_ptcls_per_species), 2 * self.acf_slice_steps))

        for isl in tqdm(
            range(self.no_slices),
            desc=f"\nCalculating {self.__name__.swapcase()} for slice ",
            disable=not self.verbose,
            position=0,
        ):
            self.grab_sim_data(start_slice, end_slice, vel, time)

            if isl == 0:
                self.dataframe_acf["Time"] = time[: self.acf_slice_steps]
                self.dataframe_acf_slices["Time"] = time[: self.acf_slice_steps]

            # Return an array of shape = ( num_species, dim + 1, slice_steps)
            vacf = self.calculate_vacf(vel)

            self.save_acf_slices_data_to_hdf(vacf, isl)

            start_slice += self.acf_slice_steps * self.dump_step
            end_slice += self.acf_slice_steps * self.dump_step

    def save_acf_slices_data_to_hdf(self, vacf, isl):
        """Store ACF data of slice `isl` into a hierarchical dataframe.

        Parameters
        ----------
        vacf: numpy.ndarray
            Data to store,

        isl: int
            Number of the slice being evaluated.
        """
        for i, sp1 in enumerate(self.species_names):
            sp_vacf_str = f"{sp1} {self.__name__.swapcase()}"
            for d in range(self.dimensions):
                col_name = f"{sp_vacf_str}_{self.dim_labels[d]}_slice {isl}"
                col_data = vacf[i, d, :]
                self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, col_data, col_name)

            col_name = f"{sp_vacf_str}_Total_slice {isl}"
            col_data = vacf[i, -1, :]
            self.dataframe_acf_slices = add_col_to_df(self.dataframe_acf_slices, col_data, col_name)

    def grab_sim_data(self, start_dump_no, end_dump_no, vel, time):
        """
        Grab the data from simulation dumps.

        Parameters
        ----------
        start_dump_no: int
            Initial number of the checkpoint dump.

        end_dump_no: int
            Final number of the checkpoint dump.

        vel: numpy.ndarray
            Array in which to store the collected velocities.
            Shape = (:attr:`dimensions`, `len(`:attr:`no_ptcls_per_species``)`, `2`:attr:`acf_slice_steps`)

        time: numpy.ndarray
            Array in which to store the time collected from the dump.
            Shape = (`2`:attr:`acf_slice_steps`)

        """

        for it, dump in enumerate(
            tqdm(
                range(start_dump_no, end_dump_no, self.dump_step),
                desc="Reading data",
                disable=not self.verbose,
                position=1,
                leave=False,
            )
        ):
            datap = load_from_restart(self.dump_dir, dump)
            time[it] = datap["time"]
            for d in range(self.dimensions):
                vel[d, :, it] = datap["vel"][self.particles_id, d]

    def select_random_indices(self, no_ptcls_per_species=None):
        """Randomly select a given number of indices that indicate the particles to be used to average the VACF.
        The random number of indices is stored in :attr:`particles_id`.

        Parameters
        ----------
        no_ptcls_per_species: list, int, optional
            List containing the number of particles to randomly select for each species.
            If `None` it uses :attr:`no_ptcls_per_species` attribute which is equal to `[10]` by default.

        Raises
        ------
        _: ValueError
            If the chosen `no_ptcls_per_species` is less than the total number of particles in :attr:`species_num`.

        Notes
        -----
        If the length of :attr:`no_ptcls_per_species` is less than the length of :attr:`species_num`, it selects the minimum of :attr:`no_ptcls_per_species` to be used for all species.

        """

        # Check whether the user passed an integer
        if no_ptcls_per_species:
            if isinstance(no_ptcls_per_species, int):
                self.no_ptcls_per_species = [no_ptcls_per_species]

        # Total number of indices to select for each range
        rng = default_rng()

        # Check if the length of no_ptcls_per_species is equal to the length of species_num
        if len(self.no_ptcls_per_species) != len(self.species_num):
            # If not, adjust no_ptcls_per_species to have the same length and values
            self.no_ptcls_per_species = min(self.no_ptcls_per_species) * ones_like(self.species_num)

        # Initialize an empty list to store the combined random indices
        combined_random_indices = []

        species_start = 0
        species_end = 0
        for ip, num_ptcls in enumerate(self.no_ptcls_per_species):
            species_end += self.species_num[ip]
            # Check whether there are enough particles to choose from
            if num_ptcls > self.species_num[ip]:
                raise ValueError(
                    f"Species {self.species_names[ip]}: the chosen random number of particles, {num_ptcls}, is less than its total species number of particles, {self.species_num[ip]}"
                )

            # Generate unique random indices for the current range
            random_indices = rng.choice(range(species_start, species_end), size=num_ptcls, replace=False)

            # Append the current random indices to the combined list
            combined_random_indices.extend(random_indices)

        self.particles_id = combined_random_indices

    @avg_acf_slices_doc
    def average_acf_slices_data(self):
        """Average the data from all the slices and add it to the dataframe."""

        # Average the stuff
        for i, sp1 in enumerate(self.species_names):
            sp_vacf_str = f"{sp1} {self.__name__.swapcase()}"
            for d in range(self.dimensions):
                dl = self.dim_labels[d]
                dcol_str = [sp_vacf_str + f"_{dl}_slice {isl}" for isl in range(self.no_slices)]
                self.dataframe_acf[sp_vacf_str + f"_{dl}_Mean"] = self.dataframe_acf_slices[dcol_str].mean(axis=1)
                self.dataframe_acf[sp_vacf_str + f"_{dl}_Std"] = self.dataframe_acf_slices[dcol_str].std(axis=1)

            tot_col_str = [sp_vacf_str + f"_Total_slice {isl}" for isl in range(self.no_slices)]
            col_name = sp_vacf_str + "_Total_Mean"
            col_data = self.dataframe_acf_slices[tot_col_str].mean(axis=1).values
            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

            col_name = sp_vacf_str + "_Total_Std"
            col_data = self.dataframe_acf_slices[tot_col_str].std(axis=1).values
            self.dataframe_acf = add_col_to_df(self.dataframe_acf, col_data, col_name)

    # @jit Numba doesn't like Scipy
    def calculate_vacf(self, vel):
        """
        Calculate the velocity autocorrelation function of each species and in each direction.

        Parameters
        ----------
        vel : numpy.ndarray
            Particles' velocities stored in a 3D array with shape = (D x Np x Nt).
            D = cartesian dimensions, Np = Number of particles, Nt = number of dumps.

        Returns
        -------
        vacf: numpy.ndarray
            Velocity autocorrelation functions. Shape= (No_species, D + 1, Nt)

        """
        no_dim = vel.shape[0]

        vacf = zeros((self.num_species, no_dim + 1, self.acf_slice_steps))
        species_vacf = zeros(self.acf_slice_steps)
        ptcl_vacf = zeros(self.acf_slice_steps)

        # Calculate the vacf of each species in each dimension
        for d in tqdm(range(no_dim), desc=f"Dimension", position=1, disable=not self.verbose, leave=False):
            species_start = 0
            species_end = 0
            for sp, np in enumerate(
                tqdm(self.no_ptcls_per_species, desc=f"Species", position=2, disable=not self.verbose, leave=False)
            ):
                species_end += np
                # Temporary species vacf

                # Calculate the vacf for each particle of species sp
                for ptcl in range(species_start, species_end):

                    for it in range(self.acf_slice_steps):
                        v = vel[d, ptcl, : self.acf_slice_steps + it]
                        # Calculate the correlation function and add it to the array
                        ptcl_vacf[it] = correlationfunction(v, v)[it]

                    # Add this particle vacf to the species vacf and normalize by the time origins
                    species_vacf += ptcl_vacf

                # Save the species vacf for dimension i
                vacf[sp, d, :] = species_vacf / np
                # Save the total vacf
                vacf[sp, -1, :] += species_vacf / np
                # Move to the next species first particle position
                species_start += np

        return vacf


# TODO: Review and fix this class
class VelocityDistribution(Observable):
    """
    Moments of the velocity distributions defined as

    .. math::
        \\langle v^{\\alpha} \\rangle = \\int_{-\\infty}^{\\infty} d v \\, f(v) v^{2 \\alpha}.

    Attributes
    ----------
    no_bins: int
        Number of bins used to calculate the velocity distribution.

    plots_dir: str
        Directory in which to store Hermite coefficients plots.

    species_plots_dirs : list, str
        Directory for each species where to save Hermite coefficients plots.

    max_no_moment: int
        Maximum number of moments = :math:`\\alpha`. Default = 6.

    """

    def __init__(self):
        super(VelocityDistribution, self).__init__()
        self.max_no_moment = None
        self.__name__ = "vd"
        self.__long_name__ = "Velocity Distribution"

    def setup(
        self,
        params,
        phase: str = None,
        no_slices: int = None,
        hist_kwargs: dict = None,
        max_no_moment: int = None,
        multi_run_average: bool = None,
        dimensional_average: bool = None,
        runs: int = 1,
        curve_fit_kwargs: dict = None,
        **kwargs,
    ):

        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        runs
        dimensional_average
        multi_run_average
        params : :class:`sarkas.core.Parameters`
            Simulation's parameters.

        phase : str, optional
            Phase to compute. Default = 'production'.

        no_slices : int, optional

        max_no_moment : int, optional
            Maximum number of moments to calculate. Default = 6.

        hist_kwargs : dict, optional
            Dictionary of keyword arguments to pass to ``histogram`` for the calculation of the distributions.

        curve_fit_kwargs: dict, optional
            Dictionary of keyword arguments to pass to ``scipy.curve_fit`` for fitting of Hermite coefficients.

        **kwargs :
            These will overwrite any :class:`sarkas.core.Parameters`
            or default :class:`sarkas.tools.observables.Observable`
            attributes and/or add new ones.

        """

        super().setup_init(
            params,
            phase=phase,
            dimensional_average=dimensional_average,
            multi_run_average=multi_run_average,
            runs=runs,
            no_slices=no_slices,
        )
        self.update_args(hist_kwargs, max_no_moment, curve_fit_kwargs, **kwargs)

    @arg_update_doc
    def update_args(self, hist_kwargs: dict = None, max_no_moment: int = None, curve_fit_kwargs: dict = None, **kwargs):

        if curve_fit_kwargs:
            self.curve_fit_kwargs = curve_fit_kwargs

        # Check on hist_kwargs
        if hist_kwargs:
            # Is it a dictionary ?
            if not isinstance(hist_kwargs, dict):
                raise TypeError("hist_kwargs not a dictionary. Please pass a dictionary.")
            # Did you pass a single dictionary for multispecies?
            for key, value in hist_kwargs.items():
                # The elements of hist_kwargs should be lists
                if not isinstance(hist_kwargs[key], list):
                    hist_kwargs[key] = [value for i in range(self.num_species)]

            self.hist_kwargs = hist_kwargs

        # Default number of moments to calculate
        if max_no_moment:
            self.max_no_moment = max_no_moment
        else:
            self.max_no_moment = 6

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())
        self.update_finish()
        # # Directories in which to store plots
        # self.plots_dir = os_path_join(self.saving_dir, "Plots")
        # if not os_path_exists(self.plots_dir):
        #     mkdir(self.plots_dir)

        # Paths where to store the dataframes
        if self.multi_run_average:
            self.filename_csv = os_path_join(self.saving_dir, "VelocityDistribution.csv")
            self.filename_hdf = os_path_join(self.saving_dir, "VelocityDistribution.h5")
        else:
            self.filename_csv = os_path_join(self.saving_dir, "VelocityDistribution_" + self.job_id + ".csv")
            self.filename_hdf = os_path_join(self.saving_dir, "VelocityDistribution_" + self.job_id + ".h5")

        if hasattr(self, "max_no_moment"):
            self.moments_dataframe = None
            self.mom_df_filename_csv = os_path_join(self.saving_dir, "Moments_" + self.job_id + ".csv")

        if hasattr(self, "max_hermite_order"):
            self.hermite_dataframe = None
            self.herm_df_filename_csv = os_path_join(self.saving_dir, "HermiteCoefficients_" + self.job_id + ".csv")
            # some checks
            if not hasattr(self, "hermite_rms_tol"):
                self.hermite_rms_tol = 0.05

        self.species_plots_dirs = None

        # Need this for pretty print
        # Calculate the dimension of the velocity container
        # 2nd Dimension of the raw velocity array
        self.dim = 1 if self.dimensional_average else self.dimensions
        # range(inv_dim) for the loop over dimension
        self.inv_dim = self.dimensions if self.dimensional_average else 1

        self.prepare_histogram_args()

        self.save_pickle()

    def compute(self, compute_moments: bool = False, compute_Grad_expansion: bool = False):
        """
        Calculate the moments of the velocity distributions and save them to a pandas dataframes and csv.

        Parameters
        ----------
        hist_kwargs : dict, optional
            Dictionary with arguments to pass to ``numpy.histogram``.

        **kwargs :
            These will overwrite any :class:`sarkas.core.Parameters`
            or default :class:`sarkas.tools.observables.Observable`
            attributes and/or add new ones.

        """

        # Print info to screen
        self.pretty_print()

        # Grab simulation data
        time, vel_raw = self.grab_sim_data(pva="vel")
        # Normality test
        self.normality_tests(time=time, vel_data=vel_raw)

        # Make the velocity distribution
        self.create_distribution(vel_raw, time)

        # Calculate velocity moments
        if compute_moments:
            self.compute_moments(parse_data=False, vel_raw=vel_raw, time=time)
        #
        if compute_Grad_expansion:
            self.compute_hermite_expansion(compute_moments=False)

    def normality_tests(self, time, vel_data):
        """
        Calculate the Shapiro-Wilks test for each timestep from the raw velocity data and store it into a dataframe.

        Parameters
        ----------
        Returns
        -------
        time : numpy.ndarray
            One dimensional array with time data.

        vel_data : numpy.ndarray
            See `Returns` of :meth:`sarkas.tools.observables.Observable.grab_sim_data`
            Array with shape (:attr:`sarkas.tools.observables.Observable.no_dumps`,
        :attr:`sarkas.tools.observables.Observable.self.dim`, :attr:`sarkas.tools.observables.Observable.runs` *
        :attr:`sarkas.tools.observables.Observable.inv_dim` * :attr:`sarkas.tools.observables.Observable.total_num_ptcls`).
        `.dim` = 1 if :attr:`sarkas.tools.observables.Observable.dimensional_average = True` otherwise equals the number
        of dimensions, (e.g. 3D : 3) `.runs` is the number of runs to be averaged over. Default = 1. `.inv_dim` is
        the else option of `dim`. If `dim = 1` then `.inv_dim = .dimensions` and viceversa.

        """

        no_dim = vel_data.shape[1]

        stats_df_columns = (
            "Time",
            *[
                "{}_{}_{}".format(sp, d, st)
                for sp in self.species_names
                for _, d in zip(range(no_dim), ["X", "Y", "Z"])
                for st in ["s", "p"]
            ],
        )

        stats_mat = zeros((len(time), len(self.species_num) * no_dim * 2 + 1))
        for it, tme in enumerate(time):
            stats_mat[it, 0] = tme
            for d, ds in zip(range(no_dim), ["X", "Y", "Z"]):
                for sp, sp_start in enumerate(self.species_index_start[:-1]):
                    # Calculate the correct start and end index for storage
                    sp_end = self.species_index_start[sp + 1]

                    statcs, p_value = scp_stats.shapiro(vel_data[it, d, sp_start:sp_end])
                    stats_mat[it, 1 + 6 * sp + 2 * d] = statcs
                    stats_mat[it, 1 + 6 * sp + 2 * d + 1] = p_value

        self.norm_test_df = DataFrame(stats_mat, columns=stats_df_columns)
        self.norm_test_df.columns = MultiIndex.from_tuples([tuple(c.split("_")) for c in stats_df_columns])

    def prepare_histogram_args(self):

        # Initialize histograms arguments
        if not hasattr(self, "hist_kwargs"):
            self.hist_kwargs = {"density": [], "bins": [], "range": []}
        # Default values
        bin_width = 0.05
        # Range of the histogram = (-wid * vth, wid * vth)
        wid = 5
        # The number of bins is calculated from default values of bin_width and wid
        no_bins = int(2.0 * wid / bin_width)
        # Calculate thermal speed from energy/temperature data.
        try:
            energy_fle = self.prod_energy_filename if self.phase == "production" else self.eq_energy_filename
            energy_df = read_csv(energy_fle, index_col=False, encoding="utf-8")
            if self.num_species > 1:
                vth = zeros(self.num_species)
                for sp, (sp_mass, sp_name) in enumerate(zip(self.species_masses, self.species_names)):
                    vth[sp] = sqrt(energy_df["{} Temperature".format(sp_name)].mean() * self.kB / sp_mass)
            else:
                vth = sqrt(energy_df["Temperature"].mean() * self.kB / self.species_masses)

        except FileNotFoundError:
            # In case you are using this in PreProcessing stage
            vth = sqrt(self.kB * self.T_desired / self.species_masses)

        self.vth = vth.copy()

        # Create the default dictionary of histogram args
        default_hist_kwargs = {"density": [], "bins": [], "range": []}
        if self.num_species > 1:
            for sp in range(self.num_species):
                default_hist_kwargs["density"].append(True)
                default_hist_kwargs["bins"].append(no_bins)
                default_hist_kwargs["range"].append((-wid * vth[sp], wid * vth[sp]))
        else:
            default_hist_kwargs["density"].append(True)
            default_hist_kwargs["bins"].append(no_bins)
            default_hist_kwargs["range"].append((-wid * vth[0], wid * vth[0]))

        # Now do some checks.
        # Check for numpy.histogram args in kwargs
        must_have_keys = ["bins", "range", "density"]

        for k, key in enumerate(must_have_keys):
            try:
                # Is it empty?
                if len(self.hist_kwargs[key]) == 0:
                    self.hist_kwargs[key] = default_hist_kwargs[key]
            except KeyError:
                self.hist_kwargs[key] = default_hist_kwargs[key]

        # Ok at this point I have a dictionary whose elements are list.
        # I want the inverse: a list whose elements are dicts
        self.list_hist_kwargs = []
        for indx in range(self.num_species):
            another_dict = {}
            # Loop over the keys and grab the species value
            for key, values in self.hist_kwargs.items():
                another_dict[key] = values[indx]

            self.list_hist_kwargs.append(another_dict)

    def create_distribution(self, vel_raw: ndarray = None, time: ndarray = None):
        """
        Calculate the velocity distribution of each species and save the corresponding dataframes.

        Parameters
        ----------
        vel_raw: ndarray, optional
            Container of particles velocity at each time step.

        time: ndarray, optional
            Time array.

        """

        print("\nCreating velocity distributions ...")
        tinit = self.timer.current()
        no_dumps = vel_raw.shape[0]
        no_dim = vel_raw.shape[1]
        # I want a Hierarchical dataframe
        # Example:
        #   Ca	                        Yb	    Ca	                        Yb	    Ca	                        Yb
        #   X	                        X	    Y	                        Y	    Z	                        Z
        #   1.54e-03 -2.54e+22 3.54e+00	1 2     1.54e-03 -2.54e+22 3.54e+00	1 2	    1.54e-03 -2.54e+22 3.54e+00 1 2
        # This has 3 rows of columns. The first identifies the species, The second the axis,
        # and the third the value of the bin_edge
        # This can be easily obtained from pandas dataframe MultiIndex. But first I will create a dataframe
        # from a huge matrix. The columns of this dataframe will be
        # Ca_X_1.54e-03 Ca_X_-2.54e+22 Ca_X_3.54e+00 Yb_X_1	Yb_X_2 Ca_Y_1.54e-03 Ca_Y_-2.54e+22 Ca_Y_3.54e+00 Yb_Y_1 ...
        # using MultiIndex.from_tuples([tuple(c.split("_")) for c in df.columns]) I can create a hierarchical df.
        # This is because I can access the data as
        # df['Ca']['X']
        # >>>       1.54e-03	-2.54e+22	3.54e+00
        # >>> Time
        # >>>   0   -1.694058	1.217008	-0.260678
        # with the index = to my time array.
        # see https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#reconstructing-the-level-labels

        # Columns

        full_df_columns = []
        # At the first time step I will create the columns list.
        dist_matrix = zeros((len(time), self.dim * (sum(self.hist_kwargs["bins"]) + self.num_species)))
        # The +1 at the end is because I decided to add a column containing the timestep
        # For convenience save the bin edges somewhere else. The columns of the dataframe are string. This will give
        # problems when plotting.
        # Use a dictionary since the arrays could be different lengths
        self.species_bin_edges = {}
        for k in self.species_names:
            # Initialize the sub-dictionary
            self.species_bin_edges[k] = {}

        for it in range(no_dumps):
            for d, ds in zip(range(no_dim), ["X", "Y", "Z"]):
                indx_0 = 0
                for indx, (sp_name, sp_start) in enumerate(zip(self.species_names, self.species_index_start)):
                    # Calculate the correct start and end index for storage
                    sp_end = self.species_index_start[indx + 1]

                    bin_count, bin_edges = histogram(vel_raw[it, d, sp_start:sp_end], **self.list_hist_kwargs[indx])

                    # Executive decision: Center the bins
                    bin_loc = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    # Create the second_column_row the dataframe
                    if it == 0:
                        # Create the column array
                        full_df_columns.append(["{}_{}_Time".format(sp_name, ds)])
                        full_df_columns.append(["{}_{}_{:6e}".format(sp_name, ds, be) for be in bin_loc])
                        self.species_bin_edges[sp_name][ds] = bin_edges
                        # Ok. Now I have created the bins columns' names
                    # Time to insert in the huge matrix.
                    indx_1 = indx_0 + 1 + len(bin_count)
                    dist_matrix[it, indx_0] = time[it]
                    dist_matrix[it, indx_0 + 1 : indx_1] = bin_count
                    indx_0 = indx_1
        # Alright. The matrix is filled now onto the dataframe
        # First let's flatten the columns array. This is because I have something like this
        # Example: Binary Mixture with 3 H bins per axis and 2 He bins per axis
        # columns =[['H_X', 'H_X', 'H_X'], ['He_X', 'He_X'], ['H_Y', 'H_Y', 'H_Y'], ['He_Y', 'He_Y'] ... Z-axis]
        # Flatten with list(concatenate(columns).flat)
        # Now has become
        # first_column_row=['H_X', 'H_X', 'H_X', 'He_X', 'He_X', 'H_Y', 'H_Y', 'H_Y', 'He_Y', 'He_Y' ... Z-axis]
        # I think this is easier to understand than using nested list comprehension
        # see https://stackabuse.com/python-how-to-flatten-list-of-lists/
        full_df_columns = list(concatenate(full_df_columns).flat)
        self.dataframe = DataFrame(dist_matrix, columns=full_df_columns)
        # Save it
        self.dataframe.to_csv(self.filename_csv, encoding="utf-8", index=False)

        # Hierarchical DataFrame
        self.hierarchical_dataframe = self.dataframe.copy()
        self.hierarchical_dataframe.columns = MultiIndex.from_tuples(
            [tuple(c.split("_")) for c in self.hierarchical_dataframe.columns]
        )
        self.hierarchical_dataframe.to_hdf(self.filename_hdf, key=self.__name__, encoding="utf-8")

        tend = self.timer.current()
        time_stamp(
            self.log_file, "Velocity distribution calculation", self.timer.time_division(tend - tinit), self.verbose
        )

    def compute_moments(self, parse_data: bool = False, vel_raw: ndarray = None, time: ndarray = None):
        """Calculate and save moments of the distribution.

        Parameters
        ----------
        parse_data: bool
            Flag for reading data. Default = False. If False, must pass ``vel_raw`` and ``time``.
            If True it will parse data from simulations dumps.

        vel_raw: ndarray, optional
            Container of particles velocity at each time step.

        time: ndarray, optional
            Time array.

        """
        self.moments_dataframe = DataFrame()
        self.moments_hdf_dataframe = DataFrame()

        if parse_data:
            time, vel_raw = self.grab_sim_data()

        self.moments_dataframe["Time"] = time

        print("\nCalculating velocity moments ...")
        tinit = self.timer.current()
        moments, ratios = calc_moments(vel_raw, self.max_no_moment, self.species_index_start)
        tend = self.timer.current()
        time_stamp(self.log_file, "Velocity moments calculation", self.timer.time_division(tend - tinit), self.verbose)

        # Save the dataframe
        if self.dimensional_average:
            for i, sp in enumerate(self.species_names):
                self.moments_hdf_dataframe[f"{sp}_X_Time"] = time
                for m in range(self.max_no_moment):
                    self.moments_dataframe[f"{sp} {m + 1} moment"] = moments[i, :, 0, m]
                    self.moments_hdf_dataframe[f"{sp}_X_{m + 1} moment"] = moments[i, :, 0, m]
                for m in range(self.max_no_moment):
                    self.moments_dataframe[f"{sp} {m + 1} moment ratio"] = ratios[i, :, 0, m]
                    self.moments_hdf_dataframe[f"{sp}_X_{m + 1}-2 ratio"] = ratios[i, :, 0, m]
        else:
            for i, sp in enumerate(self.species_names):
                for d, ds in zip(range(self.dim), ["X", "Y", "Z"]):
                    self.moments_hdf_dataframe[f"{sp}_{ds}_Time"] = time
                    for m in range(self.max_no_moment):
                        self.moments_dataframe[f"{sp} {m + 1} moment axis {ds}"] = moments[i, :, d, m]
                        self.moments_hdf_dataframe[f"{sp}_{ds}_{m + 1} moment"] = moments[i, :, d, m]

                for d, ds in zip(range(self.dim), ["X", "Y", "Z"]):
                    self.moments_hdf_dataframe[f"{sp}_{ds}_Time"] = time
                    for m in range(self.max_no_moment):
                        self.moments_dataframe[f"{sp} {m + 1} moment ratio axis {ds}"] = ratios[i, :, d, m]
                        self.moments_hdf_dataframe[f"{sp}_{ds}_{m + 1}-2 ratio"] = ratios[i, :, d, m]

        self.moments_dataframe.to_csv(self.filename_csv, index=False, encoding="utf-8")
        # Hierarchical DF Save
        # Make the columns
        self.moments_hdf_dataframe.columns = MultiIndex.from_tuples(
            [tuple(c.split("_")) for c in self.moments_hdf_dataframe.columns]
        )
        # Save the df in the hierarchical df with a new key/group
        self.moments_hdf_dataframe.to_hdf(self.filename_hdf, mode="a", key="velocity_moments", encoding="utf-8")

    def compute_hermite_expansion(self, compute_moments: bool = False):
        """
        Calculate and save Hermite coefficients of the Grad expansion.

        Parameters
        ----------
        compute_moments: bool
            Flag for calculating velocity moments. These are needed for the hermite calculation.
            Default = True.

        Notes
        -----
        This is still in the development stage. Do not trust the results. It requires more study in non-equilibrium
        statistical mechanics.
        """
        from scipy.optimize import curve_fit

        self.hermite_dataframe = DataFrame()
        self.hermite_hdf_dataframe = DataFrame()

        if compute_moments:
            self.compute_moments(parse_data=True)

        if not hasattr(self, "hermite_rms_tol"):
            self.hermite_rms_tol = 0.05

        self.hermite_dataframe["Time"] = self.moments_dataframe["Time"].copy()
        self.hermite_sigmas = zeros((self.num_species, self.dim, len(self.hermite_dataframe["Time"])))
        self.hermite_epochs = zeros((self.num_species, self.dim, len(self.hermite_dataframe["Time"])))
        hermite_coeff = zeros(
            (self.num_species, self.dim, self.max_hermite_order + 1, len(self.hermite_dataframe["Time"]))
        )

        print("\nCalculating Hermite coefficients ...")
        tinit = self.timer.current()

        for sp, sp_name in enumerate(tqdm(self.species_names, desc="Species")):
            for it, t in enumerate(tqdm(self.hermite_dataframe["Time"], desc="Time")):
                # Grab the thermal speed from the moments
                vrms = self.moments_dataframe["{} 2 moment".format(sp_name)].iloc[it]
                # Loop over dimensions. No tensor availability yet
                for d, ds in zip(range(self.dim), ["X", "Y", "Z"]):

                    # Grab the distribution from the hierarchical df
                    dist = self.hierarchical_dataframe[sp_name][ds].iloc[it, 1:]

                    # Grab and center the bins
                    v_bins = 0.5 * (self.species_bin_edges[sp_name][ds][1:] + self.species_bin_edges[sp_name][ds][:-1])
                    cntrl = True
                    j = 0

                    # This is a routine to calculate the hermite coefficient in the case of non-equilibrium simulations.
                    # In non-equilibrium we cannot define a temperature. This is because the temperature is defined from
                    # the rms width of a Gaussian distribution. In non-equilibrium we don't have a Gaussian, thus the
                    # first thing to do is to find the underlying Gaussian in our distribution. This is what this
                    # iterative procedure is for.

                    while cntrl:
                        # Normalize
                        norm = trapz(dist, x=v_bins / vrms)

                        # Calculate the hermite coeff
                        h_coeff = calculate_herm_coeff(v_bins / vrms, dist / norm, self.max_hermite_order)

                        # Fit the rms only to the Grad expansion. This finds the underlying Gaussian
                        res, _ = curve_fit(
                            # the lambda func is because i need to fit only rms not the h_coeff
                            lambda x, rms: grad_expansion(x, rms, h_coeff),
                            v_bins / vrms,
                            dist / norm,
                            self.curve_fit_kwargs,
                        )

                        vrms *= res[0]

                        if abs(1.0 - res[0]) < self.hermite_rms_tol:
                            cntrl = False
                            self.hermite_sigmas[sp, d, it] = vrms
                            self.hermite_epochs[sp, d, it] = j
                            hermite_coeff[sp, d, :, it] = h_coeff
                        j += 1

        tend = self.timer.current()

        for sp, sp_name in enumerate(self.species_names):
            for d, ds in zip(range(self.dim), ["X", "Y", "Z"]):
                for h in range(self.max_hermite_order):
                    self.hermite_dataframe["{} {} {} Hermite coeff".format(sp_name, ds, h)] = hermite_coeff[sp, d, h, :]
                    if h == 0:
                        self.hermite_hdf_dataframe["{}_{}_Time".format(sp_name, ds, h)] = hermite_coeff[sp, d, h, :]
                        self.hermite_hdf_dataframe["{}_{}_RMS".format(sp_name, ds, h)] = self.hermite_sigmas[sp, d, :]
                        self.hermite_hdf_dataframe["{}_{}_epoch".format(sp_name, ds, h)] = self.hermite_epochs[sp, d, :]
                    else:
                        self.hermite_hdf_dataframe["{}_{}_{} coeff".format(sp_name, ds, h)] = hermite_coeff[sp, d, h, :]

        # Save the CSV
        self.hermite_dataframe.to_csv(self.herm_df_filename_csv, index=False, encoding="utf-8")
        # Make the columns
        self.hermite_hdf_dataframe.columns = MultiIndex.from_tuples(
            [tuple(c.split("_")) for c in self.hermite_hdf_dataframe.columns]
        )
        # Save the df in the hierarchical df with a new key/group
        self.hermite_hdf_dataframe.to_hdf(self.filename_hdf, mode="a", key="hermite_coefficients", encoding="utf-8")

        time_stamp(self.log_file, "Hermite expansion calculation", self.timer.time_division(tend - tinit), self.verbose)

    def pretty_print(self):
        """Print information in a user-friendly way."""

        print("\n\n{:=^70} \n".format(" " + self.__long_name__ + " "))
        print("CSV dataframe saved in:\n\t ", self.filename_csv)
        print("HDF5 dataframe saved in:\n\t ", self.filename_hdf)
        print("Data accessible at: self.dataframe, self.hierarchical_dataframe, self.species_bin_edges")
        print("\nMulti run average: ", self.multi_run_average)
        print("No. of runs: ", self.runs)
        print(
            "Size of the parsed velocity array: {} x {} x {}".format(
                self.no_dumps, self.dim, self.runs * self.inv_dim * self.total_num_ptcls
            )
        )
        print("\nHistograms Information:")
        for sp, (sp_name, dics) in enumerate(zip(self.species_names, self.list_hist_kwargs)):
            if sp == 0:
                print("Species: {}".format(sp_name))
            else:
                print("\nSpecies: {}".format(sp_name))
            print("No. of samples = {}".format(self.species_num[sp] * self.inv_dim * self.runs))
            print("Thermal speed: v_th = {:.6e} ".format(self.vth[sp]), end="")
            print("[cm/s]" if self.units == "cgs" else "[m/s]")
            for key, values in dics.items():
                if key == "range":
                    print(
                        "{} : ( {:.2f}, {:.2f} ) v_th,"
                        "\n\t( {:.4e}, {:.4e} ) ".format(key, *values / self.vth[sp], *values),
                        end="",
                    )
                    print("[cm/s]" if self.units == "cgs" else "[m/s]")
                else:
                    print("{}: {}".format(key, values))
            bin_width = abs(dics["range"][1] - dics["range"][0]) / (self.vth[sp] * dics["bins"])
            print("Bin Width = {:.4f}".format(bin_width))

        if hasattr(self, "max_no_moment"):
            print("\nMoments Information:")
            print("CSV dataframe saved in:\n\t ", self.mom_df_filename_csv)
            print("Data accessible at: self.moments_dataframe, self.moments_hdf_dataframe")
            print("Highest moment to calculate: {}".format(self.max_no_moment))

        if hasattr(self, "max_hermite_order"):
            print("\nGrad Expansion Information:")
            print("CSV dataframe saved in:\n\t ", self.herm_df_filename_csv)
            print("Data accessible at: self.hermite_dataframe, self.hermite_hdf_dataframe")
            print("Highest order to calculate: {}".format(self.max_hermite_order))
            print("RMS Tolerance: {:.3f}".format(self.hermite_rms_tol))


@njit
def calc_Sk(nkt, k_list, k_counts, species_np, no_dumps):
    """
    Calculate :math:`S_{ij}(k)` at each saved timestep.

    Parameters
    ----------
    nkt : numpy.ndarray, complex
        Density fluctuations of all species. Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    k_list :
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``k_counts``.
        Shape=(``no_ka_values``, 5)

    k_counts : numpy.ndarray
        Number of times each :math:`k` magnitude appears.

    species_np : numpy.ndarray
        Array with number of particles of each species.

    no_dumps : int
        Number of dumps.

    Returns
    -------

    Sk_all : numpy.ndarray
        Array containing :math:`S_{ij}(k)`. Shape=(``no_Sk``, ``no_ka_values``, ``no_dumps``)

    """

    no_sk = int(len(species_np) * (len(species_np) + 1) / 2)
    Sk_raw = zeros((no_sk, len(k_counts), no_dumps))
    pair_indx = 0
    for ip, si in enumerate(species_np):
        for jp in range(ip, len(species_np)):
            sj = species_np[jp]
            dens_const = 1.0 / sqrt(si * sj)
            for it in range(no_dumps):
                for ik, ka in enumerate(k_list):
                    indx = int(ka[-1])
                    nk_i = nkt[ip, it, ik]
                    nk_j = nkt[jp, it, ik]
                    Sk_raw[pair_indx, indx, it] += real(nk_i.conjugate() * nk_j) * dens_const / k_counts[indx]
            pair_indx += 1

    return Sk_raw


def calc_Skw(nkt, ka_list, species_np, no_dumps, dt, dump_step):
    """
    Calculate the Fourier transform of the correlation function of ``nkt``.

    Parameters
    ----------
    nkt :  complex, numpy.ndarray
        Particles' density or velocity fluctuations.
        Shape = ( ``no_species``, ``no_dumps``, ``no_k_list``)

    ka_list : list
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``ka_counts``.
        Shape=(``no_ka_values``, 5)

    species_np : numpy.ndarray
        Array with one element giving number of particles.

    no_dumps : int
        Number of dumps.

    dt : float
        Time interval.

    dump_step : int
        Snapshot interval.

    Returns
    -------
    Skw_all : numpy.ndarray
        DSF/CCF of each species and pair of species.
        Shape = (``no_skw``, ``no_ka_values``, ``no_dumps``)
    """
    # Fourier transform normalization: norm = dt / Total time
    norm = dt / sqrt(no_dumps * dt * dump_step)
    # number of independent observables
    no_skw = int(len(species_np) * (len(species_np) + 1) / 2)
    # DSF
    # Skw = zeros((no_skw, len(ka_counts), no_dumps))
    Skw_all = zeros((no_skw, len(ka_list), no_dumps))
    pair_indx = 0
    for ip, si in enumerate(species_np):
        for jp in range(ip, len(species_np)):
            sj = species_np[jp]
            dens_const = 1.0 / sqrt(si * sj)
            for ik, ka in enumerate(ka_list):
                # indx = int(ka[-1])
                nkw_i = fft(nkt[ip, :, ik]) * norm
                nkw_j = fft(nkt[jp, :, ik]) * norm
                Skw_all[pair_indx, ik, :] = fftshift(real(nkw_i.conjugate() * nkw_j) * dens_const)
                # Skw[pair_indx, indx, :] += Skw_all[pair_indx, ik, :] / ka_counts[indx]
            pair_indx += 1

    return Skw_all


@njit
def calc_elec_current(vel, sp_charge, sp_num):
    """
    Calculate the total electric current and electric current of each species.

    Parameters
    ----------
    vel: numpy.ndarray
        Particles' velocities.

    sp_charge: numpy.ndarray
        Charge of each species.

    sp_num: numpy.ndarray
        Number of particles of each species.

    Returns
    -------
    Js : numpy.ndarray
        Electric current of each species. Shape = (``no_species``, ``no_dim``, ``no_dumps``)

    Jtot : numpy.ndarray
        Total electric current. Shape = (``no_dim``, ``no_dumps``)
    """

    no_dumps = vel.shape[1]
    no_dim = vel.shape[0]

    Js = zeros((sp_num.shape[0], no_dim, no_dumps))
    Jtot = zeros((no_dim, no_dumps))

    sp_start = 0
    sp_end = 0
    for s, (q_sp, n_sp) in enumerate(zip(sp_charge, sp_num)):
        # Find the index of the last particle of species s
        sp_end += n_sp
        # Calculate the current of each species
        Js[s, :, :] = q_sp * vel[:, :, sp_start:sp_end].sum(axis=-1)
        # Add to the total current
        Jtot[:, :] += Js[s, :, :]

        sp_start += n_sp

    return Js, Jtot


def calc_moments(dist, max_moment, species_index_start):
    """
    Calculate the moments of the (velocity) distribution.

    Parameters
    ----------
    dist: numpy.ndarray
        Distribution of each time step. Shape = (``no_dumps``, ``dim``, ``runs * inv_dim * total_num_ptlcs``)

    max_moment: int
        Maximum moment to calculate

    species_index_start: numpy.ndarray
        Array containing the start index of each species. The last value is equivalent to dist.shape[-1]

    Returns
    -------
    moments: numpy.ndarray
        Moments of the distribution.
        Shape=( ``no_species``, ``no_dumps``, ``dim``, ``max_moment`` )

    ratios: numpy.ndarray
        Ratios of each moment with respect to the expected Maxwellian value.
        Shape=( ``no_species``, ``no_dumps``,  ``no_dim``, ``max_moment - 1``)

    Notes
    -----
    See these `equations <https://en.wikipedia.org/wiki/Normal_distribution#Moments:~:text=distribution.-,Moments>`_
    """

    # from scipy.stats import moment as scp_moment
    from scipy.special import gamma as scp_gamma

    no_species = len(species_index_start)
    no_dumps = dist.shape[0]
    dim = dist.shape[1]
    moments = zeros((no_species, no_dumps, dim, max_moment))
    ratios = zeros((no_species, no_dumps, dim, max_moment))

    for indx, sp_start in enumerate(species_index_start[:-1]):
        # Calculate the correct start and end index for storage
        sp_end = species_index_start[indx + 1]

        for mom in range(max_moment):
            moments[indx, :, :, mom] = scp_stats.moment(dist[:, :, sp_start:sp_end], moment=mom + 1, axis=-1)

    # sqrt( <v^2> ) = standard deviation = moments[:, :, :, 1] ** (1/2)
    for mom in range(max_moment):
        pwr = mom + 1
        const = 2.0 ** (pwr / 2) * scp_gamma((pwr + 1) / 2) / sqrt(pi)
        ratios[:, :, :, mom] = moments[:, :, :, mom] / (const * moments[:, :, :, 1] ** (pwr / 2.0))

    return moments, ratios


@njit
def calc_nk(pos_data, k_list):
    """
    Calculate the instantaneous microscopic density :math:`n(k)` defined as

    .. math::
        n_{A} ( k ) = \\sum_i^{N_A} \\exp [ -i \\mathbf k \\cdot \\mathbf r_{i} ]

    Parameters
    ----------
    pos_data : numpy.ndarray
        Particles' position scaled by the box lengths.
        Shape = ( ``no_dumps``, ``no_dim``, ``tot_no_ptcls``)

    k_list : list
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``ka_counts``.
        Shape=(``no_ka_values``, 5)

    Returns
    -------
    nk : numpy.ndarray
        Array containing :math:`n(k)`.
    """

    nk = zeros(len(k_list), dtype=complex128)

    for ik, k_vec in enumerate(k_list):
        kr_i = 2.0 * pi * (k_vec[0] * pos_data[:, 0] + k_vec[1] * pos_data[:, 1] + k_vec[2] * pos_data[:, 2])
        nk[ik] = (exp(-1j * kr_i)).sum()

    return nk


def calc_nkt(fldr, slices, dump_step, species_np, k_list, verbose):
    """
    Calculate density fluctuations :math:`n(k,t)` of all species.

    .. math::
        n_{A} ( k, t ) = \\sum_i^{N_A} \\exp [ -i \\mathbf k \\cdot \\mathbf r_{i}(t) ]

    where :math:`N_A` is the number of particles of species :math:`A`.

    Parameters
    ----------
    fldr : str
        Name of folder containing particles data.

    slices : tuple, int
        Initial, final step number of the slice, total number of slice steps.

    dump_step : int
        Snapshot interval.

    species_np : numpy.ndarray
        Number of particles of each species.

    k_list : list
        List of :math: `k` vectors.

    Return
    ------
    nkt : numpy.ndarray, complex
        Density fluctuations.  Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)
    """

    # Read particles' position for times in the slice
    nkt = zeros((len(species_np), slices[2], len(k_list)), dtype=complex128)
    for it, dump in enumerate(
        tqdm(range(slices[0], slices[1], dump_step), desc="Timestep", position=1, disable=not verbose, leave=False)
    ):
        data = load_from_restart(fldr, dump)
        pos = data["pos"]
        sp_start = 0
        sp_end = 0
        for i, sp in enumerate(species_np):
            sp_end += sp
            nkt[i, it, :] = calc_nk(pos[sp_start:sp_end, :], k_list)
            sp_start += sp

    return nkt


@njit
def calc_statistical_efficiency(observable, run_avg, run_std, max_no_divisions, no_dumps):
    """
    Todo:

    Parameters
    ----------
    observable
    run_avg
    run_std
    max_no_divisions
    no_dumps

    Returns
    -------

    """
    tau_blk = zeros(max_no_divisions)
    sigma2_blk = zeros(max_no_divisions)
    statistical_efficiency = zeros(max_no_divisions)
    for i in range(2, max_no_divisions):
        tau_blk[i] = int(no_dumps / i)
        for j in range(i):
            t_start = int(j * tau_blk[i])
            t_end = int((j + 1) * tau_blk[i])
            blk_avg = observable[t_start:t_end].mean()
            sigma2_blk[i] += (blk_avg - run_avg) ** 2
        sigma2_blk[i] /= i - 1
        statistical_efficiency[i] = tau_blk[i] * sigma2_blk[i] / run_std**2

    return tau_blk, sigma2_blk, statistical_efficiency


# @jit Numba doesn't like scipy.signal
def calc_diff_flux_acf(vel, sp_num, sp_conc, sp_mass):
    """
    Calculate the diffusion fluxes and their autocorrelations functions in each direction.

    Parameters
    ----------
    vel : numpy.ndarray
        Particles' velocities. Shape = (``dimensions``, ``no_dumps``, :attr:`total_num_ptcls`)

    sp_num: numpy.ndarray
        Number of particles of each species.

    sp_conc: numpy.ndarray
        Concentration of each species.

    sp_mass: numpy.ndarray
        Particle's mass of each species.

    Returns
    -------
    J_flux: numpy.ndarray
        Diffusion fluxes.
        Shape = ( (``num_species - 1``), ``dimensions`` , ``no_dumps``)

    jr_acf: numpy.ndarray
        Relative Diffusion flux autocorrelation function.
        Shape = ( (``num_species - 1``) x (``num_species - 1``), ``no_dim + 1``, ``no_dumps``)

    """

    no_dim = vel.shape[0]
    no_dumps = vel.shape[1]
    no_species = len(sp_num)
    # number of independent fluxes = no_species - 1,
    # number of acf of ind fluxes = no_species - 1 ^2
    no_jc_acf = int((no_species - 1) * (no_species - 1))

    # Current of each species in each direction and at each timestep
    tot_vel = zeros((no_species, no_dim, no_dumps))

    sp_start = 0
    sp_end = 0
    # Calculate the total center of mass velocity (tot_com_vel)
    # and the center of mass velocity of each species (com_vel)
    for i, ns in enumerate(sp_num):
        sp_end += ns
        tot_vel[i, :, :] = vel[:, :, sp_start:sp_end].sum(axis=-1)
        # tot_com_vel += mass_densities[i] * com_vel[i, :, :] / tot_mass_dens
        sp_start += ns

    # Diffusion Fluxes
    J_flux = zeros((no_species - 1, no_dim, no_dumps))

    # Relative Diffusion fluxes for ACF and Transport calc
    jr_flux = zeros((no_species - 1, no_dim, no_dumps))

    # Relative diff flux acf
    jr_acf = zeros((no_jc_acf, no_dim + 1, no_dumps))

    m_bar = sp_mass @ sp_conc
    # the diffusion fluxes from eq.(3.5) in Zhou J Phs Chem
    for i, m_alpha in enumerate(sp_mass[:-1]):
        # Flux
        for j, m_beta in enumerate(sp_mass):
            delta_ab = 1 * (m_beta == m_alpha)
            J_flux[i, :, :] += (m_bar * delta_ab - sp_conc[i] * m_beta) * tot_vel[j, :, :]
            jr_flux[i, :, :] += (delta_ab - sp_conc[i]) * tot_vel[j, :, :]
        J_flux[i, :, :] *= m_alpha / m_bar

    indx = 0
    # Remember to change this for 3+ species
    # binary_const = ( m_bar/sp_mass.prod() )**2
    # Calculate the correlation function in each direction
    for i, sp1_flux in enumerate(jr_flux):
        for j, sp2_flux in enumerate(jr_flux):
            for d in range(no_dim):
                # Calculate the correlation function and add it to the array
                jr_acf[indx, d, :] = correlationfunction(sp1_flux[d, :], sp2_flux[d, :])
                # Calculate the total correlation function by summing the three directions
                jr_acf[indx, -1, :] += jr_acf[indx, d, :]
        indx += 1

    return J_flux, jr_acf


@njit
def calc_vk(pos_data, vel_data, k_list):
    """
    Calculate the instantaneous longitudinal and transverse velocity fluctuations.

    Parameters
    ----------
    pos_data : numpy.ndarray
        Particles' position. Shape = ( ``no_dumps``, 3, ``tot_no_ptcls``)

    vel_data : numpy.ndarray
        Particles' velocities. Shape = ( ``no_dumps``, 3, ``tot_no_ptcls``)

    k_list : list
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``ka_counts``.
        Shape=(``no_ka_values``, 5)

    Returns
    -------
    vkt : numpy.ndarray
        Array containing longitudinal velocity fluctuations.

    vkt_i : numpy.ndarray
        Array containing transverse velocity fluctuations in the :math:`x` direction.

    vkt_j : numpy.ndarray
        Array containing transverse velocity fluctuations in the :math:`y` direction.

    vkt_k : numpy.ndarray
        Array containing transverse velocity fluctuations in the :math:`z` direction.

    """

    # Longitudinal
    vk = zeros(len(k_list), dtype=complex128)

    # Transverse
    vk_i = zeros(len(k_list), dtype=complex128)
    vk_j = zeros(len(k_list), dtype=complex128)
    vk_k = zeros(len(k_list), dtype=complex128)

    for ik, k_vec in enumerate(k_list):
        # Calculate the dot product and cross product between k, r, and v
        kr_i = 2.0 * pi * (k_vec[0] * pos_data[:, 0] + k_vec[1] * pos_data[:, 1] + k_vec[2] * pos_data[:, 2])
        k_dot_v = 2.0 * pi * (k_vec[0] * vel_data[:, 0] + k_vec[1] * vel_data[:, 1] + k_vec[2] * vel_data[:, 2])

        k_cross_v_i = 2.0 * pi * (k_vec[1] * vel_data[:, 2] - k_vec[2] * vel_data[:, 1])
        k_cross_v_j = -2.0 * pi * (k_vec[0] * vel_data[:, 2] - k_vec[2] * vel_data[:, 0])
        k_cross_v_k = 2.0 * pi * (k_vec[0] * vel_data[:, 1] - k_vec[1] * vel_data[:, 0])

        # Microscopic longitudinal current
        vk[ik] = (k_dot_v * exp(-1j * kr_i)).sum()
        # Microscopic transverse current
        vk_i[ik] = (k_cross_v_i * exp(-1j * kr_i)).sum()
        vk_j[ik] = (k_cross_v_j * exp(-1j * kr_i)).sum()
        vk_k[ik] = (k_cross_v_k * exp(-1j * kr_i)).sum()

    return vk, vk_i, vk_j, vk_k


def calc_vkt(fldr, slices, dump_step, species_np, k_list, verbose):
    r"""
    Calculate the longitudinal and transverse velocities fluctuations of all species.
    Longitudinal

    .. math::
        \lambda_A(\mathbf{k}, t) = \sum_i^{N_A} \mathbf{k} \cdot \mathbf{v}_{i}(t) \exp [ - i \mathbf{k} \cdot \mathbf{r}_{i}(t) ]

    Transverse

    .. math::
        \\tau_A(\mathbf{k}, t) = \sum_i^{N_A} \mathbf{k} \\times \mathbf{v}_{i}(t) \exp [ - i \mathbf{k} \cdot \mathbf{r}_{i}(t) ]

    where :math:`N_A` is the number of particles of species :math:`A`.

    Parameters
    ----------
    fldr : str
        Name of folder containing particles data.

    slices : tuple, int
        Initial, final step number of the slice, number of steps per slice.

    dump_step : int
        Timestep interval saving.

    species_np : numpy.ndarray
        Number of particles of each species.

    k_list : list
        List of :math: `k` vectors.

    Returns
    -------
    vkt : numpy.ndarray, complex
        Longitudinal velocity fluctuations.
        Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    vkt_perp_i : numpy.ndarray, complex
        Transverse velocity fluctuations along the :math:`x` axis.
        Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    vkt_perp_j : numpy.ndarray, complex
        Transverse velocity fluctuations along the :math:`y` axis.
        Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    vkt_perp_k : numpy.ndarray, complex
        Transverse velocity fluctuations along the :math:`z` axis.
        Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    """

    # Read particles' position for all times
    no_dumps = slices[2]
    vkt_par = zeros((len(species_np), no_dumps, len(k_list)), dtype=complex128)
    vkt_perp_i = zeros((len(species_np), no_dumps, len(k_list)), dtype=complex128)
    vkt_perp_j = zeros((len(species_np), no_dumps, len(k_list)), dtype=complex128)
    vkt_perp_k = zeros((len(species_np), no_dumps, len(k_list)), dtype=complex128)
    for it, dump in enumerate(
        tqdm(range(slices[0], slices[1], dump_step), desc="Timestep", position=1, disable=not verbose, leave=False)
    ):

        data = load_from_restart(fldr, dump)
        pos = data["pos"]
        vel = data["vel"]
        sp_start = 0
        sp_end = 0
        for i, sp in enumerate(species_np):
            sp_end += sp
            vkt_par[i, it, :], vkt_perp_i[i, it, :], vkt_perp_j[i, it, :], vkt_perp_k[i, it, :] = calc_vk(
                pos[sp_start:sp_end, :], vel[sp_start:sp_end], k_list
            )
            sp_start += sp

    return vkt_par, vkt_perp_i, vkt_perp_j, vkt_perp_k


def grad_expansion(x, rms, h_coeff):
    """
    Calculate the Grad expansion as given by eq.(5.97) in Liboff

    Parameters
    ----------
    x : numpy.ndarray
        Array of the scaled velocities

    rms : float
        RMS width of the Gaussian.

    h_coeff: numpy.ndarray
        Hermite coefficients without the division by factorial.

    Returns
    -------
    _ : numpy.ndarray
        Grad expansion.

    """
    gaussian = exp(-0.5 * (x / rms) ** 2) / (sqrt(2.0 * pi * rms**2))

    herm_coef = h_coeff / [factorial(i) for i in range(len(h_coeff))]
    hermite_series = hermite_e.hermeval(x, herm_coef)

    return gaussian * hermite_series


def calculate_herm_coeff(v, distribution, maxpower):
    r"""
    Calculate Hermite coefficients by integrating the velocity distribution function. That is

    .. math::
        a_i = \int_{-\\infty}^{\infty} dv \, He_i(v)f(v)

    Parameters
    ----------
    v : numpy.ndarray
        Range of velocities.

    distribution: numpy.ndarray
        Velocity histogram.

    maxpower: int
        Hermite order

    Returns
    -------
    coeff: numpy.ndarray
        Coefficients :math:`a_i`

    """

    coeff = zeros(maxpower + 1)

    for i in range(maxpower + 1):
        hc = zeros(1 + i)
        hc[-1] = 1.0
        Hp = hermite_e.hermeval(v, hc)
        coeff[i] = trapz(distribution * Hp, x=v)

    return coeff


def kspace_setup(box_lengths, angle_averaging, max_k_harmonics, max_aa_harmonics):
    """
    Calculate all allowed :math:`k` vectors.

    Parameters
    ----------
    max_k_harmonics : numpy.ndarray
        Number of harmonics in each direction.

    box_lengths : numpy.ndarray
        Length of each box's side.

    angle_averaging : str
        Flag for calculating all the possible `k` vector directions and magnitudes. Default = 'principal_axis'

    max_aa_harmonics : numpy.ndarray
        Maximum `k` harmonics in each direction for angle average.

    Returns
    -------
    k_arr : list
        List of all possible combination of :math:`(n_x, n_y, n_z)` with their corresponding magnitudes and indexes.

    k_counts : numpy.ndarray
        Number of occurrences of each triplet :math:`(n_x, n_y, n_z)` magnitude.

    k_unique : numpy.ndarray
        Magnitude of the unique allowed triplet :math:`(n_x, n_y, n_z)`.
    """
    if angle_averaging == "full":
        # The first value of k_arr = [0, 0, 0]
        first_non_zero = 1
        # Obtain all possible permutations of the wave number arrays
        k_arr = [
            array([i / box_lengths[0], j / box_lengths[1], k / box_lengths[2]])
            for i in range(max_k_harmonics[0] + 1)
            for j in range(max_k_harmonics[1] + 1)
            for k in range(max_k_harmonics[2] + 1)
        ]
        harmonics = [
            array([i, j, k], dtype=int)
            for i in range(max_k_harmonics[0] + 1)
            for j in range(max_k_harmonics[1] + 1)
            for k in range(max_k_harmonics[2] + 1)
        ]
    elif angle_averaging == "principal_axis":
        # The first value of k_arr = [1, 0, 0]
        first_non_zero = 0
        # Calculate the k vectors along the principal axis only
        k_arr = [array([i / box_lengths[0], 0, 0]) for i in range(1, max_k_harmonics[0] + 1)]
        harmonics = [array([i, 0, 0], dtype=int) for i in range(1, max_k_harmonics[0] + 1)]
        k_arr = np_append(k_arr, [array([0, i / box_lengths[1], 0]) for i in range(1, max_k_harmonics[1] + 1)], axis=0)
        harmonics = np_append(harmonics, [array([0, i, 0], dtype=int) for i in range(1, max_k_harmonics[1] + 1)], axis=0)

        k_arr = np_append(k_arr, [array([0, 0, i / box_lengths[2]]) for i in range(1, max_k_harmonics[2] + 1)], axis=0)
        harmonics = np_append(harmonics, [array([0, 0, i], dtype=int) for i in range(1, max_k_harmonics[2] + 1)], axis=0)

    elif angle_averaging == "custom":
        # The first value of k_arr = [0, 0, 0]
        first_non_zero = 1
        # Obtain all possible permutations of the wave number arrays up to max_aa_harmonics included
        k_arr = [
            array([i / box_lengths[0], j / box_lengths[1], k / box_lengths[2]])
            for i in range(max_aa_harmonics[0] + 1)
            for j in range(max_aa_harmonics[1] + 1)
            for k in range(max_aa_harmonics[2] + 1)
        ]

        harmonics = [
            array([i, j, k], dtype=int)
            for i in range(max_aa_harmonics[0] + 1)
            for j in range(max_aa_harmonics[1] + 1)
            for k in range(max_aa_harmonics[2] + 1)
        ]
        # Append the rest of k values calculated from principal axis
        k_arr = np_append(
            k_arr,
            [array([i / box_lengths[0], 0, 0]) for i in range(max_aa_harmonics[0] + 1, max_k_harmonics[0] + 1)],
            axis=0,
        )
        harmonics = np_append(
            harmonics,
            [array([i, 0, 0], dtype=int) for i in range(max_aa_harmonics[0] + 1, max_k_harmonics[0] + 1)],
            axis=0,
        )

        k_arr = np_append(
            k_arr,
            [array([0, i / box_lengths[1], 0]) for i in range(max_aa_harmonics[1] + 1, max_k_harmonics[1] + 1)],
            axis=0,
        )
        harmonics = np_append(
            harmonics,
            [array([0, i, 0], dtype=int) for i in range(max_aa_harmonics[1] + 1, max_k_harmonics[1] + 1)],
            axis=0,
        )

        k_arr = np_append(
            k_arr,
            [array([0, 0, i / box_lengths[2]]) for i in range(max_aa_harmonics[2] + 1, max_k_harmonics[2] + 1)],
            axis=0,
        )
        harmonics = np_append(
            harmonics,
            [array([0, 0, i], dtype=int) for i in range(max_aa_harmonics[2] + 1, max_k_harmonics[2] + 1)],
            axis=0,
        )
    # Compute wave number magnitude - don't use |k| (skipping first entry in k_arr)
    # The round off is needed to avoid ka value different beyond a certain significant digit. It will break other parts
    # of the code otherwise.
    k_mag = sqrt((array(k_arr) ** 2).sum(axis=1)[..., None])
    harm_mag = sqrt((array(harmonics) ** 2).sum(axis=1)[..., None])
    for i, k in enumerate(k_mag[:-1]):
        if abs(k - k_mag[i + 1]) < 2.0e-5:
            k_mag[i + 1] = k
    # Add magnitude to wave number array
    k_arr = concatenate((k_arr, k_mag), 1)
    # Add magnitude to wave number array
    harmonics = concatenate((harmonics, harm_mag), 1)

    # Sort from lowest to highest magnitude
    ind = argsort(k_arr[:, -1])
    k_arr = k_arr[ind]
    harmonics = harmonics[ind]

    # Count how many times a |k| value appears
    k_unique, k_counts = unique(k_arr[first_non_zero:, -1], return_counts=True)

    # Generate a 1D array containing index to be used in S array
    k_index = repeat(range(len(k_counts)), k_counts)[..., None]

    # Add index to k_array
    k_arr = concatenate((k_arr[int(first_non_zero) :, :], k_index), 1)
    harmonics = concatenate((harmonics[int(first_non_zero) :, :], k_index), 1)

    return k_arr, k_counts, k_unique, harmonics


def load_from_restart(fldr, it):
    """
    Load particles' data from dumps.

    Parameters
    ----------
    fldr : str
        Folder containing dumps.

    it : int
        Timestep to load.

    Returns
    -------
    data : dict
        Particles' data.
    """

    file_name = os_path_join(fldr, "checkpoint_" + str(it) + ".npz")
    data = load(file_name, allow_pickle=True)
    return data


def plot_labels(xdata, ydata, xlbl, ylbl, units):
    """
    Create plot labels with correct units and prefixes.

    Parameters
    ----------
    xdata: numpy.ndarray
        X values.

    ydata: numpy.ndarray
        Y values.

    xlbl: str
        String of the X quantity.

    ylbl: str
        String of the Y quantity.

    units: str
        'cgs' or 'mks'.

    Returns
    -------
    xmultiplier: float
        Scaling factor for X data.

    ymultiplier: float
        Scaling factor for Y data.

    xprefix: str
        Prefix for X units label

    yprefix: str
         Prefix for Y units label.

    xlabel: str
        X label units.

    ylabel: str
        Y label units.

    """
    if isinstance(xdata, (ndarray, Series)):
        xmax = xdata.max()
    else:
        xmax = xdata

    if isinstance(ydata, (ndarray, Series)):
        ymax = ydata.max()
    else:
        ymax = ydata

    # Find the correct Units
    units_dict = UNITS[1] if units == "cgs" else UNITS[0]

    if units == "cgs" and xlbl == "Length":
        xmax *= 1e2

    if units == "cgs" and ylbl == "Length":
        ymax *= 1e2

    # Use scientific notation. This returns a string
    x_str = format_float_scientific(xmax)
    y_str = format_float_scientific(ymax)

    # Grab the exponent
    x_exp = 10.0 ** (float(x_str[x_str.find("e") + 1 :]))
    y_exp = 10.0 ** (float(y_str[y_str.find("e") + 1 :]))

    # Find the units' prefix by looping over the possible values
    xprefix = "none"
    xmul = -1.5
    # This is a 10 multiplier
    i = 1.0
    while xmul < 0:
        for key, value in PREFIXES.items():
            ratio = i * x_exp / value
            if abs(ratio - 1) < 1.0e-6:
                xprefix = key
                xmul = 1 / value
        i /= 10
    # find the prefix
    yprefix = "none"
    ymul = -1.5
    i = 1.0
    while ymul < 0:
        for key, value in PREFIXES.items():
            ratio = i * y_exp / value
            if abs(ratio - 1) < 1.0e-6:
                yprefix = key
                ymul = 1 / value
        i /= 10.0

    if "Energy" in ylbl:
        yname = "Energy"
    else:
        yname = ylbl

    if "Pressure" in ylbl:
        yname = "Pressure"
    else:
        yname = ylbl

    if yname in units_dict:
        ylabel = " [" + yprefix + units_dict[yname] + "]"
    else:
        ylabel = ""

    if "Energy" in xlbl:
        xname = "Energy"
    else:
        xname = xlbl

    if "Pressure" in xlbl:
        xname = "Pressure"
    else:
        xname = xlbl

    if xname in units_dict:
        xlabel = " [" + xprefix + units_dict[xname] + "]"
    else:
        xlabel = ""

    return xmul, ymul, xprefix, yprefix, xlabel, ylabel


def col_mapper(keys, vals):
    return dict(zip(keys, vals))


def make_gaussian_plot(time, data, observable_name, units):
    fig = plt.figure(figsize=(19, 6))
    gs = GridSpec(4, 8)

    main_plot = fig.add_subplot(gs[1:4, 0:3])
    delta_plot = fig.add_subplot(gs[0, 0:3], sharex=main_plot)
    hist_plot = fig.add_subplot(gs[1:4, 3], sharey=main_plot)

    # Remove the label from delta and hist plots
    delta_plot.tick_params(axis="x", labelbottom=False)
    hist_plot.tick_params(axis="y", labelleft=False)

    # Grab the color line list from the plt cycler. I will use this in the hist plots
    color_from_cycler = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Calculate plot's labels and multipliers
    time_mul, temp_mul, time_prefix, temp_prefix, time_lbl, temp_lbl = plot_labels(
        time, data, "Time", observable_name, units
    )

    # Rescale quantities
    time = time_mul * time
    data_plot = temp_mul * data

    data_mean = data_plot.mean()
    data_std = data_plot.std()

    # moving average
    data_cumavg = data_plot.expanding().mean()

    # deviation and its moving average
    delta_data = (data_plot - data_mean) * 100 / data_mean
    delta_data_cum_avg = delta_data.expanding().mean()

    # Temperature Main plot
    main_plot.plot(time, data_plot, alpha=0.7)
    main_plot.plot(time, data_cumavg, label="Moving Average")
    main_plot.axhline(data_mean, ls="--", c="r", alpha=0.7, label="Mean")
    main_plot.legend(loc="best")
    main_plot.set(ylabel=f"{observable_name} {temp_lbl}", xlabel=f"Time {time_lbl}")

    # Temperature Deviation plot
    delta_plot.plot(time, delta_data, alpha=0.5)
    delta_plot.plot(time, delta_data_cum_avg, alpha=0.8)
    delta_plot.set(ylabel=r"Deviation [%]")

    dist_desired = scp_stats.norm(loc=data_mean, scale=data_std)
    # Histogram plot

    # sns_histplot(y=Temperature, bins="auto", stat="density", alpha=0.75, legend="False", ax=T_hist_plot)
    hist_plot.hist(data_plot, bins="fd", density=True, alpha=0.75, orientation="horizontal")
    hist_plot.plot(dist_desired.pdf(data_plot.sort_values()), data_plot.sort_values(), alpha=0.7, label="Gaussian")
    hist_plot.grid(False)
    hist_plot.legend()

    return fig, (delta_plot, main_plot, hist_plot)
