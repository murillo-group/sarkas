"""
Module for calculating physical quantities from Sarkas checkpoints.
"""
from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from numba import njit
from matplotlib.gridspec import GridSpec

import os
import pickle
import copy as pycopy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as scp_stats

from sarkas.utilities.timing import SarkasTimer
from sarkas.utilities.io import num_sort
from sarkas.utilities.maths import correlationfunction

UNITS = [
    # MKS Units
    {
        "Energy": "J",
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
        "Bulk Viscosity": r"kg/m s",
        "Shear Viscosity": r"kg/m s",
        "none": "",
    },
    # CGS Units
    {
        "Energy": "erg",
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
        "Bulk Viscosity": r"g/ cm s",
        "Shear Viscosity": r"g/ cm s",
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
    Calculate the observable (and its autocorrelation function). See class doc for exact quantities. \n 
    The data of each slice is saved in hierarchical dataframes,
    :attr:`~.dataframe_slices` (:attr:`~.dataframe_acf_slices`). \n
    
    The sliced averaged data is saved in other hierarchical dataframes,
    :attr:`~.dataframe` (:attr:`~.dataframe_acf_slices`).
     
    """
    return func


def setup_doc(func):
    func.__doc__ = """
    Assign attributes from simulation's parameters.

    Parameters
    ----------
    params : sarkas.core.Parameters
        Simulation's parameters.

    phase : str, optional
        Phase to compute. Default = 'production'.

    no_slices : int, optional
        Number of independent runs inside a long simulation. Default = 1.

    **kwargs :
        These will overwrite any :attr:`sarkas.core.Parameters`
        or default :attr:`sarkas.tools.observables.Observable`
        attributes and/or add new ones.

   """
    return func


def arg_update_doc(func):
    func.__doc__ = """Update observable specific attributes and call :meth:`~.update_finish` to save info."""
    return func


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
        It is either :attr:`sarkas.core.Parameters.prod_dump_step` or :attr:`sarkas.core.Parameters.eq_dump_step`.

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
        self.saving_dir = None
        self.phase = "production"
        self.multi_run_average = False
        self.dimensional_average = False
        self.runs = 1
        self.no_slices = 1
        self.slice_steps = None
        self.screen_output = True
        self.timer = SarkasTimer()
        self.k_observable = False
        self.kw_observable = False
        self.acf_observable = False
        self.dataframe_slices = None
        self.dataframe = None
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

    def from_dict(self, input_dict: dict):
        """
        Update attributes from input dictionary.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """
        self.__dict__.update(input_dict)

    def setup_init(self, params, phase: str = None, no_slices: int = None):
        """
        Assign Observables attributes and copy the simulation's parameters.

        Parameters
        ----------
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

        # The dict update could overwrite the names
        name = self.__name__
        long_name = self.__long_name__

        self.__dict__.update(params.__dict__)

        # Restore the correct names
        self.__name__ = name
        self.__long_name__ = long_name

        if self.k_observable:
            # Check for k space information.
            if not hasattr(self, "angle_averaging"):
                self.angle_averaging = "principal_axis"
                self.max_aa_harmonics = np.array([0, 0, 0])

            if self.angle_averaging == "custom":
                if not hasattr(self, "max_aa_ka_value"):
                    if self.max_aa_harmonics is None:
                        raise AttributeError("max_aa_harmonics and max_aa_ka_value not defined.")
                elif not hasattr(self, "max_aa_harmonics"):
                    if self.max_aa_ka_value is None:
                        raise AttributeError("max_aa_harmonics and max_aa_ka_value not defined.")

            # More checks on k attributes and initialization of k vectors

            # Dev Notes:
            #           Make sure that max_k_harmonics and max_aa_harmonics are defined once this if is done.
            #           The user can either define max_k_harmonics or max_ka_value
            #           Based on this choice the user can define max_aa_harmonics or max_aa_ka_value

            if hasattr(self, "max_k_harmonics"):
                # Convert max_k_harmonics to a numpy array
                if isinstance(self.max_k_harmonics, np.ndarray) == 0:
                    self.max_k_harmonics = np.ones(3, dtype=int) * self.max_k_harmonics

                # Calculate max_aa_harmonics based on the choice of angle averaging and inputs
                if self.angle_averaging == "full":
                    self.max_aa_harmonics = np.copy(self.max_k_harmonics)

                elif self.angle_averaging == "custom":
                    # Check if the user has defined the max_aa_harmonics
                    if self.max_aa_ka_value:
                        nx = int(self.max_aa_ka_value * self.box_lengths[0] / (2.0 * np.pi * self.a_ws * np.sqrt(3.0)))
                        self.max_aa_harmonics = np.array([nx, nx, nx])
                    # else max_aa_harmonics is user defined
                elif self.angle_averaging == "principal_axis":
                    self.max_aa_harmonics = np.array([0, 0, 0])

            elif hasattr(self, "max_ka_value"):
                # Calculate max_k_harmonics from max_ka_value

                # Check for angle_averaging choice
                if self.angle_averaging == "full":
                    # The maximum value is calculated assuming that max nx = max ny = max nz
                    # ka_max = 2pi a/L sqrt( nx^2 + ny^2 + nz^2) = 2pi a/L nx sqrt(3)
                    nx = int(self.max_ka_value * self.box_lengths[0] / (2.0 * np.pi * self.a_ws * np.sqrt(3.0)))
                    self.max_k_harmonics = np.array([nx, nx, nx])
                    self.max_aa_harmonics = np.array([nx, nx, nx])

                elif self.angle_averaging == "custom":
                    # ka_max = 2pi a/L sqrt( nx^2 + 0 + 0) = 2pi a/L nx
                    nx = int(self.max_ka_value * self.box_lengths[0] / (2.0 * np.pi * self.a_ws))
                    self.max_k_harmonics = np.array([nx, nx, nx])
                    # Check if the user has defined the max_aa_harmonics
                    if self.max_aa_ka_value:
                        nx = int(self.max_aa_ka_value * self.box_lengths[0] / (2.0 * np.pi * self.a_ws * np.sqrt(3.0)))
                        self.max_aa_harmonics = np.array([nx, nx, nx])
                    # else max_aa_harmonics is user defined
                elif self.angle_averaging == "principal_axis":
                    # ka_max = 2pi a/L sqrt( nx^2 + 0 + 0) = 2pi a/L nx
                    nx = int(self.max_ka_value * self.box_lengths[0] / (2.0 * np.pi * self.a_ws))
                    self.max_k_harmonics = np.array([nx, nx, nx])
                    self.max_aa_harmonics = np.array([0, 0, 0])

            else:
                # Executive decision
                self.max_k_harmonics = np.array([5, 5, 5])
                self.max_aa_harmonics = np.array([0, 0, 0])
                self.angle_averaging = "principal_axis"

            # Calculate the maximum ka value based on user's choice of angle_averaging
            # Dev notes: Make sure max_ka_value, max_aa_ka_value are defined when this if is done
            if self.angle_averaging == "full":
                self.max_ka_value = 2.0 * np.pi * self.a_ws * np.linalg.norm(self.max_k_harmonics / self.box_lengths)
                self.max_aa_ka_value = 2.0 * np.pi * self.a_ws * np.linalg.norm(self.max_k_harmonics / self.box_lengths)

            elif self.angle_averaging == "principal_axis":
                self.max_ka_value = 2.0 * np.pi * self.a_ws * self.max_k_harmonics[0] / self.box_lengths[0]
                self.max_aa_ka_value = 0.0

            elif self.angle_averaging == "custom":
                self.max_aa_ka_value = 2.0 * np.pi * self.a_ws * np.linalg.norm(self.max_aa_harmonics / self.box_lengths)
                self.max_ka_value = 2.0 * np.pi * self.a_ws * self.max_k_harmonics[0] / self.box_lengths[0]

            # Create paths for files
            self.k_space_dir = os.path.join(self.postprocessing_dir, "k_space_data")
            self.k_file = os.path.join(self.k_space_dir, "k_arrays.npz")
            self.nkt_hdf_file = os.path.join(self.k_space_dir, "nkt.h5")
            self.vkt_hdf_file = os.path.join(self.k_space_dir, "vkt.h5")

        # Get the number of independent observables if multi-species
        self.no_obs = int(self.num_species * (self.num_species + 1) / 2)

        # Get the total number of dumps by looking at the files in the directory
        self.prod_no_dumps = len(os.listdir(self.prod_dump_dir))
        self.eq_no_dumps = len(os.listdir(self.eq_dump_dir))

        # Check for magnetized plasma options
        if self.magnetized and self.electrostatic_equilibration:
            self.mag_no_dumps = len(os.listdir(self.mag_dump_dir))

        # Assign dumps variables based on the choice of phase
        if self.phase == "equilibration":
            self.no_dumps = self.eq_no_dumps
            self.dump_step = self.eq_dump_step
            self.no_steps = self.equilibration_steps
            self.dump_dir = self.eq_dump_dir

        elif self.phase == "production":
            self.no_dumps = self.prod_no_dumps
            self.dump_step = self.prod_dump_step
            self.no_steps = self.production_steps
            self.dump_dir = self.prod_dump_dir

        elif self.phase == "magnetization":
            self.no_dumps = self.mag_no_dumps
            self.dump_step = self.mag_dump_step
            self.no_steps = self.magnetization_steps
            self.dump_dir = self.mag_dump_dir

        # Needed for preprocessing pretty print
        self.slice_steps = (
            int(self.no_steps / self.dump_step / self.no_slices)
            if self.no_dumps < self.no_slices
            else int(self.no_dumps / self.no_slices)
        )

        # Array containing the start index of each species.
        self.species_index_start = np.array([0, *np.cumsum(self.species_num)], dtype=int)

    def create_dirs_filenames(self):
        # Saving Directory
        saving_dir = os.path.join(self.postprocessing_dir, self.__long_name__.replace(" ", ""))
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        # Filenames and strings
        self.filename_hdf = os.path.join(self.saving_dir, self.__long_name__.replace(" ", "") + "_" + self.job_id + ".h5")

        self.filename_hdf_slices = os.path.join(
            self.saving_dir, self.__long_name__.replace(" ", "") + "_slices_" + self.job_id + ".h5"
        )

        if self.acf_observable:
            self.filename_hdf_acf = os.path.join(
                self.saving_dir, self.__long_name__.replace(" ", "") + "ACF_" + self.job_id + ".h5"
            )

            self.filename_hdf_acf_slices = os.path.join(
                self.saving_dir, self.__long_name__.replace(" ", "") + "ACF_slices_" + self.job_id + ".h5"
            )

    def parse(self):
        """
        Grab the pandas dataframe from the saved csv file. If file does not exist call ``compute``.
        """
        if self.k_observable:
            try:
                self.dataframe = pd.read_hdf(self.filename_hdf, mode="r", index_col=False)

                k_data = np.load(self.k_file)
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
                    self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
                else:
                    self.dataframe = pd.read_hdf(self.filename_hdf, mode="r", index_col=False)

            except FileNotFoundError:
                if hasattr(self, "filename_csv"):
                    data_file = self.filename_csv
                else:
                    data_file = self.filename_hdf
                print("\nData file not found! \n {}".format(data_file))
                print("\nComputing Observable now ...")
                self.compute()

            if hasattr(self, "dataframe_slices"):
                self.dataframe_slices = pd.read_hdf(self.filename_hdf_slices, mode="r", index_col=False)

            if self.acf_observable:
                self.dataframe_acf = pd.read_hdf(self.filename_hdf_acf, mode="r", index_col=False)
                self.dataframe_acf_slices = pd.read_hdf(self.filename_hdf_acf_slices, mode="r", index_col=False)

    def parse_k_data(self):
        """Read in the precomputed Fourier space data. Recalculate if not correct."""

        try:
            k_data = np.load(self.k_file)
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
        self.ka_values = 2.0 * np.pi * k_unique * self.a_ws
        self.k_values = 2.0 * np.pi * k_unique
        self.no_ka_values = len(self.ka_values)

        # Check if the writing folder exist
        if not (os.path.exists(self.k_space_dir)):
            os.mkdir(self.k_space_dir)

        # Write the npz file
        np.savez(
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
                with pd.HDFStore(self.nkt_hdf_file, mode="r") as nkt_hfile:
                    metadata = nkt_hfile.get_storer("nkt").attrs.metadata

                if metadata["no_slices"] == self.no_slices:
                    # Check for the correct number of k values
                    if metadata["angle_averaging"] == self.angle_averaging:
                        # Check for the correct max harmonics
                        comp = self.max_k_harmonics == metadata["max_k_harmonics"]
                        if not comp.all():
                            self.calc_kt_data(nkt_flag=True)
                    else:
                        self.calc_kt_data(nkt_flag=True)
                else:
                    self.calc_kt_data(nkt_flag=True)

                # elif metadata['max_k_harmonics']
                #
                # if self.angle_averaging == nkt_data["angle_averaging"]:
                #
                #     comp = self.max_k_harmonics == nkt_data["max_harmonics"]
                #     if not comp.all():
                #         self.calc_kt_data(nkt_flag=True)
                # else:
                #     self.calc_kt_data(nkt_flag=True)

            except OSError:
                self.calc_kt_data(nkt_flag=True)

        if vkt_flag:

            try:
                # Check that what was already calculated is correct
                with pd.HDFStore(self.vkt_hdf_file, mode="r") as vkt_hfile:
                    metadata = vkt_hfile.get_storer("vkt").attrs.metadata

                if metadata["no_slices"] == self.no_slices:
                    # Check for the correct number of k values
                    if metadata["angle_averaging"] == self.angle_averaging:
                        # Check for the correct max harmonics
                        comp = self.max_k_harmonics == metadata["max_k_harmonics"]
                        if not comp.all():
                            self.calc_kt_data(vkt_flag=True)
                    else:
                        self.calc_kt_data(vkt_flag=True)
                else:
                    self.calc_kt_data(vkt_flag=True)

            except OSError:
                self.calc_kt_data(vkt_flag=True)

    def calc_kt_data(self, nkt_flag: bool = False, vkt_flag: bool = False):
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
        start_slice = 0
        end_slice = self.slice_steps * self.dump_step
        if nkt_flag:
            nkt_dataframe = pd.DataFrame()
            tinit = self.timer.current()
            for isl in range(self.no_slices):
                print("\nCalculating n(k,t) for slice {}/{}.".format(isl + 1, self.no_slices))
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
                    nkt_dataframe = pd.concat([nkt_dataframe, pd.DataFrame(nkt[isp, :, :], columns=df_columns)], axis=1)

            tuples = [tuple(c.split("_")) for c in nkt_dataframe.columns]
            nkt_dataframe.columns = pd.MultiIndex.from_tuples(tuples, names=["slices", "species", "harmonics"])

            # Sample nkt_dataframe
            # slices slice 1
            # species H
            # harmonics k = [0, 0, 1] | k = [0, 1, 0] | ...

            # Save the data and append metadata
            if os.path.exists(self.nkt_hdf_file):
                os.remove(self.nkt_hdf_file)

            hfile = pd.HDFStore(self.nkt_hdf_file, mode="w")
            hfile.put("nkt", nkt_dataframe)
            # This metadata is needed to check if I need to recalculate
            metadata = {
                "no_slices": self.no_slices,
                "max_k_harmonics": self.max_k_harmonics,
                "angle_averaging": self.angle_averaging,
            }

            hfile.get_storer("nkt").attrs.metadata = metadata
            hfile.close()

            tend = self.timer.current()
            self.time_stamp("n(k,t) Calculation", self.timer.time_division(tend - tinit))

        if vkt_flag:
            vkt_dataframe = pd.DataFrame()
            tinit = self.timer.current()
            for isl in range(self.no_slices):
                print(
                    "\nCalculating longitudinal and transverse "
                    "velocity fluctuations v(k,t) for slice {}/{}.".format(isl + 1, self.no_slices)
                )
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
                    vkt_dataframe = pd.concat([vkt_dataframe, pd.DataFrame(vkt[isp, :, :], columns=df_columns)], axis=1)
                    df_columns = [
                        slc_column
                        + "_Transverse i_{}_k = [{}, {}, {}]".format(sp_name, *self.k_harmonics[ik, :-2].astype(int))
                        for ik in range(len(self.k_harmonics))
                    ]
                    vkt_dataframe = pd.concat([vkt_dataframe, pd.DataFrame(vkt_i[isp, :, :], columns=df_columns)], axis=1)
                    df_columns = [
                        slc_column
                        + "_Transverse j_{}_k = [{}, {}, {}]".format(sp_name, *self.k_harmonics[ik, :-2].astype(int))
                        for ik in range(len(self.k_harmonics))
                    ]
                    vkt_dataframe = pd.concat([vkt_dataframe, pd.DataFrame(vkt_j[isp, :, :], columns=df_columns)], axis=1)

                    df_columns = [
                        slc_column
                        + "_Transverse k_{}_k = [{}, {}, {}]".format(sp_name, *self.k_harmonics[ik, :-2].astype(int))
                        for ik in range(len(self.k_harmonics))
                    ]
                    vkt_dataframe = pd.concat([vkt_dataframe, pd.DataFrame(vkt_k[isp, :, :], columns=df_columns)], axis=1)

            # Full string: slice 1_Longitudinal_He_k = [1, 0, 0]
            tuples = [tuple(c.split("_")) for c in vkt_dataframe.columns]
            vkt_dataframe.columns = pd.MultiIndex.from_tuples(
                tuples, names=["slices", "species", "direction", "harmonics"]
            )

            # Sample nkt_dataframe
            # slices slice 1
            # species H
            # direction Longitudinal/Transverse
            # harmonics k = [0, 0, 1] | k = [0, 1, 0] | ...

            if os.path.exists(self.vkt_hdf_file):
                os.remove(self.vkt_hdf_file)
            # Save the data and append metadata
            hfile = pd.HDFStore(self.vkt_hdf_file, mode="w")
            hfile.put("vkt", vkt_dataframe)
            # This metadata is needed to check if I need to recalculate
            metadata = {
                "no_slices": self.no_slices,
                "max_k_harmonics": self.max_k_harmonics,
                "angle_averaging": self.angle_averaging,
            }

            hfile.get_storer("vkt").attrs.metadata = metadata
            hfile.close()

            tend = self.timer.current()
            self.time_stamp("v(k,t) Calculation", self.timer.time_division(tend - tinit))

            # np.savez(self.vkt_file + '_slice_' + str(isl) + '.npz',
            #          longitudinal=vkt,
            #          transverse_i=vkt_i,
            #          transverse_j=vkt_j,
            #          transverse_k=vkt_k,
            #          max_harmonics=self.max_k_harmonics,
            #          angle_averaging=self.angle_averaging)

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

        # Grab the data
        # self.parse()
        # Make a copy of the dataframe for plotting

        plot_dataframe = self.dataframe.copy()

        if scaling:
            if isinstance(scaling, tuple):
                plot_dataframe.iloc[:, 0] /= scaling[0]
                plot_dataframe[kwargs["y"]] /= scaling[1]
            else:
                plot_dataframe.iloc[:, 0] /= scaling

        # Autocorrelation function renormalization
        if acf:
            for i, col in enumerate(plot_dataframe.columns[1:], 1):
                plot_dataframe[col] /= plot_dataframe[col].iloc[0]
            kwargs["logx"] = True
            # kwargs['xlabel'] = 'Time difference'

        # if self.__class__.__name__ == 'StaticStructureFactor':
        #     errorbars = plot_dataframe.copy()
        #     for i, col in enumerate(self.dataframe.columns):
        #         if col[-8:] == 'Errorbar':
        #             errorbars.drop(col, axis=1, inplace=True)
        #             errorbars.rename({col: col[:-9]}, axis=1, inplace=True)
        #             plot_dataframe.drop(col, axis=1, inplace=True)
        #     kwargs['yerr'] = errorbars
        #

        axes_handle = plot_dataframe.plot(x=plot_dataframe.columns[0], **kwargs)

        fig = axes_handle.figure
        fig.tight_layout()

        # Saving
        if figname:
            fig.savefig(os.path.join(self.saving_dir, figname + "_" + self.job_id + ".png"))
        else:
            fig.savefig(os.path.join(self.saving_dir, "Plot_" + self.__name__ + "_" + self.job_id + ".png"))

        if show:
            fig.show()

        return axes_handle

    def time_stamp(self, message: str, timing: tuple):
        """Print to screen the elapsed time of the calculation.

        Parameters
        ----------
        message : str
            Message to print.

        timing : tuple
            Time in hrs, min, sec, msec, usec, nsec.

        """

        t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = timing

        if t_hrs == 0 and t_min == 0 and t_sec <= 2:
            print_message = "\n{} Time: {} sec {} msec {} usec {} nsec".format(
                message, int(t_sec), int(t_msec), int(t_usec), int(t_nsec)
            )

        else:
            print_message = "\n{} Time: {} hrs {} min {} sec".format(message, int(t_hrs), int(t_min), int(t_sec))

        # logging.info(print_message)
        print(print_message)

    def save_hdf(self):

        # Create the columns for the HDF df
        if not self.k_observable:
            if not isinstance(self.dataframe_slices.columns, pd.MultiIndex):
                self.dataframe_slices.columns = pd.MultiIndex.from_tuples(
                    [tuple(c.split("_")) for c in self.dataframe_slices.columns]
                )

        if not isinstance(self.dataframe.columns, pd.MultiIndex):
            self.dataframe.columns = pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in self.dataframe.columns])

        # Sort the index for speed
        # see https://stackoverflow.com/questions/54307300/what-causes-indexing-past-lexsort-depth-warning-in-pandas
        self.dataframe = self.dataframe.sort_index()
        self.dataframe_slices = self.dataframe_slices.sort_index()

        # TODO: Fix this hack. We should be able to add data to HDF instead of removing it and rewriting it.
        # Save the data.
        if os.path.exists(self.filename_hdf_slices):
            os.remove(self.filename_hdf_slices)
        self.dataframe_slices.to_hdf(self.filename_hdf_slices, mode="w", key=self.__name__)

        if os.path.exists(self.filename_hdf):
            os.remove(self.filename_hdf)
        self.dataframe.to_hdf(self.filename_hdf, mode="w", key=self.__name__)

        if self.acf_observable:

            if not isinstance(self.dataframe_acf.columns, pd.MultiIndex):
                self.dataframe_acf.columns = pd.MultiIndex.from_tuples(
                    [tuple(c.split("_")) for c in self.dataframe_acf.columns]
                )

            if not isinstance(self.dataframe_acf_slices.columns, pd.MultiIndex):
                self.dataframe_acf_slices.columns = pd.MultiIndex.from_tuples(
                    [tuple(c.split("_")) for c in self.dataframe_acf_slices.columns]
                )

            self.dataframe_acf = self.dataframe_acf.sort_index()
            self.dataframe_acf_slices = self.dataframe_acf_slices.sort_index()

            if os.path.exists(self.filename_hdf_acf):
                os.remove(self.filename_hdf_acf)
            self.dataframe_acf.to_hdf(self.filename_hdf_acf, mode="w", key=self.__name__)

            if os.path.exists(self.filename_hdf_acf_slices):
                os.remove(self.filename_hdf_acf_slices)
            self.dataframe_acf_slices.to_hdf(self.filename_hdf_acf_slices, mode="w", key=self.__name__)

    def save_pickle(self):
        """Save the observable's info into a pickle file."""
        self.filename_pickle = os.path.join(self.saving_dir, self.__long_name__.replace(" ", "") + ".pickle")
        pickle_file = open(self.filename_pickle, "wb")
        pickle.dump(self, pickle_file)
        pickle_file.close()

    def read_pickle(self):
        """Read the observable's info from the pickle file."""
        self.filename_pickle = os.path.join(self.saving_dir, self.__long_name__.replace(" ", "") + ".pickle")
        with open(self.filename_pickle, "rb") as pkl_data:
            data = pickle.load()
        self.from_dict(data.__dict__)

    def update_finish(self):
        """Update the :attr:`~.slice_steps`, CCF's and DSF's attributes, and save pickle file with observable's info.

        Notes
        -----
        The information is saved without the dataframe(s).

        """
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

            self.w_min = 2.0 * np.pi / (self.slice_steps * self.dt * self.dump_step)
            self.w_max = np.pi / dt_r  # Half because np.fft calculates negative and positive frequencies
            self.frequencies = 2.0 * np.pi * np.fft.fftfreq(self.slice_steps, self.dt * self.dump_step)
            self.frequencies = np.fft.fftshift(self.frequencies)

        self.create_dirs_filenames()
        self.save_pickle()

        # This re-initialization of the dataframe is needed to avoid len mismatch conflicts when re-calculating
        self.dataframe = pd.DataFrame()
        self.dataframe_slices = pd.DataFrame()

        if self.acf_observable:
            self.dataframe_acf = pd.DataFrame()
            self.dataframe_acf_slices = pd.DataFrame()


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
        Length of :attr:`~.ka_values` array.

    """

    def __init__(self):
        super().__init__()
        self.__name__ = "ccf"
        self.__long_name__ = "Current Correlation Function"
        self.k_observable = True
        self.kw_observable = True

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

        self.parse_kt_data(nkt_flag=False, vkt_flag=True)

        tinit = self.timer.current()

        vkt_df = pd.read_hdf(self.vkt_hdf_file, mode="r", key="vkt")
        # Containers
        vkt = np.zeros((self.num_species, self.slice_steps, len(self.k_list)), dtype=np.complex128)
        vkt_i = np.zeros((self.num_species, self.slice_steps, len(self.k_list)), dtype=np.complex128)
        vkt_j = np.zeros((self.num_species, self.slice_steps, len(self.k_list)), dtype=np.complex128)
        vkt_k = np.zeros((self.num_species, self.slice_steps, len(self.k_list)), dtype=np.complex128)

        for isl in range(self.no_slices):

            # Put data in the container to pass
            for sp, sp_name in enumerate(self.species_names):
                vkt[sp] = np.array(vkt_df["slice {}".format(isl + 1)]["Longitudinal"][sp_name])
                vkt_i[sp] = np.array(vkt_df["slice {}".format(isl + 1)]["Transverse i"][sp_name])
                vkt_j[sp] = np.array(vkt_df["slice {}".format(isl + 1)]["Transverse j"][sp_name])
                vkt_k[sp] = np.array(vkt_df["slice {}".format(isl + 1)]["Transverse k"][sp_name])

            # Calculate Lkw and Tkw
            Lkw = calc_Skw(vkt, self.k_list, self.species_num, self.slice_steps, self.dt, self.dump_step)
            Tkw_i = calc_Skw(vkt_i, self.k_list, self.species_num, self.slice_steps, self.dt, self.dump_step)
            Tkw_j = calc_Skw(vkt_j, self.k_list, self.species_num, self.slice_steps, self.dt, self.dump_step)
            Tkw_k = calc_Skw(vkt_k, self.k_list, self.species_num, self.slice_steps, self.dt, self.dump_step)

            Tkw = (Tkw_i + Tkw_j + Tkw_k) / 3.0

            # Lkw_tot += Lkw / self.no_slices
            # Tkw_tot += Tkw / self.no_slices
            # Create the dataframe's column names
            slc_column = "slice {}".format(isl + 1)
            ka_columns = ["ka = {:.6f}".format(ka) for ik, ka in enumerate(self.ka_values)]

            # Save the full Lkw into a Dataframe
            sp_indx = 0
            for i, sp1 in enumerate(self.species_names):
                for j, sp2 in enumerate(self.species_names[i:]):
                    columns = [
                        "Longitudinal_{}-{}_".format(sp1, sp2)
                        + slc_column
                        + "_{}_k = [{}, {}, {}]".format(
                            ka_columns[int(self.k_harmonics[ik, -1])], *self.k_harmonics[ik, :-2].astype(int)
                        )
                        # Final string : Longitudinal_H-He_slice 1_ka = 0.123456_k = [0, 0, 1]
                        for ik in range(len(self.k_harmonics))
                    ]
                    self.dataframe_slices = pd.concat(
                        [self.dataframe_slices, pd.DataFrame(Lkw[sp_indx, :, :].T, columns=columns)], axis=1
                    )

                    columns = [
                        "Transverse_{}-{}_".format(sp1, sp2)
                        + slc_column
                        + "_{}_k = [{}, {}, {}]".format(
                            ka_columns[int(self.k_harmonics[ik, -1])], *self.k_harmonics[ik, :-2].astype(int)
                        )
                        # Final string : Transverse_H-He_slice 1_ka = 0.123456_k = [0, 0, 1]
                        for ik in range(len(self.k_harmonics))
                    ]
                    self.dataframe_slices = pd.concat(
                        [self.dataframe_slices, pd.DataFrame(Tkw[sp_indx, :, :].T, columns=columns)], axis=1
                    )

                    sp_indx += 1

        # Create the MultiIndex
        tuples = [tuple(c.split("_")) for c in self.dataframe_slices.columns]
        self.dataframe_slices.columns = pd.MultiIndex.from_tuples(
            tuples, names=["direction", "species", "slices", "ka_value", "k_harmonics"]
        )
        # Now the actual dataframe
        self.dataframe[" _ _ _ _Frequencies"] = self.frequencies
        # Take the mean and std and store them into the dataframe to return
        for sp1, sp1_name in enumerate(self.species_names):
            for sp2, sp2_name in enumerate(self.species_names[sp1:], sp1):
                comp_name = "{}-{}".format(sp1_name, sp2_name)
                ### LONGITUDINAL
                # Rename the columns with values of ka
                ka_columns = [
                    "Longitudinal_" + comp_name + "_Mean_ka{} = {:.4f}".format(ik + 1, ka)
                    for ik, ka in enumerate(self.ka_values)
                ]

                # Mean: level = 1 corresponds to averaging all the k harmonics with the same magnitude
                df_mean = self.dataframe_slices["Longitudinal"][comp_name].mean(level=1, axis="columns")
                df_mean = df_mean.rename(col_mapper(df_mean.columns, ka_columns), axis=1)
                # Std
                ka_columns = [
                    "Longitudinal_" + comp_name + "_Std_ka{} = {:.4f}".format(ik + 1, ka)
                    for ik, ka in enumerate(self.ka_values)
                ]
                df_std = self.dataframe_slices["Longitudinal"][comp_name].std(level=1, axis="columns")
                df_std = df_std.rename(col_mapper(df_std.columns, ka_columns), axis=1)

                self.dataframe = pd.concat([self.dataframe, df_mean, df_std], axis=1)

                ### Transverse
                # Rename the columns with values of ka
                ka_columns = [
                    "Transverse_" + comp_name + "_Mean_ka{} = {:.4f}".format(ik + 1, ka)
                    for ik, ka in enumerate(self.ka_values)
                ]

                # Mean: level = 1 corresponds to averaging all the k harmonics with the same magnitude
                tdf_mean = self.dataframe_slices["Transverse"][comp_name].mean(level=1, axis="columns")
                tdf_mean = tdf_mean.rename(col_mapper(tdf_mean.columns, ka_columns), axis=1)
                # Std
                ka_columns = [
                    "Transverse_" + comp_name + "_Std_ka{} = {:.4f}".format(ik + 1, ka)
                    for ik, ka in enumerate(self.ka_values)
                ]
                tdf_std = self.dataframe_slices["Transverse"][comp_name].std(level=1, axis="columns")
                tdf_std = tdf_std.rename(col_mapper(tdf_std.columns, ka_columns), axis=1)

                self.dataframe = pd.concat([self.dataframe, tdf_mean, tdf_std], axis=1)

        # Create the MultiIndex columns:
        # Longitudinal_H-He_Mean_Frequencies | ka = 0.123456
        # Longitudinal_H-He_Std_ka = 0.123456

        tend = self.timer.current()
        self.time_stamp(self.__long_name__ + " Calculation", self.timer.time_division(tend - tinit))

        self.save_hdf()

    def pretty_print(self):
        """Print current correlation function calculation parameters for help in choice of simulation parameters."""
        print("\n\n{:=^70} \n".format(" " + self.__long_name__ + " "))
        print("k wavevector information saved in: \n", self.k_file)
        print("v(k,t) data saved in: \n", self.vkt_hdf_file)
        print("Data saved in: \n{}".format(self.filename_hdf))
        print("Data accessible at: self.k_list, self.k_counts, self.ka_values, self.frequencies," " \n\t self.dataframe")
        print("\nFrequency Space Parameters:")
        print("\tNo. of slices = {}".format(self.no_slices))
        print("\tNo. dumps per slice = {}".format(self.slice_steps))
        print("\tFrequency step dw = 2 pi (no_slices * prod_dump_step)/(production_steps * dt)")
        print("\tdw = {:1.4f} w_p = {:1.4e} [rad/s]".format(self.w_min / self.total_plasma_frequency, self.w_min))
        print("\tMaximum Frequency w_max = 2 pi /(prod_dump_step * dt)")
        print("\tw_max = {:1.4f} w_p = {:1.4e} [rad/s]".format(self.w_max / self.total_plasma_frequency, self.w_max))

        print("\n\nWavevector parameters:")
        print("Smallest wavevector k_min = 2 pi / L = 3.9 / N^(1/3)")
        print("k_min = {:.4f} / a_ws = {:.4e} ".format(self.ka_values[0], self.ka_values[0] / self.a_ws), end="")
        print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print("\nAngle averaging choice: {}".format(self.angle_averaging))
        if self.angle_averaging == "full":
            print("\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_aa_harmonics))
            print("\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)")
            print(
                "\tk_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_aa_ka_value, self.max_aa_ka_value / self.a_ws),
                end="",
            )
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == "custom":
            print("\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_aa_harmonics))
            print("\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)")
            print(
                "\tAA k_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_aa_ka_value, self.max_aa_ka_value / self.a_ws),
                end="",
            )
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

            print("\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_k_harmonics))
            print("\tLargest wavector k_max = k_min * n_x")
            print("\tk_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_ka_value, self.max_ka_value / self.a_ws), end="")
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == "principal_axis":
            print("\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_k_harmonics))
            print("\tLargest wavector k_max = k_min * n_x")
            print("\tk_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_ka_value, self.max_ka_value / self.a_ws), end="")
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print("\nTotal number of k values to calculate = {}".format(len(self.k_list)))
        print("No. of unique ka values to calculate = {}".format(len(self.ka_values)))


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

        start_slice = 0
        end_slice = self.slice_steps * self.dump_step
        time = np.zeros(self.slice_steps)
        # Initialize timer
        t0 = self.timer.current()

        df_str = "Diffusion Flux"
        df_acf_str = "Diffusion Flux ACF"

        for isl in range(self.no_slices):
            print("\nCalculating diffusion flux and its acf for slice {}/{}.".format(isl + 1, self.no_slices))
            # Parse the particles from the dump files
            vel = np.zeros((self.dimensions, self.slice_steps, self.total_num_ptcls))
            #
            # print("\nParsing particles' velocities.")
            for it, dump in enumerate(
                tqdm(range(start_slice, end_slice, self.dump_step), desc="Read in data", disable=not self.verbose)
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
                self.dataframe_slices[df_str + " {}_X_slice {}".format(i, isl)] = flux[0, :]
                self.dataframe_slices[df_str + " {}_Y_slice {}".format(i, isl)] = flux[1, :]
                self.dataframe_slices[df_str + " {}_Z_slice {}".format(i, isl)] = flux[2, :]

            for i, flux_acf in enumerate(df_acf):
                self.dataframe_acf_slices[df_acf_str + " {}_X_slice {}".format(i, isl)] = flux_acf[0, :]
                self.dataframe_acf_slices[df_acf_str + " {}_Y_slice {}".format(i, isl)] = flux_acf[1, :]
                self.dataframe_acf_slices[df_acf_str + " {}_Z_slice {}".format(i, isl)] = flux_acf[2, :]
                self.dataframe_acf_slices[df_acf_str + " {}_Total_slice {}".format(i, isl)] = flux_acf[3, :]

            start_slice += self.slice_steps * self.dump_step
            end_slice += self.slice_steps * self.dump_step

        # Average and std over the slices
        for i in range(self.no_fluxes):
            xcol_str = [df_str + " {}_X_slice {}".format(i, isl) for isl in range(self.no_slices)]
            ycol_str = [df_str + " {}_Y_slice {}".format(i, isl) for isl in range(self.no_slices)]
            zcol_str = [df_str + " {}_Z_slice {}".format(i, isl) for isl in range(self.no_slices)]

            self.dataframe[df_str + " {}_X_Mean".format(i)] = self.dataframe_slices[xcol_str].mean(axis=1)
            self.dataframe[df_str + " {}_X_Std".format(i)] = self.dataframe_slices[xcol_str].std(axis=1)
            self.dataframe[df_str + " {}_Y_Mean".format(i)] = self.dataframe_slices[ycol_str].mean(axis=1)
            self.dataframe[df_str + " {}_Y_Std".format(i)] = self.dataframe_slices[ycol_str].std(axis=1)
            self.dataframe[df_str + " {}_Z_Mean".format(i)] = self.dataframe_slices[zcol_str].mean(axis=1)
            self.dataframe[df_str + " {}_Z_Std".format(i)] = self.dataframe_slices[zcol_str].std(axis=1)

        # Average and std over the slices
        for i in range(self.no_fluxes_acf):
            xcol_str = [df_acf_str + " {}_X_slice {}".format(i, isl) for isl in range(self.no_slices)]
            ycol_str = [df_acf_str + " {}_Y_slice {}".format(i, isl) for isl in range(self.no_slices)]
            zcol_str = [df_acf_str + " {}_Z_slice {}".format(i, isl) for isl in range(self.no_slices)]
            tot_col_str = [df_acf_str + " {}_Total_slice {}".format(i, isl) for isl in range(self.no_slices)]

            self.dataframe_acf[df_acf_str + " {}_X_Mean".format(i)] = self.dataframe_acf_slices[xcol_str].mean(axis=1)
            self.dataframe_acf[df_acf_str + " {}_X_Std".format(i)] = self.dataframe_acf_slices[xcol_str].std(axis=1)
            self.dataframe_acf[df_acf_str + " {}_Y_Mean".format(i)] = self.dataframe_acf_slices[ycol_str].mean(axis=1)
            self.dataframe_acf[df_acf_str + " {}_Y_Std".format(i)] = self.dataframe_acf_slices[ycol_str].std(axis=1)
            self.dataframe_acf[df_acf_str + " {}_Z_Mean".format(i)] = self.dataframe_acf_slices[zcol_str].mean(axis=1)
            self.dataframe_acf[df_acf_str + " {}_Z_Std".format(i)] = self.dataframe_acf_slices[zcol_str].std(axis=1)
            self.dataframe_acf[df_acf_str + " {}_Total_Mean".format(i)] = self.dataframe_acf_slices[tot_col_str].mean(
                axis=1
            )
            self.dataframe_acf[df_acf_str + " {}_Total_Std".format(i)] = self.dataframe_acf_slices[tot_col_str].std(
                axis=1
            )

        self.save_hdf()
        tend = self.timer.current()
        self.time_stamp("Diffusion Flux and its ACF Calculation", self.timer.time_division(tend - t0))

    def pretty_print(self):
        """Print observable parameters for help in choice of simulation parameters."""

        print("\n\n{:=^70} \n".format(" " + self.__long_name__ + " "))
        print("Data saved in: \n", self.filename_hdf)
        print("Data accessible at: self.dataframe")

        print("\nNo. of slices = {}".format(self.no_slices))
        print("No. dumps per slice = {}".format(int(self.slice_steps / self.dump_step)))
        print(
            "Time interval of autocorrelation function = {:.4e} [s] ~ {} w_p T".format(
                self.dt * self.slice_steps, int(self.dt * self.slice_steps * self.total_plasma_frequency)
            )
        )


class DynamicStructureFactor(Observable):
    """Dynamic Structure factor.

    The species dependent DSF :math:`S_{AB}(k,\\omega)` is calculated from

    .. math::
        S_{AB }(k,\\omega) = \\int_0^\\infty dt \\,
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

        # Parse nkt otherwise calculate it
        self.parse_kt_data(nkt_flag=True)

        tinit = self.timer.current()
        nkt_df = pd.read_hdf(self.nkt_hdf_file, mode="r", key="nkt")
        for isl in range(0, self.no_slices):
            # Initialize container
            nkt = np.zeros((self.num_species, self.slice_steps, len(self.k_list)), dtype=np.complex128)
            for sp, sp_name in enumerate(self.species_names):
                nkt[sp] = np.array(nkt_df["slice {}".format(isl + 1)][sp_name])

            # Calculate Skw
            Skw_all = calc_Skw(nkt, self.k_list, self.species_num, self.slice_steps, self.dt, self.dump_step)

            # Create the dataframe's column names
            slc_column = "slice {}".format(isl + 1)
            ka_columns = ["ka = {:.8f}".format(ka) for ik, ka in enumerate(self.ka_values)]
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
                    self.dataframe_slices = pd.concat(
                        [self.dataframe_slices, pd.DataFrame(Skw_all[sp_indx, :, :].T, columns=columns)], axis=1
                    )
                    sp_indx += 1

        # Create the MultiIndex
        tuples = [tuple(c.split("_")) for c in self.dataframe_slices.columns]
        self.dataframe_slices.columns = pd.MultiIndex.from_tuples(
            tuples, names=["species", "slices", "k_index", "k_harmonics"]
        )

        # Now for the actual df
        self.dataframe[" _ _Frequencies"] = self.frequencies

        # Take the mean and std and store them into the dataframe to return
        for sp1, sp1_name in enumerate(self.species_names):
            for sp2, sp2_name in enumerate(self.species_names[sp1:], sp1):
                skw_name = "{}-{}".format(sp1_name, sp2_name)
                # Rename the columns with values of ka
                ka_columns = [skw_name + "_Mean_ka{} = {:.4f}".format(ik + 1, ka) for ik, ka in enumerate(self.ka_values)]
                # Mean: level = 1 corresponds to averaging all the k harmonics with the same magnitude
                df_mean = self.dataframe_slices[skw_name].mean(level=1, axis="columns")
                df_mean = df_mean.rename(col_mapper(df_mean.columns, ka_columns), axis=1)
                # Std
                ka_columns = [skw_name + "_Std_ka{} = {:.4f}".format(ik + 1, ka) for ik, ka in enumerate(self.ka_values)]
                df_std = self.dataframe_slices[skw_name].std(level=1, axis="columns")
                df_std = df_std.rename(col_mapper(df_std.columns, ka_columns), axis=1)

                self.dataframe = pd.concat([self.dataframe, df_mean, df_std], axis=1)

        tend = self.timer.current()
        self.time_stamp(self.__long_name__ + " Calculation", self.timer.time_division(tend - tinit))

        self.save_hdf()

    def pretty_print(self):
        """Print dynamic structure factor calculation parameters for help in choice of simulation parameters."""

        print("\n\n{:=^70} \n".format(" " + self.__long_name__ + " "))
        print("k wavevector information saved in: \n", self.k_file)
        print("n(k,t) data saved in: \n", self.nkt_hdf_file)
        print("Data saved in: \n", self.filename_hdf)
        print("Data accessible at: self.k_list, self.k_counts, self.ka_values, self.frequencies, self.dataframe")

        print("\nFrequency Space Parameters:")
        print("\tNo. of slices = {}".format(self.no_slices))
        print("\tNo. dumps per slice = {}".format(self.slice_steps))
        print("\tFrequency step dw = 2 pi (no_slices * prod_dump_step)/(production_steps * dt)")
        print("\tdw = {:1.4f} w_p = {:1.4e} [rad/s]".format(self.w_min / self.total_plasma_frequency, self.w_min))
        print("\tMaximum Frequency w_max = 2 pi /(prod_dump_step * dt)")
        print("\tw_max = {:1.4f} w_p = {:1.4e} [rad/s]".format(self.w_max / self.total_plasma_frequency, self.w_max))

        print("\n\nWavevector parameters:")
        print("Smallest wavevector k_min = 2 pi / L = 3.9 / N^(1/3)")
        print("k_min = {:.4f} / a_ws = {:.4e} ".format(self.ka_values[0], self.ka_values[0] / self.a_ws), end="")
        print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print("\nAngle averaging choice: {}".format(self.angle_averaging))
        if self.angle_averaging == "full":
            print("\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_aa_harmonics))
            print("\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)")
            print(
                "\tk_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_aa_ka_value, self.max_aa_ka_value / self.a_ws),
                end="",
            )
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == "custom":
            print("\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_aa_harmonics))
            print("\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)")
            print(
                "\tAA k_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_aa_ka_value, self.max_aa_ka_value / self.a_ws),
                end="",
            )
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

            print("\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_k_harmonics))
            print("\tLargest wavector k_max = k_min * n_x")
            print("\tk_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_ka_value, self.max_ka_value / self.a_ws), end="")
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == "principal_axis":
            print("\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_k_harmonics))
            print("\tLargest wavector k_max = k_min * n_x")
            print("\tk_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_ka_value, self.max_ka_value / self.a_ws), end="")
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print("\nTotal number of k values to calculate = {}".format(len(self.k_list)))
        print("No. of unique ka values to calculate = {}".format(len(self.ka_values)))


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

        start_slice = 0
        end_slice = self.slice_steps * self.dump_step
        time = np.zeros(self.slice_steps)
        # Initialize timer
        t0 = self.timer.current()

        ec_str = "Electric Current"
        ec_acf_str = "Electric Current ACF"

        for isl in range(self.no_slices):
            print("\nCalculating electric current and its acf for slice {}/{}.".format(isl + 1, self.no_slices))
            # Parse the particles from the dump files
            vel = np.zeros((self.dimensions, self.slice_steps, self.total_num_ptcls))
            #
            # print("\nParsing particles' velocities.")
            for it, dump in enumerate(
                tqdm(range(start_slice, end_slice, self.dump_step), desc="Read in data", disable=not self.verbose)
            ):
                datap = load_from_restart(self.dump_dir, dump)
                time[it] = datap["time"]
                vel[0, it, :] = datap["vel"][:, 0]
                vel[1, it, :] = datap["vel"][:, 1]
                vel[2, it, :] = datap["vel"][:, 2]
            #
            if isl == 0:
                self.dataframe["Time"] = np.copy(time)
                self.dataframe_acf["Time"] = np.copy(time)
                self.dataframe_slices["Time"] = np.copy(time)
                self.dataframe_acf_slices["Time"] = np.copy(time)

            species_current, total_current = calc_elec_current(vel, self.species_charges, self.species_num)

            # # Store the data
            for i, sp_name in enumerate(self.species_names):
                col_str = "{} ".format(sp_name) + ec_str
                self.dataframe_slices[col_str + "_X_slice {}".format(isl)] = species_current[i, 0, :]
                self.dataframe_slices[col_str + "_Y_slice {}".format(isl)] = species_current[i, 1, :]
                self.dataframe_slices[col_str + "_Z_slice {}".format(isl)] = species_current[i, 2, :]
                # Calculate ACF
                sp_acf_xx = correlationfunction(species_current[i, 0, :], species_current[i, 0, :])
                sp_acf_yy = correlationfunction(species_current[i, 1, :], species_current[i, 1, :])
                sp_acf_zz = correlationfunction(species_current[i, 2, :], species_current[i, 2, :])
                tot_acf = sp_acf_xx + sp_acf_yy + sp_acf_zz
                col_str = "{} ".format(sp_name) + ec_acf_str
                # Store ACF
                self.dataframe_acf_slices[col_str + "_X_slice {}".format(isl)] = sp_acf_xx
                self.dataframe_acf_slices[col_str + "_Y_slice {}".format(isl)] = sp_acf_yy
                self.dataframe_acf_slices[col_str + "_Z_slice {}".format(isl)] = sp_acf_zz
                self.dataframe_acf_slices[col_str + "_Total_slice {}".format(isl)] = tot_acf

            # Total current and its ACF
            self.dataframe_slices[ec_str + "_X_slice {}".format(isl)] = total_current[0, :]
            self.dataframe_slices[ec_str + "_Y_slice {}".format(isl)] = total_current[1, :]
            self.dataframe_slices[ec_str + "_Z_slice {}".format(isl)] = total_current[2, :]

            sp_acf_xx = correlationfunction(total_current[0, :], total_current[0, :])
            sp_acf_yy = correlationfunction(total_current[1, :], total_current[1, :])
            sp_acf_zz = correlationfunction(total_current[2, :], total_current[2, :])
            tot_acf = sp_acf_xx + sp_acf_yy + sp_acf_zz

            self.dataframe_acf_slices[ec_acf_str + "_X_slice {}".format(isl)] = sp_acf_xx
            self.dataframe_acf_slices[ec_acf_str + "_Y_slice {}".format(isl)] = sp_acf_yy
            self.dataframe_acf_slices[ec_acf_str + "_Z_slice {}".format(isl)] = sp_acf_zz
            self.dataframe_acf_slices[ec_acf_str + "_Total_slice {}".format(isl)] = tot_acf

            start_slice += self.slice_steps * self.dump_step
            end_slice += self.slice_steps * self.dump_step

        # Average and std over the slices
        for i, sp_name in enumerate(self.species_names):
            col_str = "{} ".format(sp_name) + ec_str
            xcol_str = [col_str + "_X_slice {}".format(isl) for isl in range(self.no_slices)]
            ycol_str = [col_str + "_Y_slice {}".format(isl) for isl in range(self.no_slices)]
            zcol_str = [col_str + "_Z_slice {}".format(isl) for isl in range(self.no_slices)]

            self.dataframe[col_str + "_X_Mean"] = self.dataframe_slices[xcol_str].mean(axis=1)
            self.dataframe[col_str + "_X_Std"] = self.dataframe_slices[xcol_str].std(axis=1)
            self.dataframe[col_str + "_Y_Mean"] = self.dataframe_slices[ycol_str].mean(axis=1)
            self.dataframe[col_str + "_Y_Std"] = self.dataframe_slices[ycol_str].std(axis=1)
            self.dataframe[col_str + "_Z_Mean"] = self.dataframe_slices[zcol_str].mean(axis=1)
            self.dataframe[col_str + "_Z_Std"] = self.dataframe_slices[zcol_str].std(axis=1)
            # ACF averages
            col_str = "{} ".format(sp_name) + ec_acf_str
            xcol_str = [col_str + "_X_slice {}".format(isl) for isl in range(self.no_slices)]
            ycol_str = [col_str + "_Y_slice {}".format(isl) for isl in range(self.no_slices)]
            zcol_str = [col_str + "_Z_slice {}".format(isl) for isl in range(self.no_slices)]
            tot_col_str = [col_str + "_Total_slice {}".format(isl) for isl in range(self.no_slices)]

            self.dataframe_acf[col_str + "_X_Mean"] = self.dataframe_acf_slices[xcol_str].mean(axis=1)
            self.dataframe_acf[col_str + "_X_Std"] = self.dataframe_acf_slices[xcol_str].std(axis=1)
            self.dataframe_acf[col_str + "_Y_Mean"] = self.dataframe_acf_slices[ycol_str].mean(axis=1)
            self.dataframe_acf[col_str + "_Y_Std"] = self.dataframe_acf_slices[ycol_str].std(axis=1)
            self.dataframe_acf[col_str + "_Z_Mean"] = self.dataframe_acf_slices[zcol_str].mean(axis=1)
            self.dataframe_acf[col_str + "_Z_Std"] = self.dataframe_acf_slices[zcol_str].std(axis=1)
            self.dataframe_acf[col_str + "_Total_Mean"] = self.dataframe_acf_slices[tot_col_str].mean(axis=1)
            self.dataframe_acf[col_str + "_Total_Std"] = self.dataframe_acf_slices[tot_col_str].std(axis=1)

        # Total
        xcol_str = [ec_str + "_X_slice {}".format(isl) for isl in range(self.no_slices)]
        ycol_str = [ec_str + "_Y_slice {}".format(isl) for isl in range(self.no_slices)]
        zcol_str = [ec_str + "_Z_slice {}".format(isl) for isl in range(self.no_slices)]

        self.dataframe[ec_str + "_X_Mean"] = self.dataframe_slices[xcol_str].mean(axis=1)
        self.dataframe[ec_str + "_X_Std"] = self.dataframe_slices[xcol_str].std(axis=1)
        self.dataframe[ec_str + "_Y_Mean"] = self.dataframe_slices[ycol_str].mean(axis=1)
        self.dataframe[ec_str + "_Y_Std"] = self.dataframe_slices[ycol_str].std(axis=1)
        self.dataframe[ec_str + "_Z_Mean"] = self.dataframe_slices[zcol_str].mean(axis=1)
        self.dataframe[ec_str + "_Z_Std"] = self.dataframe_slices[zcol_str].std(axis=1)
        # Total ACF
        xcol_str = [ec_acf_str + "_X_slice {}".format(isl) for isl in range(self.no_slices)]
        ycol_str = [ec_acf_str + "_Y_slice {}".format(isl) for isl in range(self.no_slices)]
        zcol_str = [ec_acf_str + "_Z_slice {}".format(isl) for isl in range(self.no_slices)]
        tot_col_str = [ec_acf_str + "_Total_slice {}".format(isl) for isl in range(self.no_slices)]

        self.dataframe_acf[ec_acf_str + "_X_Mean"] = self.dataframe_acf_slices[xcol_str].mean(axis=1)
        self.dataframe_acf[ec_acf_str + "_X_Std"] = self.dataframe_acf_slices[xcol_str].std(axis=1)
        self.dataframe_acf[ec_acf_str + "_Y_Mean"] = self.dataframe_acf_slices[ycol_str].mean(axis=1)
        self.dataframe_acf[ec_acf_str + "_Y_Std"] = self.dataframe_acf_slices[ycol_str].std(axis=1)
        self.dataframe_acf[ec_acf_str + "_Z_Mean"] = self.dataframe_acf_slices[zcol_str].mean(axis=1)
        self.dataframe_acf[ec_acf_str + "_Z_Std"] = self.dataframe_acf_slices[zcol_str].std(axis=1)
        self.dataframe_acf[ec_acf_str + "_Total_Mean"] = self.dataframe_acf_slices[tot_col_str].mean(axis=1)
        self.dataframe_acf[ec_acf_str + "_Total_Std"] = self.dataframe_acf_slices[tot_col_str].std(axis=1)

        self.save_hdf()


class PressureTensor(Observable):
    """Pressure Tensor."""

    def __init__(self):
        super().__init__()
        self.__name__ = "pressure_tensor"
        self.__long_name__ = "Pressure Tensor"
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
    def compute(self):

        start_slice = 0
        end_slice = self.slice_steps * self.dump_step
        time = np.zeros(self.slice_steps)
        # Initialize timer
        t0 = self.timer.current()

        pt_str_kin = "Pressure Tensor Kinetic"
        pt_str_pot = "Pressure Tensor Potential"
        pt_str = "Pressure Tensor"
        pt_acf_str_kin = "Pressure Tensor Kinetic ACF"
        pt_acf_str_pot = "Pressure Tensor Potential ACF"
        pt_acf_str_kinpot = "Pressure Tensor Kin-Pot ACF"
        pt_acf_str_potkin = "Pressure Tensor Pot-Kin ACF"
        pt_acf_str = "Pressure Tensor ACF"

        for isl in range(self.no_slices):
            print("\nCalculating stress tensor and the acfs for slice {}/{}.".format(isl + 1, self.no_slices))
            # Parse the particles from the dump files
            pressure = np.zeros(self.slice_steps)
            pt_kin_temp = np.zeros((self.dimensions, self.dimensions, self.slice_steps))
            pt_pot_temp = np.zeros((self.dimensions, self.dimensions, self.slice_steps))
            pt_temp = np.zeros((self.dimensions, self.dimensions, self.slice_steps))

            for it, dump in enumerate(
                tqdm(
                    range(start_slice, end_slice, self.dump_step),
                    desc="Calculating Pressure Tensor",
                    disable=not self.verbose,
                )
            ):
                datap = load_from_restart(self.dump_dir, dump)
                time[it] = datap["time"]

                pressure[it], pt_kin_temp[:, :, it], pt_pot_temp[:, :, it], pt_temp[:, :, it] = calc_pressure_tensor(
                    datap["vel"], datap["virial"], self.species_masses, self.species_num, self.box_volume
                )

            if isl == 0:
                self.dataframe["Time"] = np.copy(time)
                self.dataframe_acf["Time"] = np.copy(time)
                self.dataframe_slices["Time"] = np.copy(time)
                self.dataframe_acf_slices["Time"] = np.copy(time)

            self.dataframe_slices["Pressure_slice {}".format(isl)] = pressure
            self.dataframe_acf_slices["Pressure ACF_slice {}".format(isl)] = correlationfunction(pressure, pressure)
            # This is needed for the bulk viscosity
            delta_pressure = pressure - pressure.mean()
            self.dataframe_slices["Delta Pressure_slice {}".format(isl)] = delta_pressure
            self.dataframe_acf_slices["Delta Pressure ACF_slice {}".format(isl)] = correlationfunction(
                delta_pressure, delta_pressure
            )

            if self.dimensions == 3:
                dim_lbl = ["x", "y", "z"]
            elif self.dimensions == 2:
                dim_lbl = ["x", "y"]

            # The reason for dividing these two loops is because I want a specific order in the dataframe.
            # Pressure, Stress Tensor, All
            for i, ax1 in enumerate(dim_lbl):
                for j, ax2 in enumerate(dim_lbl):
                    self.dataframe_slices[pt_str_kin + " {}{}_slice {}".format(ax1, ax2, isl)] = pt_kin_temp[i, j, :]
                    self.dataframe_slices[pt_str_pot + " {}{}_slice {}".format(ax1, ax2, isl)] = pt_pot_temp[i, j, :]
                    self.dataframe_slices[pt_str + " {}{}_slice {}".format(ax1, ax2, isl)] = pt_temp[i, j, :]

            # Calculate the (thermal fluctuations of the) elastic moduli from the acf of the stress tensor elements
            # Note: C_{abcd} = < sigma_{ab} sigma_{cd} >
            for i, ax1 in enumerate(dim_lbl):
                for j, ax2 in enumerate(dim_lbl):
                    for k, ax3 in enumerate(dim_lbl):
                        for l, ax4 in enumerate(dim_lbl):
                            C_ijkl_kin = correlationfunction(pt_kin_temp[i, j, :], pt_kin_temp[k, l, :])
                            C_ijkl_pot = correlationfunction(pt_pot_temp[i, j, :], pt_pot_temp[k, l, :])
                            C_ijkl_kinpot = correlationfunction(pt_kin_temp[i, j, :], pt_pot_temp[k, l, :])
                            C_ijkl_potkin = correlationfunction(pt_pot_temp[i, j, :], pt_kin_temp[k, l, :])

                            C_ijkl = correlationfunction(pt_temp[i, j, :], pt_temp[k, l, :])

                            self.dataframe_acf_slices[
                                pt_acf_str_kin + " {}{}{}{}_slice {}".format(ax1, ax2, ax3, ax4, isl)
                            ] = C_ijkl_kin
                            self.dataframe_acf_slices[
                                pt_acf_str_pot + " {}{}{}{}_slice {}".format(ax1, ax2, ax3, ax4, isl)
                            ] = C_ijkl_pot
                            self.dataframe_acf_slices[
                                pt_acf_str_kinpot + " {}{}{}{}_slice {}".format(ax1, ax2, ax3, ax4, isl)
                            ] = C_ijkl_kinpot
                            self.dataframe_acf_slices[
                                pt_acf_str_potkin + " {}{}{}{}_slice {}".format(ax1, ax2, ax3, ax4, isl)
                            ] = C_ijkl_potkin
                            self.dataframe_acf_slices[
                                pt_acf_str + " {}{}{}{}_slice {}".format(ax1, ax2, ax3, ax4, isl)
                            ] = C_ijkl

            start_slice += self.slice_steps * self.dump_step
            end_slice += self.slice_steps * self.dump_step
            # end of slice loop

        # Average and std over the slices
        col_str = ["Pressure_slice {}".format(isl) for isl in range(self.no_slices)]
        self.dataframe["Pressure_Mean"] = self.dataframe_slices[col_str].mean(axis=1)
        self.dataframe["Pressure_Std"] = self.dataframe_slices[col_str].std(axis=1)

        col_str = ["Pressure ACF_slice {}".format(isl) for isl in range(self.no_slices)]
        self.dataframe_acf["Pressure ACF_Mean"] = self.dataframe_acf_slices[col_str].mean(axis=1)
        self.dataframe_acf["Pressure ACF_Std"] = self.dataframe_acf_slices[col_str].std(axis=1)

        col_str = ["Delta Pressure_slice {}".format(isl) for isl in range(self.no_slices)]
        self.dataframe["Delta Pressure_Mean"] = self.dataframe_slices[col_str].mean(axis=1)
        self.dataframe["Delta Pressure_Std"] = self.dataframe_slices[col_str].std(axis=1)

        col_str = ["Delta Pressure ACF_slice {}".format(isl) for isl in range(self.no_slices)]
        self.dataframe_acf["Delta Pressure ACF_Mean"] = self.dataframe_acf_slices[col_str].mean(axis=1)
        self.dataframe_acf["Delta Pressure ACF_Std"] = self.dataframe_acf_slices[col_str].std(axis=1)

        for i, ax1 in enumerate(dim_lbl):
            for j, ax2 in enumerate(dim_lbl):
                # Kinetic Terms
                ij_col_str = [pt_str_kin + " {}{}_slice {}".format(ax1, ax2, isl) for isl in range(self.no_slices)]
                self.dataframe[pt_str_kin + " {}{}_Mean".format(ax1, ax2)] = self.dataframe_slices[ij_col_str].mean(
                    axis=1
                )
                self.dataframe[pt_str_kin + " {}{}_Std".format(ax1, ax2)] = self.dataframe_slices[ij_col_str].std(axis=1)

                # Potential Terms
                ij_col_str = [pt_str_pot + " {}{}_slice {}".format(ax1, ax2, isl) for isl in range(self.no_slices)]
                self.dataframe[pt_str_pot + " {}{}_Mean".format(ax1, ax2)] = self.dataframe_slices[ij_col_str].mean(
                    axis=1
                )
                self.dataframe[pt_str_pot + " {}{}_Std".format(ax1, ax2)] = self.dataframe_slices[ij_col_str].std(axis=1)

                # Full
                ij_col_str = [pt_str + " {}{}_slice {}".format(ax1, ax2, isl) for isl in range(self.no_slices)]
                self.dataframe[pt_str + " {}{}_Mean".format(ax1, ax2)] = self.dataframe_slices[ij_col_str].mean(axis=1)
                self.dataframe[pt_str + " {}{}_Std".format(ax1, ax2)] = self.dataframe_slices[ij_col_str].std(axis=1)

        for i, ax1 in enumerate(dim_lbl):
            for j, ax2 in enumerate(dim_lbl):
                for k, ax3 in enumerate(dim_lbl):
                    for l, ax4 in enumerate(dim_lbl):
                        # Kinetic Terms
                        ij_col_acf_str = [
                            pt_acf_str_kin + " {}{}{}{}_slice {}".format(ax1, ax2, ax3, ax4, isl)
                            for isl in range(self.no_slices)
                        ]
                        mean_column = pt_acf_str_kin + " {}{}{}{}_Mean".format(ax1, ax2, ax3, ax4)
                        std_column = pt_acf_str_kin + " {}{}{}{}_Std".format(ax1, ax2, ax3, ax4)
                        self.dataframe_acf[mean_column] = self.dataframe_acf_slices[ij_col_acf_str].mean(axis=1)
                        self.dataframe_acf[std_column] = self.dataframe_acf_slices[ij_col_acf_str].std(axis=1)
                        #
                        # Potential Terms
                        ij_col_acf_str = [
                            pt_acf_str_pot + " {}{}{}{}_slice {}".format(ax1, ax2, ax3, ax4, isl)
                            for isl in range(self.no_slices)
                        ]
                        mean_column = pt_acf_str_pot + " {}{}{}{}_Mean".format(ax1, ax2, ax3, ax4)
                        std_column = pt_acf_str_pot + " {}{}{}{}_Std".format(ax1, ax2, ax3, ax4)
                        self.dataframe_acf[mean_column] = self.dataframe_acf_slices[ij_col_acf_str].mean(axis=1)
                        self.dataframe_acf[std_column] = self.dataframe_acf_slices[ij_col_acf_str].std(axis=1)
                        #
                        # Kinetic-Potential Terms
                        ij_col_acf_str = [
                            pt_acf_str_kinpot + " {}{}{}{}_slice {}".format(ax1, ax2, ax3, ax4, isl)
                            for isl in range(self.no_slices)
                        ]
                        mean_column = pt_acf_str_kinpot + " {}{}{}{}_Mean".format(ax1, ax2, ax3, ax4)
                        std_column = pt_acf_str_kinpot + " {}{}{}{}_Std".format(ax1, ax2, ax3, ax4)
                        self.dataframe_acf[mean_column] = self.dataframe_acf_slices[ij_col_acf_str].mean(axis=1)
                        self.dataframe_acf[std_column] = self.dataframe_acf_slices[ij_col_acf_str].std(axis=1)
                        #
                        # Potential-Kinetic Terms
                        ij_col_acf_str = [
                            pt_acf_str_potkin + " {}{}{}{}_slice {}".format(ax1, ax2, ax3, ax4, isl)
                            for isl in range(self.no_slices)
                        ]
                        mean_column = pt_acf_str_potkin + " {}{}{}{}_Mean".format(ax1, ax2, ax3, ax4)
                        std_column = pt_acf_str_potkin + " {}{}{}{}_Std".format(ax1, ax2, ax3, ax4)
                        self.dataframe_acf[mean_column] = self.dataframe_acf_slices[ij_col_acf_str].mean(axis=1)
                        self.dataframe_acf[std_column] = self.dataframe_acf_slices[ij_col_acf_str].std(axis=1)
                        #
                        # Full
                        ij_col_acf_str = [
                            pt_acf_str + " {}{}{}{}_slice {}".format(ax1, ax2, ax3, ax4, isl)
                            for isl in range(self.no_slices)
                        ]
                        mean_column = pt_acf_str + " {}{}{}{}_Mean".format(ax1, ax2, ax3, ax4)
                        std_column = pt_acf_str + " {}{}{}{}_Std".format(ax1, ax2, ax3, ax4)
                        self.dataframe_acf[mean_column] = self.dataframe_acf_slices[ij_col_acf_str].mean(axis=1)
                        self.dataframe_acf[std_column] = self.dataframe_acf_slices[ij_col_acf_str].std(axis=1)

        self.save_hdf()
        # # TODO: Fix this hack. We should be able to add data to HDF instead of removing it and rewriting it.
        # # Save the data.
        # if os.path.exists(self.filename_hdf_slices):
        #     os.remove(self.filename_hdf_slices)
        # else:
        #     self.dataframe_slices.to_hdf(self.filename_hdf_slices, mode='w', key=self.__name__)
        #
        # if os.path.exists(self.filename_hdf):
        #     os.remove(self.filename_hdf)
        # else:
        #     self.dataframe.to_hdf(self.filename_hdf, mode='w', key=self.__name__)
        #
        # if os.path.exists(self.filename_hdf_acf):
        #     os.remove(self.filename_hdf_acf)
        # else:
        #     self.dataframe_acf.to_hdf(self.filename_hdf_acf, mode='w', key=self.__name__)
        #
        # if os.path.exists(self.filename_hdf_acf_slices):
        #     os.remove(self.filename_hdf_acf_slices)
        # else:
        #     self.dataframe_acf_slices.to_hdf(self.filename_hdf_acf_slices, mode='w', key=self.__name__)

        tend = self.timer.current()
        self.time_stamp("Stress Tensor and ACF Calculation", self.timer.time_division(tend - t0))

    def sum_rule(self, beta, rdf, potential):
        """
        Calculate the sum rule integrals from the rdf.

        .. math::
            :nowrap:

            \\begin{eqnarray}
                \\sigma_{zzzz} & = &   \\frac{n}{\\beta^2} \\left [ 3 +
                \\frac{2\\beta}{15} I^{(1)} + \\frac{\\beta}{5} I^{(2)} \\right ] , \\\\
                \\sigma_{zzxx} & =& \\frac{n}{\\beta^2} \\left [ 1
                - \\frac{2\\beta}{5} I^{(1)} + \\frac {\\beta}{15} I^{(2)} \\right ] , \\\\
                \\sigma_{xyxy} & = & \\frac{n}{\\beta^2} \\left [ 1 +
                \\frac{4\\beta}{15} I^{(2)} + \\frac {\\beta}{15} I^{(2)} \\right ] ,
            \\end{eqnarray}

        where :math:`I^{(k)} = \\sum_{A} \\sum_{B \geq A}I_{AB}^{(\\rm {Hartree}, k)} + I_{AB}^{(\\rm {Corr}, k)}`
        calculated from
        :meth:`sarkas.tools.observables.RadialDistributionFunction.compute_sum_rule_integrals`.

        Parameters
        ----------
        beta: float
            Inverse temperature factor. Grab it from :attr:`sarkas.tools.observables.Thermodynamics.beta`.

        rdf: sarkas.tools.observables.RadialDistributionFunction
            Radial Distribution function object.

        potential: sarkas.potentials.core.Potential
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
        I_1 = beta * (hartrees[:, 1].sum() + corrs[:, 1].sum())
        I_2 = beta * (hartrees[:, 2].sum() + corrs[:, 2].sum())

        sigma_zzzz = (3.0 * self.total_num_density ** 2 + 2.0 / 15.0 * I_1 + 1.0 / 5.0 * I_2) / beta ** 2
        sigma_zzxx = (1.0 * self.total_num_density ** 2 - 2.0 / 5.0 * I_1 + 1.0 / 15.0 * I_2) / beta ** 2
        sigma_xyxy = (1.0 * self.total_num_density ** 2 + 4.0 / 15.0 * I_1 + 1.0 / 15.0 * I_2) / beta ** 2

        return sigma_zzzz, sigma_zzxx, sigma_xyxy

    def pretty_print(self):
        """Print observable parameters for help in choice of simulation parameters."""

        print("\n\n{:=^70} \n".format(" " + self.__long_name__ + " "))
        print("Data saved in: \n", self.filename_hdf)
        print("Data accessible at: self.dataframe")

        print("\nNo. of slices = {}".format(self.no_slices))
        print("No. dumps per slice = {}".format(int(self.slice_steps / self.dump_step)))
        print(
            "Time interval of autocorrelation function = {:.4e} [s] ~ {} w_p T".format(
                self.dt * self.slice_steps, int(self.dt * self.slice_steps * self.total_plasma_frequency)
            )
        )


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

        # initialize temporary arrays
        r_values = np.zeros(self.no_bins)
        bin_vol = np.zeros(self.no_bins)
        pair_density = np.zeros((self.num_species, self.num_species))
        gr = np.zeros((self.no_bins, self.no_obs))

        # This is needed to be certain the number of bins is the same.
        # if not isinstance(rdf_hist, np.ndarray):
        #     # Find the last dump by looking for the largest number in the checkpoints filenames
        #     dumps_list = os.listdir(self.dump_dir)
        #     dumps_list.sort(key=num_sort)
        #     name, ext = os.path.splitext(dumps_list[-1])
        #     _, number = name.split('_')
        datap = load_from_restart(self.dump_dir, 0)

        # Make sure you are getting the right number of bins and redefine dr_rdf.
        self.no_bins = datap["rdf_hist"].shape[0]
        self.dr_rdf = self.rc / self.no_bins

        t0 = self.timer.current()
        # No. of pairs per volume
        for i, sp1 in enumerate(self.species_num):
            pair_density[i, i] = sp1 * (sp1 - 1) / self.box_volume
            if self.num_species > 1:
                for j, sp2 in enumerate(self.species_num[i + 1 :], i + 1):
                    pair_density[i, j] = sp1 * sp2 / self.box_volume

        # Calculate the volume of each bin
        sphere_shell_const = 4.0 * np.pi / 3.0
        bin_vol[0] = sphere_shell_const * self.dr_rdf ** 3
        for ir in range(1, self.no_bins):
            r1 = ir * self.dr_rdf
            r2 = (ir + 1) * self.dr_rdf
            bin_vol[ir] = sphere_shell_const * (r2 ** 3 - r1 ** 3)
            r_values[ir] = (ir + 0.5) * self.dr_rdf

        # Save the ra values for simplicity
        self.ra_values = r_values / self.a_ws

        self.dataframe["Distance"] = r_values
        self.dataframe_slices["Distance"] = r_values
        for isl in tqdm(range(self.no_slices), disable=not self.verbose):

            # Grab the data from the dumps. The -1 is for '0'-indexing
            dump_no = (isl + 1) * (self.slice_steps - 1) * self.dump_step
            datap = load_from_restart(self.dump_dir, int(dump_no))
            for i, sp1 in enumerate(self.species_names):
                for j, sp2 in enumerate(self.species_names[i:], i):
                    denom_const = pair_density[i, j] * self.slice_steps * self.dump_step
                    col_str = "{}-{} RDF_slice {}".format(sp1, sp2, isl)
                    self.dataframe_slices[col_str] = (
                        (datap["rdf_hist"][:, i, j] + datap["rdf_hist"][:, j, i]) / denom_const / bin_vol
                    )

        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:], i):
                col_str = ["{}-{} RDF_slice {}".format(sp1, sp2, isl) for isl in range(self.no_slices)]
                self.dataframe["{}-{} RDF_Mean".format(sp1, sp2)] = self.dataframe_slices[col_str].mean(axis=1)
                self.dataframe["{}-{} RDF_Std".format(sp1, sp2)] = self.dataframe_slices[col_str].std(axis=1)

        self.save_hdf()
        self.save_pickle()
        tend = self.timer.current()
        self.time_stamp(self.__long_name__ + " Calculation", self.timer.time_division(tend - t0))

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
        potential: sarkas.potentials.core.Potential
            Sarkas Potential object. Needed for all its attributes.

        Returns
        -------
        hartrees : numpy.ndarray
            Hartree integrals with :math:`k = {0, 1, 2}`. \n
            Shape = ( :attr:`~.no_obs`, 3).

        corrs : numpy.ndarray
            Correlational integrals with :math:`k = {0, 1, 2}`. \n
            Shape = ( :attr:`~.no_obs`, 3).

        """
        r = np.copy(self.dataframe["Distance"].iloc[:, 0])

        dims = self.dimensions
        dim_const = 2.0 ** (dims - 2) * np.pi

        if r[0] == 0.0:
            r[0] = r[1]

        r2 = r * r
        r3 = r2 * r

        corrs = np.zeros((self.no_obs, 3))
        hartrees = np.zeros((self.no_obs, 3))

        obs_indx = 0
        for sp1, sp1_name in enumerate(self.species_names):
            for sp2, sp2_name in enumerate(self.species_names[sp1:], sp1):
                h_r = self.dataframe[("{}-{} RDF".format(sp1_name, sp2_name), "Mean")].to_numpy() - 1.0

                if potential.type == "coulomb":
                    u_r = potential.matrix[0, sp1, sp2] / r
                    dv_dr = -potential.matrix[0, sp1, sp2] / r2
                    d2v_dr2 = 2.0 * potential.matrix[0, sp1, sp2] / r3
                    # Check for finiteness of first element when r[0] = 0.0
                    if not np.isfinite(dv_dr[0]):
                        dv_dr[0] = dv_dr[1]
                        d2v_dr2[0] = d2v_dr2[1]

                elif potential.type == "yukawa":
                    kappa = potential.matrix[1, sp1, sp2]
                    u_r = potential.matrix[0, sp1, sp2] * np.exp(-kappa * r) / r
                    dv_dr = -(1.0 + kappa * r) * u_r / r

                    d2v_dr2 = -u_r / r - (1.0 + kappa * r) ** 2 * u_r / r2 + (1.0 + kappa * r) * u_r / r2

                    # Check for finiteness of first element when r[0] = 0.0
                    if not np.isfinite(dv_dr[0]):
                        dv_dr[0] = dv_dr[1]
                        d2v_dr2[0] = d2v_dr2[1]

                else:
                    raise ValueError("Unknown potential")

                densities = self.species_num_dens[sp1] * self.species_num_dens[sp2]

                hartrees[obs_indx, 0] = dim_const * densities * np.trapz(u_r * r ** (dims - 1), x=r)
                corrs[obs_indx, 0] = dim_const * densities * np.trapz(u_r * h_r * r ** (dims - 1), x=r)

                hartrees[obs_indx, 1] = dim_const * densities * np.trapz(dv_dr * r ** dims, x=r)
                corrs[obs_indx, 1] = dim_const * densities * np.trapz(dv_dr * h_r * r ** dims, x=r)

                hartrees[obs_indx, 2] = dim_const * densities * np.trapz(d2v_dr2 * r ** (dims + 1), x=r)
                corrs[obs_indx, 2] = dim_const * densities * np.trapz(d2v_dr2 * h_r * r ** (dims + 1), x=r)

                obs_indx += 1

        return hartrees, corrs

    def pretty_print(self):
        """Print radial distribution function calculation parameters for help in choice of simulation parameters."""

        print("\n\n{:=^70} \n".format(" " + self.__long_name__ + " "))
        print("Data saved in: \n", self.filename_hdf)
        print("Data accessible at: self.ra_values, self.dataframe")
        print("\nNo. bins = {}".format(self.no_bins))
        print("dr = {:1.4f} a_ws = {:1.4e} ".format(self.dr_rdf / self.a_ws, self.dr_rdf), end="")
        print("[cm]" if self.units == "cgs" else "[m]")
        print(
            "Maximum Distance (i.e. potential.rc)= {:1.4f} a_ws = {:1.4e} ".format(self.rc / self.a_ws, self.rc), end=""
        )
        print("[cm]" if self.units == "cgs" else "[m]")


class StaticStructureFactor(Observable):
    """
    Static Structure Factors.

    The species dependent SSF :math:`S_{AB}(\\mathbf k)` is calculated from

    .. math::
        S_{AB }(\\mathbf k) = \\int_0^\\infty dt \\,
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

        self.parse_kt_data(nkt_flag=True)
        no_dumps_calculated = self.slice_steps * self.no_slices
        Sk_avg = np.zeros((self.no_obs, len(self.k_counts), no_dumps_calculated))

        print("\nCalculating S(k) ...")
        k_column = "Inverse Wavelength"
        self.dataframe_slices[k_column] = self.k_values
        self.dataframe[k_column] = self.k_values

        tinit = self.timer.current()
        nkt_df = pd.read_hdf(self.nkt_hdf_file, mode="r", key="nkt")
        for isl in tqdm(range(self.no_slices)):
            # Initialize container
            nkt = np.zeros((self.num_species, self.slice_steps, len(self.k_list)), dtype=np.complex128)
            for sp, sp_name in enumerate(self.species_names):
                nkt[sp] = np.array(nkt_df["slice {}".format(isl + 1)][sp_name])

            init = isl * self.slice_steps
            fin = (isl + 1) * self.slice_steps
            Sk_avg[:, :, init:fin] = calc_Sk(nkt, self.k_list, self.k_counts, self.species_num, self.slice_steps)
            sp_indx = 0
            for i, sp1 in enumerate(self.species_names):
                for j, sp2 in enumerate(self.species_names[i:]):
                    column = "{}-{} SSF_slice {}".format(sp1, sp2, isl)
                    self.dataframe_slices[column] = Sk_avg[sp_indx, :, init:fin].mean(axis=-1)
                    sp_indx += 1

        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:]):
                column = ["{}-{} SSF_slice {}".format(sp1, sp2, isl) for isl in range(self.no_slices)]

                self.dataframe["{}-{} SSF_Mean".format(sp1, sp2)] = self.dataframe_slices[column].mean(axis=1)
                self.dataframe["{}-{} SSF_Std".format(sp1, sp2)] = self.dataframe_slices[column].std(axis=1)

        self.save_hdf()

        tend = self.timer.current()
        self.time_stamp(self.__long_name__ + " Calculation", self.timer.time_division(tend - tinit))

    def pretty_print(self):
        """Print static structure factor calculation parameters for help in choice of simulation parameters."""

        print("\n\n{:=^70} \n".format(" " + self.__long_name__ + " "))
        print("k wavevector information saved in: \n", self.k_file)
        print("n(k,t) Data saved in: \n", self.nkt_hdf_file)
        print("Data saved in: \n", self.filename_hdf)
        print("Data accessible at: self.k_list, self.k_counts, self.ka_values, self.dataframe")
        print("\nSmallest wavevector k_min = 2 pi / L = 3.9 / N^(1/3)")
        print("k_min = {:.4f} / a_ws = {:.4e} ".format(self.ka_values[0], self.ka_values[0] / self.a_ws), end="")
        print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print("\nAngle averaging choice: {}".format(self.angle_averaging))
        if self.angle_averaging == "full":
            print("\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_aa_harmonics))
            print("\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)")
            print(
                "\tk_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_aa_ka_value, self.max_aa_ka_value / self.a_ws),
                end="",
            )
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == "custom":
            print("\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_aa_harmonics))
            print("\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)")
            print(
                "\tAA k_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_aa_ka_value, self.max_aa_ka_value / self.a_ws),
                end="",
            )
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

            print("\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_k_harmonics))
            print("\tLargest wavector k_max = k_min * n_x")
            print("\tk_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_ka_value, self.max_ka_value / self.a_ws), end="")
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == "principal_axis":
            print("\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}".format(*self.max_k_harmonics))
            print("\tLargest wavector k_max = k_min * n_x")
            print("\tk_max = {:.4f} / a_ws = {:1.4e} ".format(self.max_ka_value, self.max_ka_value / self.a_ws), end="")
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print("\nTotal number of k values to calculate = {}".format(len(self.k_list)))
        print("No. of unique ka values to calculate = {}".format(len(self.ka_values)))


class Thermodynamics(Observable):
    """
    Thermodynamic functions.
    """

    def __init__(self):
        super().__init__()
        self.__name__ = "therm"
        self.__long_name__ = "Thermodynamics"

    def setup(self, params, phase: str = None, **kwargs):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        params : sarkas.core.Parameters
            Simulation's parameters.

        phase : str, optional
            Phase to compute. Default = 'production'.

        **kwargs :
            These will overwrite any :attr:`sarkas.core.Parameters`
            or default :attr:`sarkas.tools.observables.Observable`
            attributes and/or add new ones.

        """

        super().setup_init(params, phase)
        if params.load_method == "restart":
            self.restart_sim = True
        else:
            self.restart_sim = False

        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):

        if self.phase.lower() == "production":
            self.saving_dir = self.production_dir
        elif self.phase.lower() == "equilibration":
            self.saving_dir = self.equilibration_dir
        elif self.phase.lower() == "magnetization":
            self.saving_dir = self.magnetization_dir

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

    def compute_from_rdf(self, rdf, potential):
        """
        Calculate the correlational energy and correlation pressure using
        :meth:`sarkas.tools.observables.RadialDistributionFunction.compute_sum_rule_integrals` method.

        The Hartree and correlational terms between species :math:`A` and :math:`B` are

        .. math::

            U_{AB}^{\\rm hartree} =  2 \\pi \\frac{N_iN_j}{V} \\int_0^\\infty dr \\, \\phi_{AB}(r) r^2 dr,

        .. math::

            U_{AB}^{\\rm corr} =  2 \\pi \\frac{N_iN_j}{V} \\int_0^\\infty dr \\, \\phi_{AB}(r) h(r) r^2 dr,

        .. math::

            P_{AB}^{\\rm hartree} =  - \\frac{2 \\pi}{3} \\frac{N_iN_j}{V^2} \\int_0^\\infty dr \\, \\frac{d\\phi_{AB}(r)}{dr} r^3 dr,

        .. math::

            P_{AB}^{\\rm corr} =  - \\frac{2 \\pi}{3} \\frac{N_iN_j}{V^2} \\int_0^\\infty dr \\, \\frac{d\\phi_{AB}(r)}{dr} h(r) r^3 dr,


        Parameters
        ----------
        rdf: sarkas.tools.observables.RadialDistributionFunction
            Radial Distribution Function object.

        potential: sarkas.potentials.core.Potential
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

        nkT = self.total_num_density / self.beta
        return nkT, u_hartree, u_corr, p_hartree, p_corr

    def parse(self, phase=None):
        """
        Grab the pandas dataframe from the saved csv file.
        """
        if phase:
            self.phase = phase.lower()

        if self.phase == "equilibration":
            self.dataframe = pd.read_csv(self.eq_energy_filename, index_col=False)
            self.fldr = self.equilibration_dir
        elif self.phase == "production":
            self.dataframe = pd.read_csv(self.prod_energy_filename, index_col=False)
            self.fldr = self.production_dir
        elif self.phase == "magnetization":
            self.dataframe = pd.read_csv(self.mag_energy_filename, index_col=False)
            self.fldr = self.magnetization_dir

        self.beta = 1.0 / (self.dataframe["Temperature"].mean() * self.kB)

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
                self.saving_dir = self.equilibration_dir
                self.no_steps = self.equilibration_steps
                self.parse(self.phase)
                self.dataframe = self.dataframe.iloc[1:, :]

            elif self.phase == "production":
                self.no_dumps = self.prod_no_dumps
                self.dump_dir = self.prod_dump_dir
                self.dump_step = self.prod_dump_step
                self.saving_dir = self.production_dir
                self.no_steps = self.production_steps
                self.parse(self.phase)

            elif self.phase == "magnetization":
                self.no_dumps = self.mag_no_dumps
                self.dump_dir = self.mag_dump_dir
                self.dump_step = self.mag_dump_step
                self.saving_dir = self.magnetization_dir
                self.no_steps = self.magnetization_steps
                self.parse(self.phase)

        else:
            self.parse()

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

            gs = GridSpec(4, 8)

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

        # Grab the color line list from the plt cycler. I will used this in the hist plots
        color_from_cycler = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # ------------------------------------------- Temperature -------------------------------------------#
        # Calculate Temperature plot's labels and multipliers
        time_mul, temp_mul, time_prefix, temp_prefix, time_lbl, temp_lbl = plot_labels(
            self.dataframe["Time"], self.dataframe["Temperature"], "Time", "Temperature", self.units
        )
        # Rescale quantities
        time = time_mul * self.dataframe["Time"]
        Temperature = temp_mul * self.dataframe["Temperature"]
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

        # This was a failed attempt to calculate the theoretical Temperature distribution.
        # Gaussian
        T_dist = scp_stats.norm(loc=T_desired, scale=Temperature.std())
        # Histogram plot
        sns.histplot(y=Temperature, bins="auto", stat="density", alpha=0.75, legend="False", ax=T_hist_plot)
        T_hist_plot.set(ylabel=None, xlabel=None, xticks=[], yticks=[])
        T_hist_plot.plot(T_dist.pdf(Temperature), Temperature, color=color_from_cycler[1])

        # ------------------------------------------- Total Energy -------------------------------------------#
        # Calculate Energy plot's labels and multipliers
        time_mul, energy_mul, _, _, time_lbl, energy_lbl = plot_labels(
            self.dataframe["Time"], self.dataframe["Total Energy"], "Time", "Energy", self.units
        )

        Energy = energy_mul * self.dataframe["Total Energy"]
        # Total Energy moving average
        E_cumavg = Energy.expanding().mean()
        # Total Energy Deviation and its moving average
        Delta_E = (Energy - Energy.iloc[0]) * 100 / Energy.iloc[0]
        Delta_E_cum_avg = Delta_E.expanding().mean()

        # Energy main plot
        E_main_plot.plot(time, Energy, alpha=0.7)
        E_main_plot.plot(time, E_cumavg, label="Moving Average")
        E_main_plot.axhline(Energy.mean(), ls="--", c="r", alpha=0.7, label="Avg")
        E_main_plot.legend(loc="best")
        E_main_plot.set(ylabel="Total Energy" + energy_lbl, xlabel="Time" + time_lbl)

        # Deviation Plot
        E_delta_plot.plot(time, Delta_E, alpha=0.5)
        E_delta_plot.plot(time, Delta_E_cum_avg, alpha=0.8)
        E_delta_plot.set(xticks=[], ylabel=r"Deviation [%]")

        # (Failed) Attempt to calculate the theoretical Energy distribution
        # In an NVT ensemble Energy fluctuation are given by sigma(E) = sqrt( k_B T^2 C_v)
        # where C_v is the isothermal heat capacity
        # Since this requires a lot of prior calculation I skip it and just make a Gaussian
        E_dist = scp_stats.norm(loc=Energy.mean(), scale=Energy.std())
        # Histogram plot
        sns.histplot(y=Energy, bins="auto", stat="density", alpha=0.75, legend="False", ax=E_hist_plot)
        # Grab the second color since the first is used for histplot
        E_hist_plot.plot(E_dist.pdf(Energy), Energy, color=color_from_cycler[1])

        E_hist_plot.set(ylabel=None, xlabel=None, xticks=[], yticks=[])

        if not publication:
            dt_mul, _, _, _, dt_lbl, _ = plot_labels(
                process.integrator.dt, self.dataframe["Total Energy"], "Time", "Energy", self.units
            )

            # Information section
            Info_plot.axis([0, 10, 0, 10])
            Info_plot.grid(False)

            Info_plot.text(0.0, 10, "Job ID: {}".format(self.job_id))
            Info_plot.text(0.0, 9.5, "Phase: {}".format(self.phase.capitalize()))
            Info_plot.text(0.0, 9.0, "No. of species = {}".format(len(self.species_num)))
            y_coord = 8.5
            for isp, sp in enumerate(process.species):
                Info_plot.text(0.0, y_coord, "Species {} : {}".format(isp + 1, sp.name))
                Info_plot.text(0.0, y_coord - 0.5, "  No. of particles = {} ".format(sp.num))
                Info_plot.text(
                    0.0, y_coord - 1.0, "  Temperature = {:.2f} {}".format(temp_mul * sp.temperature, temp_lbl)
                )
                y_coord -= 1.5

            y_coord -= 0.25
            delta_t = dt_mul * process.integrator.dt
            if info_list is None:
                info_list = [
                    "Total $N$ = {}".format(process.parameters.total_num_ptcls),
                    "Thermostat: {}".format(process.thermostat.type),
                    "  Berendsen rate = {:.2f}".format(process.thermostat.relaxation_rate),
                    "  Equilibration cycles = {}".format(
                        int(
                            process.parameters.equilibration_steps
                            * process.integrator.dt
                            * process.parameters.total_plasma_frequency
                        )
                    ),
                    "Potential: {}".format(process.potential.type),
                    "  Coupling Const = {:.2e}".format(process.parameters.coupling_constant),
                    "  Tot Force Error = {:.2e}".format(process.parameters.force_error),
                    "Integrator: {}".format(process.integrator.type),
                    "  $\Delta t$ = {:.2f} {}".format(delta_t, dt_lbl),
                    # "Step interval = {}".format(self.dump_step),
                    # "Step interval time = {:.2f} {}".format(self.dump_step * delta_t, dt_lbl),
                    "Completed steps = {}".format(completed_steps),
                    "Total steps = {}".format(self.no_steps),
                    "{:1.2f} % Completed".format(100 * completed_steps / self.no_steps),
                    # "Completed time = {:.2f} {}".format(completed_steps * delta_t / dt_mul * time_mul, time_lbl),
                    "Production time = {:.2f} {}".format(self.no_steps * delta_t / dt_mul * time_mul, time_lbl),
                    "Production cycles = {}".format(
                        int(
                            process.parameters.production_steps
                            * process.integrator.dt
                            * process.parameters.total_plasma_frequency
                        )
                    ),
                ]
            for itext, text_str in enumerate(info_list):
                Info_plot.text(0.0, y_coord, text_str)
                y_coord -= 0.5

            Info_plot.axis("off")

        if not publication:
            fig.tight_layout()

        # Saving
        if figname:
            fig.savefig(os.path.join(self.saving_dir, figname + "_" + self.job_id + ".png"))
        else:
            fig.savefig(os.path.join(self.saving_dir, "Plot_EnsembleCheck_" + self.job_id + ".png"))

        if show:
            fig.show()

        # Restore the previous rcParams
        plt.rcParams = current_rcParams


class VelocityAutoCorrelationFunction(Observable):
    """Velocity Auto-correlation function."""

    def __init__(self):
        super(VelocityAutoCorrelationFunction, self).__init__()
        self.__name__ = "vacf"
        self.__long_name__ = "Velocity AutoCorrelation Function"
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

    @compute_doc
    def compute(self):

        start_slice = 0
        end_slice = self.slice_steps * self.dump_step
        time = np.zeros(self.slice_steps)

        vacf_str = "VACF"
        t0 = self.timer.current()
        for isl in range(self.no_slices):
            print("\nCalculating vacf for slice {}/{}.".format(isl + 1, self.no_slices))
            # Parse the particles from the dump files
            vel = np.zeros((self.dimensions, self.total_num_ptcls, self.slice_steps))
            #
            for it, dump in enumerate(
                tqdm(range(start_slice, end_slice, self.dump_step), desc="Read in data", disable=not self.verbose)
            ):
                datap = load_from_restart(self.dump_dir, dump)
                time[it] = datap["time"]
                vel[0, :, it] = datap["vel"][:, 0]
                vel[1, :, it] = datap["vel"][:, 1]
                vel[2, :, it] = datap["vel"][:, 2]
            #
            if isl == 0:
                self.dataframe["Time"] = time
                self.dataframe_slices["Time"] = time
                self.dataframe_acf["Time"] = time
                self.dataframe_acf_slices["Time"] = time

            # Return an array of shape( num_species, dim + 1, slice_steps)
            vacf = calc_vacf(vel, self.species_num)

            for i, sp1 in enumerate(self.species_names):
                sp_vacf_str = "{} ".format(sp1) + vacf_str
                self.dataframe_acf_slices[sp_vacf_str + "_X_slice {}".format(isl)] = vacf[i, 0, :]
                self.dataframe_acf_slices[sp_vacf_str + "_Y_slice {}".format(isl)] = vacf[i, 1, :]
                self.dataframe_acf_slices[sp_vacf_str + "_Z_slice {}".format(isl)] = vacf[i, 2, :]
                self.dataframe_acf_slices[sp_vacf_str + "_Total_slice {}".format(isl)] = vacf[i, 3, :]

            start_slice += self.slice_steps * self.dump_step
            end_slice += self.slice_steps * self.dump_step

        # Average the stuff
        for i, sp1 in enumerate(self.species_names):
            sp_vacf_str = "{} ".format(sp1) + vacf_str
            xcol_str = [sp_vacf_str + "_X_slice {}".format(isl) for isl in range(self.no_slices)]
            ycol_str = [sp_vacf_str + "_Y_slice {}".format(isl) for isl in range(self.no_slices)]
            zcol_str = [sp_vacf_str + "_Z_slice {}".format(isl) for isl in range(self.no_slices)]
            tot_col_str = [sp_vacf_str + "_Total_slice {}".format(isl) for isl in range(self.no_slices)]

            self.dataframe_acf[sp_vacf_str + "_X_Mean"] = self.dataframe_acf_slices[xcol_str].mean(axis=1)
            self.dataframe_acf[sp_vacf_str + "_X_Std"] = self.dataframe_acf_slices[xcol_str].std(axis=1)
            self.dataframe_acf[sp_vacf_str + "_Y_Mean"] = self.dataframe_acf_slices[ycol_str].mean(axis=1)
            self.dataframe_acf[sp_vacf_str + "_Y_Std"] = self.dataframe_acf_slices[ycol_str].std(axis=1)
            self.dataframe_acf[sp_vacf_str + "_Z_Mean"] = self.dataframe_acf_slices[zcol_str].mean(axis=1)
            self.dataframe_acf[sp_vacf_str + "_Z_Std"] = self.dataframe_acf_slices[zcol_str].std(axis=1)
            self.dataframe_acf[sp_vacf_str + "_Total_Mean"] = self.dataframe_acf_slices[tot_col_str].mean(axis=1)
            self.dataframe_acf[sp_vacf_str + "_Total_Std"] = self.dataframe_acf_slices[tot_col_str].std(axis=1)

        self.save_hdf()
        tend = self.timer.current()
        self.time_stamp("VACF Calculation", self.timer.time_division(tend - t0))

    def pretty_print(self):
        """Print observable parameters for help in choice of simulation parameters."""

        print("\n\n{:=^70} \n".format(" " + self.__long_name__ + " "))
        print("Data saved in: \n", self.filename_acf_hdf)
        print("Data accessible at: self.dataframe_acf")

        print("\nNo. of slices = {}".format(self.no_slices))
        print("No. dumps per slice = {}".format(int(self.slice_steps / self.dump_step)))

        print(
            "Time interval of autocorrelation function = {:.4e} [s] ~ {} w_p T".format(
                self.dt * self.slice_steps, int(self.dt * self.slice_steps * self.total_plasma_frequency)
            )
        )


class VelocityDistribution(Observable):
    """
    Moments of the velocity distributions defined as

    .. math::
        \\langle v^{\\alpha} \\rangle = \\int_{-\\infty}^{\\infty} d v \, f(v) v^{2 \\alpha}.

    Attributes
    ----------
    no_bins: int
        Number of bins used to calculate the velocity distribution.

    plots_dir: str
        Directory in which to store Hermite coefficients plots.

    species_plots_dirs : list, str
        Directory for each species where to save Hermite coefficients plots.

    max_no_moment: int
        Maximum number of moments = :math:`\alpha`. Default = 3.

    """

    def __init__(self):
        super(VelocityDistribution, self).__init__()
        self.__name__ = "vd"
        self.__long_name__ = "Velocity Distribution"

    def setup(
        self,
        params,
        phase: str = None,
        no_slices: int = None,
        hist_kwargs: dict = None,
        max_no_moment: int = None,
        curve_fit_kwargs: dict = None,
        **kwargs
    ):

        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        params : sarkas.core.Parameters
            Simulation's parameters.

        phase : str, optional
            Phase to compute. Default = 'production'.

        no_slices : int, optional

        max_no_moment : int, optional
            Maximum number of moments to calculate. Default = 6.

        hist_kwargs : dict, optional
            Dictionary of keyword arguments to pass to ``np.histogram`` for the calculation of the distributions.

        curve_fit_kwargs: dict, optional
            Dictionary of keyword arguments to pass to ``scipy.curve_fit`` for fitting of Hermite coefficients.

        **kwargs :
            These will overwrite any :attr:`sarkas.core.Parameters`
            or default :attr:`sarkas.tools.observables.Observable`
            attributes and/or add new ones.

        """

        super().setup_init(params, self.phase)
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

        # Create the directory where to store the computed data
        # First the name of the observable
        saving_dir = os.path.join(self.postprocessing_dir, "VelocityDistribution")
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)
        # then the phase
        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())

        self.adjusted_dump_dir = []

        if self.multi_run_average:
            for r in range(self.runs):
                # Direct to the correct dumps directory
                dump_dir = os.path.join(
                    "run{}".format(r), os.path.join("Simulation", os.path.join(self.phase.capitalize(), "dumps"))
                )
                dump_dir = os.path.join(self.simulations_dir, dump_dir)
                self.adjusted_dump_dir.append(dump_dir)
            # Re-path the saving directory
            saving_dir = os.path.join(self.simulations_dir, "PostProcessing")
            if not os.path.exists(saving_dir):
                os.mkdir(saving_dir)
            saving_dir = os.path.join(saving_dir, self.phase.capitalize())
            if not os.path.exists(saving_dir):
                os.mkdir(saving_dir)
            self.saving_dir = os.path.join(saving_dir, "VelocityDistribution")
        else:
            self.adjusted_dump_dir = [self.dump_dir]

        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        # Directories in which to store plots
        self.plots_dir = os.path.join(self.saving_dir, "Plots")
        if not os.path.exists(self.plots_dir):
            os.mkdir(self.plots_dir)

        # Paths where to store the dataframes
        if self.multi_run_average:
            self.filename_csv = os.path.join(self.saving_dir, "VelocityDistribution.csv")
            self.filename_hdf = os.path.join(self.saving_dir, "VelocityDistribution.h5")
        else:
            self.filename_csv = os.path.join(self.saving_dir, "VelocityDistribution_" + self.job_id + ".csv")
            self.filename_hdf = os.path.join(self.saving_dir, "VelocityDistribution_" + self.job_id + ".h5")

        if hasattr(self, "max_no_moment"):
            self.moments_dataframe = None
            self.mom_df_filename_csv = os.path.join(self.saving_dir, "Moments_" + self.job_id + ".csv")

        if hasattr(self, "max_hermite_order"):
            self.hermite_dataframe = None
            self.herm_df_filename_csv = os.path.join(self.saving_dir, "HermiteCoefficients_" + self.job_id + ".csv")
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

    def grab_sim_data(self):
        """Read in velocity data"""

        # Velocity array for storing simulation data
        vel_raw = np.zeros((self.no_dumps, self.dim, self.runs * self.inv_dim * self.total_num_ptcls))
        time = np.zeros(self.no_dumps)

        print("\nCollecting data from snapshots ...")
        if self.dimensional_average:
            # Loop over the runs
            for r, dump_dir_r in enumerate(tqdm(self.adjusted_dump_dir, disable=(not self.verbose), desc="Runs Loop")):
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
                        vel_raw[it, 0, start_indx:end_indx] = datap["vel"][datap["names"] == sp_name].flatten("F")

                    time[it] = datap["time"]
        else:  # Dimensional Average = False
            # Loop over the runs
            for r, dump_dir_r in enumerate(tqdm(self.adjusted_dump_dir, disable=(not self.verbose), desc="Runs Loop")):
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
                        vel_raw[it, :, start_indx:end_indx] = datap["vel"][datap["names"] == sp_name].transpose()

                time[it] = datap["time"]

        return time, vel_raw

    def compute(self, compute_moments: bool = False, compute_Grad_expansion: bool = False):
        """
        Calculate the moments of the velocity distributions and save them to a pandas dataframes and csv.

        Parameters
        ----------
        hist_kwargs : dict, optional
            Dictionary with arguments to pass to ``numpy.histogram``.

        **kwargs :
            These will overwrite any :attr:`sarkas.core.Parameters`
            or default :attr:`sarkas.tools.observables.Observable`
            attributes and/or add new ones.

        """

        # Print info to screen
        self.pretty_print()

        # Grab simulation data
        time, vel_raw = self.grab_sim_data()

        # Make the velocity distribution
        self.create_distribution(vel_raw, time)

        # Calculate velocity moments
        if compute_moments:
            self.compute_moments(parse_data=False, vel_raw=vel_raw, time=time)
        #
        if compute_Grad_expansion:
            self.compute_hermite_expansion(compute_moments=False)

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
            energy_df = pd.read_csv(energy_fle, index_col=False, encoding="utf-8")
            if self.num_species > 1:
                vth = np.zeros(self.num_species)
                for sp, (sp_mass, sp_name) in enumerate(zip(self.species_masses, self.species_names)):
                    vth[sp] = np.sqrt(energy_df["{} Temperature".format(sp_name)].mean() * self.kB / sp_mass)
            else:
                vth = np.sqrt(energy_df["Temperature"].mean() * self.kB / self.species_masses)

        except FileNotFoundError:
            # In case you are using this in PreProcessing stage
            vth = np.sqrt(self.kB * self.T_desired / self.species_masses)

        self.vth = np.copy(vth)

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
        # I want the inverse a list whose elements are dicts
        self.list_hist_kwargs = []
        for indx in range(self.num_species):
            another_dict = {}
            # Loop over the keys and grab the species value
            for key, values in self.hist_kwargs.items():
                another_dict[key] = values[indx]

            self.list_hist_kwargs.append(another_dict)

    def create_distribution(self, vel_raw: np.ndarray = None, time: np.ndarray = None):
        """
        Calculate the velocity distribution of each species and save the corresponding dataframes.

        Parameters
        ----------
        vel_raw: np.ndarray, optional
            Container of particles velocity at each time step.

        time: np.ndarray, optional
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
        # using pd.MultiIndex.from_tuples([tuple(c.split("_")) for c in df.columns]) I can create a hierarchical df.
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
        dist_matrix = np.zeros((len(time), self.dim * (np.sum(self.hist_kwargs["bins"]) + self.num_species)))
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

                    bin_count, bin_edges = np.histogram(vel_raw[it, d, sp_start:sp_end], **self.list_hist_kwargs[indx])

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
        # Flatten with list(np.concatenate(columns).flat)
        # Now has become
        # first_column_row=['H_X', 'H_X', 'H_X', 'He_X', 'He_X', 'H_Y', 'H_Y', 'H_Y', 'He_Y', 'He_Y' ... Z-axis]
        # I think this is easier to understand than using nested list comprehension
        # see https://stackabuse.com/python-how-to-flatten-list-of-lists/
        full_df_columns = list(np.concatenate(full_df_columns).flat)
        self.dataframe = pd.DataFrame(dist_matrix, columns=full_df_columns)
        # Save it
        self.dataframe.to_csv(self.filename_csv, encoding="utf-8", index=False)

        # Hierarchical DataFrame
        self.hierarchical_dataframe = self.dataframe.copy()
        self.hierarchical_dataframe.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("_")) for c in self.hierarchical_dataframe.columns]
        )
        self.hierarchical_dataframe.to_hdf(self.filename_hdf, key=self.__name__, encoding="utf-8")

        tend = self.timer.current()
        self.time_stamp("Velocity distribution calculation", self.timer.time_division(tend - tinit))

    def compute_moments(self, parse_data: bool = False, vel_raw: np.ndarray = None, time: np.ndarray = None):
        """Calculate and save moments of the distribution.

        Parameters
        ----------
        parse_data: bool
            Flag for reading data. Default = False. If False, must pass ``vel_raw`` and ``time.
            If True it will parse data from simulations dumps.

        vel_raw: np.ndarray, optional
            Container of particles velocity at each time step.

        time: np.ndarray, optional
            Time array.

        """
        self.moments_dataframe = pd.DataFrame()
        self.moments_hdf_dataframe = pd.DataFrame()

        if parse_data:
            time, vel_raw = self.grab_sim_data()

        self.moments_dataframe["Time"] = time

        print("\nCalculating velocity moments ...")
        tinit = self.timer.current()
        moments, ratios = calc_moments(vel_raw, self.max_no_moment, self.species_index_start)
        tend = self.timer.current()
        self.time_stamp("Velocity moments calculation", self.timer.time_division(tend - tinit))

        # Save the dataframe
        if self.dimensional_average:
            for i, sp in enumerate(self.species_names):
                self.moments_hdf_dataframe["{}_X_Time".format(sp)] = time
                for m in range(self.max_no_moment):
                    self.moments_dataframe["{} {} moment".format(sp, m + 1)] = moments[i, :, :, m][:, 0]
                    self.moments_hdf_dataframe["{}_X_{} moment".format(sp, m + 1)] = moments[i, :, :, m][:, 0]
                for m in range(self.max_no_moment):
                    self.moments_dataframe["{} {} moment ratio".format(sp, m + 1)] = ratios[i, :, :, m][:, 0]
                    self.moments_hdf_dataframe["{}_X_{}-2 ratio".format(sp, m + 1)] = ratios[i, :, :, m][:, 0]
        else:
            for i, sp in enumerate(self.species_names):
                for d, ds in zip(range(self.dim), ["X", "Y", "Z"]):
                    self.moments_hdf_dataframe["{}_{}_Time".format(sp, ds)] = time
                    for m in range(self.max_no_moment):
                        self.moments_dataframe["{} {} moment axis {}".format(sp, m + 1, ds)] = moments[i, :, d, m][:, 0]
                        self.moments_hdf_dataframe["{}_{}_{} moment".format(sp, ds, m + 1)] = moments[i, :, :, m][:, 0]

                for d, ds in zip(range(self.dim), ["X", "Y", "Z"]):
                    self.moments_hdf_dataframe["{}_{}_Time".format(sp, ds)] = time
                    for m in range(self.max_no_moment):
                        self.moments_dataframe["{} {} moment ratio axis {}".format(sp, m + 1, ds)] = ratios[i, :, d, m][
                            :, 0
                        ]
                        self.moments_hdf_dataframe["{}_{}_{}-2 ratio ".format(sp, ds, m + 1)] = ratios[i, :, d, m][:, 0]

        self.moments_dataframe.to_csv(self.filename_csv, index=False, encoding="utf-8")
        # Hierarchical DF Save
        # Make the columns
        self.moments_hdf_dataframe.columns = pd.MultiIndex.from_tuples(
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

        self.hermite_dataframe = pd.DataFrame()
        self.hermite_hdf_dataframe = pd.DataFrame()

        if compute_moments:
            self.compute_moments(parse_data=True)

        if not hasattr(self, "hermite_rms_tol"):
            self.hermite_rms_tol = 0.05

        self.hermite_dataframe["Time"] = np.copy(self.moments_dataframe["Time"])
        self.hermite_sigmas = np.zeros((self.num_species, self.dim, len(self.hermite_dataframe["Time"])))
        self.hermite_epochs = np.zeros((self.num_species, self.dim, len(self.hermite_dataframe["Time"])))
        hermite_coeff = np.zeros(
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
                        norm = np.trapz(dist, x=v_bins / vrms)

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
        self.hermite_hdf_dataframe.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("_")) for c in self.hermite_hdf_dataframe.columns]
        )
        # Save the df in the hierarchical df with a new key/group
        self.hermite_hdf_dataframe.to_hdf(self.filename_hdf, mode="a", key="hermite_coefficients", encoding="utf-8")

        self.time_stamp("Hermite expansion calculation", self.timer.time_division(tend - tinit))

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
    Sk_raw = np.zeros((no_sk, len(k_counts), no_dumps))
    pair_indx = 0
    for ip, si in enumerate(species_np):
        for jp in range(ip, len(species_np)):
            sj = species_np[jp]
            dens_const = 1.0 / np.sqrt(si * sj)
            for it in range(no_dumps):
                for ik, ka in enumerate(k_list):
                    indx = int(ka[-1])
                    nk_i = nkt[ip, it, ik]
                    nk_j = nkt[jp, it, ik]
                    Sk_raw[pair_indx, indx, it] += np.real(np.conj(nk_i) * nk_j) * dens_const / k_counts[indx]
            pair_indx += 1

    return Sk_raw


def calc_Skw(nkt, ka_list, species_np, no_dumps, dt, dump_step):
    """
    Calculate the Fourier transform of the correlation function of ``nkt``.

    Parameters
    ----------
    dt : float
        Time interval.

    dump_step : int
        Snapshot interval.

    nkt :  complex, numpy.ndarray
        Particles' density or velocity fluctuations.
        Shape = ( ``no_species``, ``no_dumps``, ``no_k_list``)

    ka_list : list
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``ka_counts``.
        Shape=(``no_ka_values``, 5)

    ka_counts : numpy.ndarray
        Number of times each :math:`k` magnitude appears.

    species_np : numpy.ndarray
        Array with one element giving number of particles.

    no_dumps : int
        Number of dumps.

    Returns
    -------
    Skw : numpy.ndarray
        DSF/CCF of each species and pair of species.
        Shape = (``no_skw``, ``no_ka_values``, ``no_dumps``)
    """
    # Fourier transform normalization: norm = dt / Total time
    norm = dt / np.sqrt(no_dumps * dt * dump_step)
    # number of independent observables
    no_skw = int(len(species_np) * (len(species_np) + 1) / 2)
    # DSF
    # Skw = np.zeros((no_skw, len(ka_counts), no_dumps))
    Skw_all = np.zeros((no_skw, len(ka_list), no_dumps))
    pair_indx = 0
    for ip, si in enumerate(species_np):
        for jp in range(ip, len(species_np)):
            sj = species_np[jp]
            dens_const = 1.0 / np.sqrt(si * sj)
            for ik, ka in enumerate(ka_list):
                # indx = int(ka[-1])
                nkw_i = np.fft.fft(nkt[ip, :, ik]) * norm
                nkw_j = np.fft.fft(nkt[jp, :, ik]) * norm
                Skw_all[pair_indx, ik, :] = np.fft.fftshift(np.real(np.conj(nkw_i) * nkw_j) * dens_const)
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

    Js = np.zeros((sp_num.shape[0], no_dim, no_dumps))
    Jtot = np.zeros((no_dim, no_dumps))

    sp_start = 0
    sp_end = 0
    for s, (q_sp, n_sp) in enumerate(zip(sp_charge, sp_num)):
        # Find the index of the last particle of species s
        sp_end += n_sp
        # Calculate the current of each species
        Js[s, :, :] = q_sp * np.sum(vel[:, :, sp_start:sp_end], axis=-1)
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
        Ratios of each moments with respect to the expected Maxwellian value.
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
    moments = np.zeros((no_species, no_dumps, dim, max_moment))
    ratios = np.zeros((no_species, no_dumps, dim, max_moment))

    for indx, sp_start in enumerate(species_index_start[:-1]):
        # Calculate the correct start and end index for storage
        sp_end = species_index_start[indx + 1]

        for mom in range(max_moment):
            moments[indx, :, :, mom] = scp_stats.moment(dist[:, :, sp_start:sp_end], moment=mom + 1, axis=2)

    # sqrt( <v^2> ) = standard deviation = moments[:, :, :, 1] ** (1/2)
    for mom in range(max_moment):
        pwr = mom + 1
        const = 2.0 ** (pwr / 2) * scp_gamma((pwr + 1) / 2) / np.sqrt(np.pi)
        ratios[:, :, :, mom] = moments[:, :, :, mom] / (const * moments[:, :, :, 1] ** (pwr / 2.0))

    return moments, ratios


@njit
def calc_nk(pos_data, k_list):
    """
    Calculate the instantaneous microscopic density :math:`n(k)` defined as

    .. math::
        n_{A} ( k ) = \sum_i^{N_A} \exp [ -i \mathbf k \cdot \mathbf r_{i} ]

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

    nk = np.zeros(len(k_list), dtype=np.complex128)

    for ik, k_vec in enumerate(k_list):
        kr_i = 2.0 * np.pi * (k_vec[0] * pos_data[:, 0] + k_vec[1] * pos_data[:, 1] + k_vec[2] * pos_data[:, 2])
        nk[ik] = np.sum(np.exp(-1j * kr_i))

    return nk


def calc_nkt(fldr, slices, dump_step, species_np, k_list, verbose):
    """
    Calculate density fluctuations :math:`n(k,t)` of all species.

    .. math::
        n_{A} ( k, t ) = \sum_i^{N_A} \exp [ -i \mathbf k \cdot \mathbf r_{i}(t) ]

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
    nkt = np.zeros((len(species_np), slices[2], len(k_list)), dtype=np.complex128)
    for it, dump in enumerate(tqdm(range(slices[0], slices[1], dump_step), disable=not verbose)):
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
def calc_pressure_tensor(vel, virial, species_mass, species_np, box_volume):
    """
    Calculate the pressure tensor.

    Parameters
    ----------
    pos : numpy.ndarray
        Particles' positions.

    vel : numpy.ndarray
        Particles' velocities.

    acc : numpy.ndarray
        Particles' accelerations.

    species_mass : numpy.ndarray
        Mass of each species.

    species_np : numpy.ndarray
        Number of particles of each species.

    box_volume : float
        Volume of simulation's box.

    Returns
    -------
    pressure : float
        Scalar Pressure i.e. trace of the pressure tensor

    pressure_tensor : numpy.ndarray
        Pressure tensor. Shape(``no_dim``,``no_dim``)

    """
    sp_start = 0
    sp_end = 0

    # Rescale vel of each particle by their individual mass
    for sp, num in enumerate(species_np):
        sp_end += num
        vel[sp_start:sp_end, :] *= np.sqrt(species_mass[sp])
        sp_start += num

    pressure_kin = (vel.transpose() @ vel) / box_volume
    pressure_pot = virial.sum(axis=-1) / box_volume
    pressure_tensor = pressure_kin + pressure_pot

    pressure = np.trace(pressure_tensor) / 3.0

    return pressure, pressure_kin, pressure_pot, pressure_tensor


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
    tau_blk = np.zeros(max_no_divisions)
    sigma2_blk = np.zeros(max_no_divisions)
    statistical_efficiency = np.zeros(max_no_divisions)
    for i in range(2, max_no_divisions):
        tau_blk[i] = int(no_dumps / i)
        for j in range(i):
            t_start = int(j * tau_blk[i])
            t_end = int((j + 1) * tau_blk[i])
            blk_avg = observable[t_start:t_end].mean()
            sigma2_blk[i] += (blk_avg - run_avg) ** 2
        sigma2_blk[i] /= i - 1
        statistical_efficiency[i] = tau_blk[i] * sigma2_blk[i] / run_std ** 2

    return tau_blk, sigma2_blk, statistical_efficiency


# @jit Numba doesn't like scipy.signal
def calc_diff_flux_acf(vel, sp_num, sp_conc, sp_mass):
    """
    Calculate the diffusion fluxes and their autocorrelations functions in each direction.

    Parameters
    ----------
    vel : numpy.ndarray
        Particles' velocities. Shape = (``dimensions``, ``no_dumps``, ``total_num_ptcls``)

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
    tot_vel = np.zeros((no_species, no_dim, no_dumps))

    sp_start = 0
    sp_end = 0
    # Calculate the total center of mass velocity (tot_com_vel)
    # and the center of mass velocity of each species (com_vel)
    for i, ns in enumerate(sp_num):
        sp_end += ns
        tot_vel[i, :, :] = np.sum(vel[:, :, sp_start:sp_end], axis=-1)
        # tot_com_vel += mass_densities[i] * com_vel[i, :, :] / tot_mass_dens
        sp_start += ns

    # Diffusion Fluxes
    J_flux = np.zeros((no_species - 1, no_dim, no_dumps))

    # Relative Diffusion fluxes for ACF and Transport calc
    jr_flux = np.zeros((no_species - 1, no_dim, no_dumps))

    # Relative diff flux acf
    jr_acf = np.zeros((no_jc_acf, no_dim + 1, no_dumps))

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


# @jit Numba doesn't like Scipy
def calc_vacf(vel, sp_num):
    """
    Calculate the velocity autocorrelation function of each species and in each direction.

    Parameters
    ----------
    vel : numpy.ndarray
        Particles' velocities stored in a 3D array with shape = (D x Np x Nt).
        D = cartesian dimensions, Np = Number of particles, Nt = number of dumps.

    sp_num: numpy.ndarray
        Number of particles of each species.

    Returns
    -------
    vacf: numpy.ndarray
        Velocity autocorrelation functions. Shape= (No_species, D + 1, Nt)

    """
    no_dim = vel.shape[0]
    no_dumps = vel.shape[2]

    vacf = np.zeros((len(sp_num), no_dim + 1, no_dumps))

    # Calculate the vacf of each species in each dimension
    for i in range(no_dim):
        sp_start = 0
        sp_end = 0
        for sp, n_sp in enumerate(sp_num):
            sp_end += n_sp
            # Temporary species vacf
            sp_vacf = np.zeros(no_dumps)
            # Calculate the vacf for each particle of species sp
            for ptcl in range(sp_start, sp_end):
                # Calculate the correlation function and add it to the array
                ptcl_vacf = correlationfunction(vel[i, ptcl, :], vel[i, ptcl, :])

                # Add this particle vacf to the species vacf and normalize by the time origins
                sp_vacf += ptcl_vacf

            # Save the species vacf for dimension i
            vacf[sp, i, :] = sp_vacf / n_sp
            # Save the total vacf
            vacf[sp, -1, :] += sp_vacf / n_sp
            # Move to the next species first particle position
            sp_start += n_sp

    return vacf


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
    vk = np.zeros(len(k_list), dtype=np.complex128)

    # Transverse
    vk_i = np.zeros(len(k_list), dtype=np.complex128)
    vk_j = np.zeros(len(k_list), dtype=np.complex128)
    vk_k = np.zeros(len(k_list), dtype=np.complex128)

    for ik, k_vec in enumerate(k_list):
        # Calculate the dot product and cross product between k, r, and v
        kr_i = 2.0 * np.pi * (k_vec[0] * pos_data[:, 0] + k_vec[1] * pos_data[:, 1] + k_vec[2] * pos_data[:, 2])
        k_dot_v = 2.0 * np.pi * (k_vec[0] * vel_data[:, 0] + k_vec[1] * vel_data[:, 1] + k_vec[2] * vel_data[:, 2])

        k_cross_v_i = 2.0 * np.pi * (k_vec[1] * vel_data[:, 2] - k_vec[2] * vel_data[:, 1])
        k_cross_v_j = -2.0 * np.pi * (k_vec[0] * vel_data[:, 2] - k_vec[2] * vel_data[:, 0])
        k_cross_v_k = 2.0 * np.pi * (k_vec[0] * vel_data[:, 1] - k_vec[1] * vel_data[:, 0])

        # Microscopic longitudinal current
        vk[ik] = np.sum(k_dot_v * np.exp(-1j * kr_i))
        # Microscopic transverse current
        vk_i[ik] = np.sum(k_cross_v_i * np.exp(-1j * kr_i))
        vk_j[ik] = np.sum(k_cross_v_j * np.exp(-1j * kr_i))
        vk_k[ik] = np.sum(k_cross_v_k * np.exp(-1j * kr_i))

    return vk, vk_i, vk_j, vk_k


def calc_vkt(fldr, slices, dump_step, species_np, k_list, verbose):
    """
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
    vkt_par = np.zeros((len(species_np), no_dumps, len(k_list)), dtype=np.complex128)
    vkt_perp_i = np.zeros((len(species_np), no_dumps, len(k_list)), dtype=np.complex128)
    vkt_perp_j = np.zeros((len(species_np), no_dumps, len(k_list)), dtype=np.complex128)
    vkt_perp_k = np.zeros((len(species_np), no_dumps, len(k_list)), dtype=np.complex128)
    for it, dump in enumerate(tqdm(range(slices[0], slices[1], dump_step), disable=not verbose)):
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
        Hermite coefficients withouth the division by factorial.

    Returns
    -------
    _ : numpy.ndarray
        Grad expansion.

    """
    gaussian = np.exp(-0.5 * (x / rms) ** 2) / (np.sqrt(2.0 * np.pi * rms ** 2))

    herm_coef = h_coeff / [np.math.factorial(i) for i in range(len(h_coeff))]
    hermite_series = np.polynomial.hermite_e.hermeval(x, herm_coef)

    return gaussian * hermite_series


def calculate_herm_coeff(v, distribution, maxpower):
    """
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

    coeff = np.zeros(maxpower + 1)

    for i in range(maxpower + 1):
        hc = np.zeros(1 + i)
        hc[-1] = 1.0
        Hp = np.polynomial.hermite_e.hermeval(v, hc)
        coeff[i] = np.trapz(distribution * Hp, x=v)

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
            np.array([i / box_lengths[0], j / box_lengths[1], k / box_lengths[2]])
            for i in range(max_k_harmonics[0] + 1)
            for j in range(max_k_harmonics[1] + 1)
            for k in range(max_k_harmonics[2] + 1)
        ]
        harmonics = [
            np.array([i, j, k], dtype=np.int)
            for i in range(max_k_harmonics[0] + 1)
            for j in range(max_k_harmonics[1] + 1)
            for k in range(max_k_harmonics[2] + 1)
        ]
    elif angle_averaging == "principal_axis":
        # The first value of k_arr = [1, 0, 0]
        first_non_zero = 0
        # Calculate the k vectors along the principal axis only
        k_arr = [np.array([i / box_lengths[0], 0, 0]) for i in range(1, max_k_harmonics[0] + 1)]
        harmonics = [np.array([i, 0, 0], dtype=np.int) for i in range(1, max_k_harmonics[0] + 1)]
        k_arr = np.append(k_arr, [np.array([0, i / box_lengths[1], 0]) for i in range(1, max_k_harmonics[1] + 1)], axis=0)
        harmonics = np.append(
            harmonics, [np.array([0, i, 0], dtype=np.int) for i in range(1, max_k_harmonics[1] + 1)], axis=0
        )

        k_arr = np.append(k_arr, [np.array([0, 0, i / box_lengths[2]]) for i in range(1, max_k_harmonics[2] + 1)], axis=0)
        harmonics = np.append(
            harmonics, [np.array([0, 0, i], dtype=np.int) for i in range(1, max_k_harmonics[2] + 1)], axis=0
        )

    elif angle_averaging == "custom":
        # The first value of k_arr = [0, 0, 0]
        first_non_zero = 1
        # Obtain all possible permutations of the wave number arrays up to max_aa_harmonics included
        k_arr = [
            np.array([i / box_lengths[0], j / box_lengths[1], k / box_lengths[2]])
            for i in range(max_aa_harmonics[0] + 1)
            for j in range(max_aa_harmonics[1] + 1)
            for k in range(max_aa_harmonics[2] + 1)
        ]

        harmonics = [
            np.array([i, j, k], dtype=np.int)
            for i in range(max_aa_harmonics[0] + 1)
            for j in range(max_aa_harmonics[1] + 1)
            for k in range(max_aa_harmonics[2] + 1)
        ]
        # Append the rest of k values calculated from principal axis
        k_arr = np.append(
            k_arr,
            [np.array([i / box_lengths[0], 0, 0]) for i in range(max_aa_harmonics[0] + 1, max_k_harmonics[0] + 1)],
            axis=0,
        )
        harmonics = np.append(
            harmonics,
            [np.array([i, 0, 0], dtype=np.int) for i in range(max_aa_harmonics[0] + 1, max_k_harmonics[0] + 1)],
            axis=0,
        )

        k_arr = np.append(
            k_arr,
            [np.array([0, i / box_lengths[1], 0]) for i in range(max_aa_harmonics[1] + 1, max_k_harmonics[1] + 1)],
            axis=0,
        )
        harmonics = np.append(
            harmonics,
            [np.array([0, i, 0], dtype=np.int) for i in range(max_aa_harmonics[1] + 1, max_k_harmonics[1] + 1)],
            axis=0,
        )

        k_arr = np.append(
            k_arr,
            [np.array([0, 0, i / box_lengths[2]]) for i in range(max_aa_harmonics[2] + 1, max_k_harmonics[2] + 1)],
            axis=0,
        )
        harmonics = np.append(
            harmonics,
            [np.array([0, 0, i], dtype=np.int) for i in range(max_aa_harmonics[2] + 1, max_k_harmonics[2] + 1)],
            axis=0,
        )
    # Compute wave number magnitude - don't use |k| (skipping first entry in k_arr)
    # The round off is needed to avoid ka value different beyond a certain significant digit. It will break other parts
    # of the code otherwise.
    k_mag = np.sqrt(np.sum(np.array(k_arr) ** 2, axis=1)[..., None])
    harm_mag = np.sqrt(np.sum(np.array(harmonics) ** 2, axis=1)[..., None])
    for i, k in enumerate(k_mag[:-1]):
        if abs(k - k_mag[i + 1]) < 2.0e-5:
            k_mag[i + 1] = k
    # Add magnitude to wave number array
    k_arr = np.concatenate((k_arr, k_mag), 1)
    # Add magnitude to wave number array
    harmonics = np.concatenate((harmonics, harm_mag), 1)

    # Sort from lowest to highest magnitude
    ind = np.argsort(k_arr[:, -1])
    k_arr = k_arr[ind]
    harmonics = harmonics[ind]

    # Count how many times a |k| value appears
    k_unique, k_counts = np.unique(k_arr[first_non_zero:, -1], return_counts=True)

    # Generate a 1D array containing index to be used in S array
    k_index = np.repeat(range(len(k_counts)), k_counts)[..., None]

    # Add index to k_array
    k_arr = np.concatenate((k_arr[int(first_non_zero) :, :], k_index), 1)
    harmonics = np.concatenate((harmonics[int(first_non_zero) :, :], k_index), 1)

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

    file_name = os.path.join(fldr, "checkpoint_" + str(it) + ".npz")
    data = np.load(file_name, allow_pickle=True)
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
    if isinstance(xdata, (np.ndarray, pd.core.series.Series)):
        xmax = xdata.max()
    else:
        xmax = xdata

    if isinstance(ydata, (np.ndarray, pd.core.series.Series)):
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
    x_str = np.format_float_scientific(xmax)
    y_str = np.format_float_scientific(ymax)

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
