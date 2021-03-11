"""
Module for calculating physical quantities from Sarkas checkpoints.
"""
from IPython import get_ipython

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from numba import njit
from matplotlib.gridspec import GridSpec

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import h5py
# import logging

import scipy.signal as scp_signal
import scipy.stats as scp_stats

from sarkas.utilities.timing import SarkasTimer
from sarkas.utilities.io import num_sort

UNITS = [
    # MKS Units
    {"Energy": 'J',
     "Time": 's',
     "Length": 'm',
     "Charge": 'C',
     "Temperature": 'K',
     "ElectronVolt": 'eV',
     "Mass": 'kg',
     "Magnetic Field": 'T',
     "Current": "A",
     "Power": "erg/s",
     "Pressure": "Pa",
     "Conductivity": "S/m",
     "Diffusion": r"m$^2$/s",
     "Viscosity": r"Pa s",
     "none": ""},
    # CGS Units
    {"Energy": 'erg',
     "Time": 's',
     "Length": 'm',
     "Charge": 'esu',
     "Temperature": 'K',
     "ElectronVolt": 'eV',
     "Mass": 'g',
     "Magnetic Field": 'G',
     "Current": "esu/s",
     "Power": "erg/s",
     "Pressure": "Ba",
     "Conductivity": "mho/m",
     "Diffusion": r"m$^2$/s",
     "Viscosity": r"Ba s",
     "none": ""}
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
    "Y": 1e24  # yotta
}


class Observable:
    """
    Parent class of all the observables.

    Attributes
    ----------
    dataframe : pandas.DataFrame
        Dataframe containing the computed data.

    dataframe_longitudinal : pandas.DataFrame
        Dataframe containing the longitudinal part of the computed observable.

    dataframe_transverse : pandas.DataFrame
        Dataframe containing the transverse part of the computed observable.

    max_k_harmonics : list
        Maximum number of :math:`\mathbf{k}` harmonics to calculatealong each dimension.

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
        It is either ``sarkas.base.Parameters.prod_dump_step`` or ``sarkas.base.Parameters.eq_dump_step``.

    no_obs : int
        Number of independent binary observable quantities.
        It is calculated as :math:`N_s (N_s + 1) / 2` where :math: `N_s` is the number of species.

    k_file : str
        Path to the npz file storing the :math:`k` vector values.

    nkt_file : str
        Path to the npy file containing the Fourier transform of density fluctuations. :math:`n(\mathbf k, t)`.

    vkt_file : str
        Path to the npz file containing the Fourier transform of velocity fluctuations. :math:`\mathbf v(\mathbf k, t)`.

    k_space_dir : str
        Directory where :math:`\mathbf {k}` data is stored.

    filename_csv_longitudinal : str
        Path to to the csv file containing the longitudinal part of the computed observable.

    filename_csv_transverse : str
        Path to the csv file containing the transverse part of the computed observable.

    saving_dir : str
        Path to the directory where computed data is stored.

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
        self.dataframe = pd.DataFrame()
        self.dataframe_longitudinal = pd.DataFrame()
        self.dataframe_transverse = pd.DataFrame()
        self.saving_dir = None
        self.filename_csv = None
        self.filename_csv_longitudinal = None
        self.filename_csv_transverse = None
        self.phase = 'production'
        self.multi_run_average = False
        self.dimensional_average = False
        self.runs = 1
        self.screen_output = True
        self.timer = SarkasTimer()
        self.k_observable = False

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = 'Observable( ' + self.__class__.__name__ + '\n'
        for key, value in sortedDict.items():
            if not key in ['dataframe', 'dataframe_longitudinal', 'dataframe_transverse']:
                disp += "\t{} : {}\n".format(key, value)
        disp += ')'
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

    def setup_init(self, params, phase):
        """Assign Observables attributes and copy the simulation's parameters.

        Parameters
        ----------
        params : sarkas.base.Parameters
            Simulation's Parameters.

        phase : str
            Phase to compute.

        """

        if phase:
            self.phase = phase.lower()

        self.__dict__.update(params.__dict__)

        if self.k_observable:
            # Check for k space information.
            if not hasattr(self, 'angle_averaging'):
                self.angle_averaging = 'principal_axis'
                self.max_aa_harmonics = np.array([0, 0, 0])

            if self.angle_averaging == 'custom':
                if not hasattr(self, 'max_aa_ka_value'):
                    assert self.max_aa_harmonics, 'max_aa_harmonics and max_aa_ka_value not defined.'
                elif not hasattr(self, 'max_aa_harmonics'):
                    assert self.max_aa_ka_value, 'max_aa_harmonics and max_aa_ka_value not defined.'

            # More checks on k attributes and initialization of k vectors

            # Dev Notes:
            #           Make sure that max_k_harmonics and max_aa_harmonics are defined once this if is done.
            #           The user can either define max_k_harmonics or max_ka_value
            #           Based on this choice the user can define max_aa_harmonics or max_aa_ka_value

            if hasattr(self, 'max_k_harmonics'):
                # Convert max_k_harmonics to a numpy array
                if isinstance(self.max_k_harmonics, np.ndarray) == 0:
                    self.max_k_harmonics = np.ones(3, dtype=int) * self.max_k_harmonics

                # Calculate max_aa_harmonics based on the choice of angle averaging and inputs
                if self.angle_averaging == 'full':
                    self.max_aa_harmonics = np.copy(self.max_k_harmonics)

                elif self.angle_averaging == 'custom':
                    # Check if the user has defined the max_aa_harmonics
                    if self.max_aa_ka_value:
                        nx = int(self.max_aa_ka_value * self.box_lengths[0] / (2.0 * np.pi * self.a_ws * np.sqrt(3.0)))
                        self.max_aa_harmonics = np.array([nx, nx, nx])
                    # else max_aa_harmonics is user defined
                elif self.angle_averaging == 'principal_axis':
                    self.max_aa_harmonics = np.array([0, 0, 0])

            elif hasattr(self, 'max_ka_value'):
                # Calculate max_k_harmonics from max_ka_value

                # Check for angle_averaging choice
                if self.angle_averaging == 'full':
                    # The maximum value is calculated assuming that max nx = max ny = max nz
                    # ka_max = 2pi a/L sqrt( nx^2 + ny^2 + nz^2) = 2pi a/L nx sqrt(3)
                    nx = int(self.max_ka_value * self.box_lengths[0] / (2.0 * np.pi * self.a_ws * np.sqrt(3.0)))
                    self.max_k_harmonics = np.array([nx, nx, nx])
                    self.max_aa_harmonics = np.array([nx, nx, nx])

                elif self.angle_averaging == 'custom':
                    # ka_max = 2pi a/L sqrt( nx^2 + 0 + 0) = 2pi a/L nx
                    nx = int(self.max_ka_value * self.box_lengths[0] / (2.0 * np.pi * self.a_ws))
                    self.max_k_harmonics = np.array([nx, nx, nx])
                    # Check if the user has defined the max_aa_harmonics
                    if self.max_aa_ka_value:
                        nx = int(self.max_aa_ka_value * self.box_lengths[0] / (2.0 * np.pi * self.a_ws * np.sqrt(3.0)))
                        self.max_aa_harmonics = np.array([nx, nx, nx])
                    # else max_aa_harmonics is user defined
                elif self.angle_averaging == 'principal_axis':
                    # ka_max = 2pi a/L sqrt( nx^2 + 0 + 0) = 2pi a/L nx
                    nx = int(self.max_ka_value * self.box_lengths[0] / (2.0 * np.pi * self.a_ws))
                    self.max_k_harmonics = np.array([nx, nx, nx])
                    self.max_aa_harmonics = np.array([0, 0, 0])

            else:
                # Executive decision
                self.max_k_harmonics = np.array([5, 5, 5])
                self.max_aa_harmonics = np.array([5, 5, 5])
                self.angle_averaging = 'full'

            # Calculate the maximum ka value based on user's choice of angle_averaging
            # Dev notes: Make sure max_ka_value, max_aa_ka_value are defined when this if is done
            if self.angle_averaging == 'full':
                self.max_ka_value = 2.0 * np.pi * self.a_ws * np.linalg.norm(self.max_k_harmonics / self.box_lengths)
                self.max_aa_ka_value = 2.0 * np.pi * self.a_ws * np.linalg.norm(self.max_k_harmonics / self.box_lengths)

            elif self.angle_averaging == 'principal_axis':
                self.max_ka_value = 2.0 * np.pi * self.a_ws * self.max_k_harmonics[0] / self.box_lengths[0]
                self.max_aa_ka_value = 0.0

            elif self.angle_averaging == 'custom':
                self.max_aa_ka_value = 2.0 * np.pi * self.a_ws * np.linalg.norm(
                    self.max_aa_harmonics / self.box_lengths)
                self.max_ka_value = 2.0 * np.pi * self.a_ws * self.max_k_harmonics[0] / self.box_lengths[0]

            # Create paths for files
            self.k_space_dir = os.path.join(self.postprocessing_dir, "k_space_data")
            self.k_file = os.path.join(self.k_space_dir, "k_arrays.npz")
            self.nkt_file = os.path.join(self.k_space_dir, "nkt")
            self.vkt_file = os.path.join(self.k_space_dir, "vkt")

        # Get the number of independent observables if multi-species
        self.no_obs = int(self.num_species * (self.num_species + 1) / 2)

        # Get the total number of dumps by looking at the files in the directory
        self.prod_no_dumps = len(os.listdir(self.prod_dump_dir))
        self.eq_no_dumps = len(os.listdir(self.eq_dump_dir))
        # Check for magnetized plasma options
        if self.magnetized and self.electrostatic_equilibration:
            self.mag_no_dumps = len(os.listdir(self.mag_dump_dir))

        # Assign dumps variables based on the choice of phase
        if self.phase == 'equilibration':
            self.no_dumps = self.eq_no_dumps
            self.dump_step = self.eq_dump_step
            self.no_steps = self.equilibration_steps
            self.dump_dir = self.eq_dump_dir

        elif self.phase == 'production':
            self.no_dumps = self.prod_no_dumps
            self.dump_step = self.prod_dump_step
            self.no_steps = self.production_steps
            self.dump_dir = self.prod_dump_dir

        elif self.phase == 'magnetization':
            self.no_dumps = self.mag_no_dumps
            self.dump_step = self.mag_dump_step
            self.no_steps = self.magnetization_steps
            self.dump_dir = self.mag_dump_dir

        if not hasattr(self, 'no_slices'):
            self.no_slices = 1

        self.slice_steps = int(self.no_dumps / self.no_slices)

        # Array containing the start index of each species. The last value is equivalent to vel_raw.shape[-1]
        self.species_index_start = np.array([0, *np.cumsum(self.species_num)], dtype=int)

        # Logger
        # log_file = os.path.join(self.postprocessing_dir, 'Logger_PostProcessing_' + self.job_id + '.out')
        # logging.basicConfig(filename=log_file,
        #                     filemode='a',
        #                     # format='%(levelname)s:%(message)s',
        #                     # format='%(levelname)s: %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
        #                     level=logging.INFO)

    def parse(self):
        """
        Grab the pandas dataframe from the saved csv file. If file does not exist call ``compute``.
        """
        if self.__class__.__name__ == 'CurrentCorrelationFunction':
            try:
                self.dataframe_longitudinal = pd.read_csv(self.filename_csv_longitudinal, index_col=False)
                self.dataframe_transverse = pd.read_csv(self.filename_csv_transverse, index_col=False)
                k_data = np.load(self.k_file)
                self.k_list = k_data["k_list"]
                self.k_counts = k_data["k_counts"]
                self.ka_values = k_data["ka_values"]

            except FileNotFoundError:
                print("\nFile {} not found!".format(self.filename_csv_longitudinal))
                print("\nFile {} not found!".format(self.filename_csv_transverse))
                print("\nComputing Observable now ...")
                self.compute()
        else:
            try:
                self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
            except FileNotFoundError:
                print("\nFile {} not found!".format(self.filename_csv))
                print("\nComputing Observable now ...")
                self.compute()

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
                    self.k_counts = k_data["k_counts"]
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
        assert isinstance(self.angle_averaging, str), "angle_averaging not a string. Choose from ['full', 'custom', 'principal_axis']"
        assert self.max_k_harmonics.all(), 'max_k_harmonics not defined.'

        # Calculate the k arrays
        self.k_list, self.k_counts, k_unique = kspace_setup(self.box_lengths,
                                                            self.angle_averaging,
                                                            self.max_k_harmonics,
                                                            self.max_aa_harmonics)
        # Save the ka values
        self.ka_values = 2.0 * np.pi * k_unique * self.a_ws
        self.no_ka_values = len(self.ka_values)

        # Check if the writing folder exist
        if not (os.path.exists(self.k_space_dir)):
            os.mkdir(self.k_space_dir)

        # Write the npz file
        np.savez(self.k_file,
                 k_list=self.k_list,
                 k_counts=self.k_counts,
                 ka_values=self.ka_values,
                 angle_averaging=self.angle_averaging,
                 max_k_harmonics=self.max_k_harmonics,
                 max_aa_harmonics=self.max_aa_harmonics)

    def parse_kt_data(self, nkt_flag=False, vkt_flag=False):
        """Read in the precomputed time dependent Fourier space data. Recalculate if not.

        Parameters
        ----------
        nkt_flag : bool
            Flag for reading microscopic density Fourier components ``n(\mathbf k, t)``. Default = False.

        vkt_flag : bool
            Flag for reading microscopic velocity Fourier components,``v(\mathbf k, t)``. Default = False.

        """
        if nkt_flag:
            try:
                nkt_data = np.load(self.nkt_file + '_slice_' + str(self.no_slices - 1) + '.npz')
                # Check for the correct number of k values
                if self.angle_averaging == nkt_data["angle_averaging"]:
                    # Check for the correct max harmonics
                    comp = self.max_k_harmonics == nkt_data["max_harmonics"]
                    if not comp.all():
                        self.calc_kt_data(nkt_flag=True)
                else:
                    self.calc_kt_data(nkt_flag=True)

            except FileNotFoundError:
                self.calc_kt_data(nkt_flag=True)

        if vkt_flag:
            try:
                vkt_data = np.load(self.vkt_file + '_slice_' + str(self.no_slices - 1) + '.npz')
                # Check for the correct number of k values
                if self.angle_averaging == vkt_data["angle_averaging"]:
                    # Check for the correct max harmonics
                    comp = self.max_k_harmonics == vkt_data["max_harmonics"]
                    if not comp.all():
                        self.calc_kt_data(vkt_flag=True)
                else:
                    self.calc_kt_data(vkt_flag=True)

            except FileNotFoundError:
                self.calc_kt_data(vkt_flag=True)

    def calc_kt_data(self, nkt_flag=False, vkt_flag=False):
        """Calculate Time dependent Fourier space quantities.

        Parameters
        ----------
        nkt_flag : bool
            Flag for calculating microscopic density Fourier components ``n(\mathbf k, t)``. Default = False.

        vkt_flag : bool
            Flag for calculating microscopic velocity Fourier components,``v(\mathbf k, t)``. Default = False.

        """
        start_slice = 0
        end_slice = self.slice_steps * self.dump_step
        if nkt_flag:
            for isl in range(self.no_slices):
                print("\nCalculating n(k,t) for slice {}/{}.".format(isl, self.no_slices))
                nkt = calc_nkt(self.dump_dir,
                               (start_slice, end_slice, self.slice_steps),
                               self.dump_step,
                               self.species_num,
                               self.k_list,
                               self.verbose)
                start_slice += self.slice_steps * self.dump_step
                end_slice += self.slice_steps * self.dump_step

                np.savez(self.nkt_file + '_slice_' + str(isl) + '.npz',
                         nkt=nkt,
                         max_harmonics=self.max_k_harmonics,
                         angle_averaging=self.angle_averaging)
        if vkt_flag:
            for isl in range(self.no_slices):
                print("\nCalculating longitudinal and transverse "
                      "velocity fluctuations v(k,t) for slice {}/{}.".format(isl, self.no_slices))
                vkt, vkt_i, vkt_j, vkt_k = calc_vkt(self.dump_dir,
                                                    (start_slice, end_slice, self.slice_steps),
                                                    self.dump_step,
                                                    self.species_num,
                                                    self.k_list,
                                                    self.verbose)
                start_slice += self.slice_steps * self.dump_step
                end_slice += self.slice_steps * self.dump_step

                np.savez(self.vkt_file + '_slice_' + str(isl) + '.npz',
                         longitudinal=vkt,
                         transverse_i=vkt_i,
                         transverse_j=vkt_j,
                         transverse_k=vkt_k,
                         max_harmonics=self.max_k_harmonics,
                         angle_averaging=self.angle_averaging)

    def plot(self, scaling=None, figname=None, show=False, acf=False, longitudinal=True, **kwargs):
        """
        Plot the observable by calling the pandas.DataFrame.plot() function and save the figure.

        Parameters
        ----------
        longitudinal : bool
            Flag for longitudinal plot in case of CurrenCurrelationFunction

        acf : bool
            Flag for renormalizing the autocorrelation functions. Default= False

        figname : str
            Name with which to save the file. It automatically saves it in the correct directory.

        scaling: float, tuple
            Factor by which to rescale the x and y quantities.

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
        if self.__class__.__name__ != 'CurrentCorrelationFunction':
            plot_dataframe = self.dataframe.copy()
        elif longitudinal:
            plot_dataframe = self.dataframe_longitudinal.copy()
        else:
            plot_dataframe = self.dataframe_transverse.copy()

        if scaling:
            if isinstance(scaling, tuple):
                plot_dataframe.iloc[:, 0] /= scaling[0]
                plot_dataframe[kwargs['y']] /= scaling[1]
            else:
                plot_dataframe.iloc[:, 0] /= scaling

        # Autocorrelation function renormalization
        if acf:
            for i, col in enumerate(plot_dataframe.columns[1:], 1):
                plot_dataframe[col] /= plot_dataframe[col].iloc[0]
            kwargs['logx'] = True

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
            fig.savefig(os.path.join(self.saving_dir, figname + '_' + self.job_id + '.png'))
        else:
            fig.savefig(os.path.join(self.saving_dir, 'Plot_' + self.__class__.__name__ + '_' + self.job_id + '.png'))

        if show:
            fig.show()

        return axes_handle

    def time_stamp(self, message: str, timing: tuple):
        """Print to screen the elapsed time of the calculation."""

        t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = timing

        if t_hrs == 0 and t_min == 0 and t_sec <= 2:
            print_message = '\n{} Time: {} sec {} msec {} usec {} nsec'.format(message,
                                                                               int(t_sec),
                                                                               int(t_msec),
                                                                               int(t_usec),
                                                                               int(t_nsec))

        else:
            print_message = '\n{} Time: {} hrs {} min {} sec'.format(message, int(t_hrs), int(t_min), int(t_sec))

        # logging.info(print_message)
        print(print_message)


class CurrentCorrelationFunction(Observable):
    """
    Current Correlation Functions: :math:`L(k,\\omega)` and :math:`T(k,\\omega)`.

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

    def setup(self, params, phase: str = None, **kwargs):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to compute. Default = 'production'.

        params : sarkas.base.Parameters
            Simulation's parameters.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """
        self.__name__ = 'ccf'
        self.__long_name__ = 'Current Correlation Function'

        if phase:
            self.phase = phase.lower()

        super().setup_init(params, self.phase)

        saving_dir = os.path.join(self.postprocessing_dir, 'CurrentCorrelationFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        # These calculation are needed for the io.postprocess_info().
        # This is a hack and we need to find a faster way to do it
        self.slice_steps = int(
            (self.production_steps + 1) / (self.dump_step * self.no_slices))
        self.no_dumps = int(self.slice_steps / self.dump_step)
        dt_r = self.dt * self.dump_step

        self.frequencies = 2.0 * np.pi * np.fft.fftfreq(self.slice_steps, self.dt * self.dump_step)

        self.w_min = 2.0 * np.pi / (self.no_dumps * dt_r)
        self.w_max = np.pi / dt_r  # Half because np.fft calculates negative and positive frequencies

        self.parse_k_data()

        self.filename_csv_longitudinal = os.path.join(self.saving_dir,
                                                      "LongitudinalCurrentCorrelationFunction_" + self.job_id + '.csv')
        self.filename_csv_transverse = os.path.join(self.saving_dir,
                                                    "TransverseCurrentCorrelationFunction_" + self.job_id + '.csv')

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

    def compute(self, **kwargs):
        """
        Calculate the microscopic current fluctuations correlation functions.

        Parameters
        ----------
        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())
        # Parse vkt otherwise calculate them
        self.parse_k_data()  # repeat from setup in case parameters have been updated
        self.parse_kt_data(nkt_flag=False, vkt_flag=True)
        # Initialize dataframes and add frequencies to it.
        # This re-initialization of the dataframe is needed to avoid len mismatch conflicts when re-calculating
        self.dataframe = pd.DataFrame()
        self.frequencies = 2.0 * np.pi * np.fft.fftfreq(self.slice_steps, self.dt * self.dump_step)
        self.dataframe_longitudinal["Frequencies"] = np.fft.fftshift(self.frequencies)
        self.dataframe_transverse["Frequencies"] = np.fft.fftshift(self.frequencies)

        temp_dataframe_longitudinal = pd.DataFrame()
        temp_dataframe_longitudinal["Frequencies"] = np.fft.fftshift(self.frequencies)

        temp_dataframe_transverse = pd.DataFrame()
        temp_dataframe_transverse["Frequencies"] = np.fft.fftshift(self.frequencies)

        Lkw_tot = np.zeros((self.no_obs, len(self.k_counts), self.slice_steps))
        Tkw_tot = np.zeros((self.no_obs, len(self.k_counts), self.slice_steps))

        for isl in range(self.no_slices):
            data = np.load(self.vkt_file + '_slice_' + str(isl) + '.npz')
            vkt = data["longitudinal"]
            vkt_i = data["transverse_i"]
            vkt_j = data["transverse_j"]
            vkt_k = data["transverse_k"]

            # Calculate Lkw and Tkw
            Lkw = calc_Skw(vkt, self.k_list, self.k_counts, self.species_num, self.slice_steps, self.dt, self.dump_step)
            Tkw_i = calc_Skw(vkt_i, self.k_list, self.k_counts, self.species_num, self.slice_steps, self.dt,
                             self.dump_step)
            Tkw_j = calc_Skw(vkt_j, self.k_list, self.k_counts, self.species_num, self.slice_steps, self.dt,
                             self.dump_step)
            Tkw_k = calc_Skw(vkt_k, self.k_list, self.k_counts, self.species_num, self.slice_steps, self.dt,
                             self.dump_step)

            Tkw = (Tkw_i + Tkw_j + Tkw_k) / 3.0

            Lkw_tot += Lkw / self.no_slices
            Tkw_tot += Tkw / self.no_slices

            sp_indx = 0
            for i, sp1 in enumerate(self.species_names):
                for j, sp2 in enumerate(self.species_names[i:]):
                    for ik in range(len(self.k_counts)):
                        if ik == 0:
                            column = "{}-{} CCF ka_min".format(sp1, sp2)
                        else:
                            column = "{}-{} CCF {} ka_min".format(sp1, sp2, ik + 1)

                        temp_dataframe_longitudinal[column] = np.fft.fftshift(Lkw[sp_indx, ik, :])
                        temp_dataframe_transverse[column] = np.fft.fftshift(Tkw[sp_indx, ik, :])
                    sp_indx += 1

            temp_dataframe_longitudinal.to_csv(self.filename_csv_longitudinal[:-4] + '_slice_' + str(isl) + '.csv',
                                               index=False, encoding='utf-8')
            temp_dataframe_transverse.to_csv(self.filename_csv_transverse[:-4] + '_slice_' + str(isl) + '.csv',
                                             index=False, encoding='utf-8')

        # Repeat the saving procedure for the total Lkw and Tkw
        sp_indx = 0
        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:]):
                for ik in range(len(self.k_counts)):
                    if ik == 0:
                        column = "{}-{} CCF ka_min".format(sp1, sp2)
                    else:
                        column = "{}-{} CCF {} ka_min".format(sp1, sp2, ik + 1)

                    self.dataframe_longitudinal[column] = np.fft.fftshift(Lkw[sp_indx, ik, :])
                    self.dataframe_transverse[column] = np.fft.fftshift(Tkw[sp_indx, ik, :])
                sp_indx += 1

        self.dataframe_longitudinal.to_csv(self.filename_csv_longitudinal, index=False, encoding='utf-8')
        self.dataframe_transverse.to_csv(self.filename_csv_transverse, index=False, encoding='utf-8')

    def pretty_print(self):
        """Print current correlation function calculation parameters for help in choice of simulation parameters."""
        print('\n\n{:=^70} \n'.format(' ' + self.__long_name__ + ' '))
        print('k wavevector information saved in: ', self.k_file)
        print('v(k,t) data saved in: ', self.vkt_file)
        print('Data saved in: \n\t{} \n\t{}'.format(self.filename_csv_longitudinal, self.filename_csv_transverse))
        print('Data accessible at: self.k_list, self.k_counts, self.ka_values, self.frequencies,'
              ' \n\t self.dataframe_longitudinal, self.dataframe_transverse')
        print('Frequency Space Parameters:')
        print('\tNo. of slices = {}'.format(self.no_slices))
        print('\tNo. steps per slice = {}'.format(self.slice_steps))
        print('\tNo. dumps per slice = {}'.format(self.no_dumps))
        print('\tFrequency step dw = 2 pi (no_slices * prod_dump_step)/(production_steps * dt)')
        print('\tdw = {:1.4f} w_p = {:1.4e} [Hz]'.format(
            self.w_min / self.total_plasma_frequency, self.w_min))
        print('\tMaximum Frequency w_max = 2 pi /(prod_dump_step * dt)')
        print('\tw_max = {:1.4f} w_p = {:1.4e} [Hz]'.format(
            self.w_max / self.total_plasma_frequency, self.w_max))

        print('\n\nWavevector parameters:')
        print('Smallest wavevector k_min = 2 pi / L = 3.9 / N^(1/3)')
        print('k_min = {:.4f} / a_ws = {:.4e} '.format(self.ka_values[0], self.ka_values[0] / self.a_ws), end='')
        print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print('\nAngle averaging choice: {}'.format(self.angle_averaging))
        if self.angle_averaging == 'full':
            print('\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_aa_harmonics))
            print('\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)')
            print('\tk_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_aa_ka_value,
                                                              self.max_aa_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == 'custom':
            print('\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_aa_harmonics))
            print('\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)')
            print('\tAA k_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_aa_ka_value,
                                                                 self.max_aa_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

            print('\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_k_harmonics))
            print('\tLargest wavector k_max = k_min * n_x')
            print('\tk_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_ka_value,
                                                              self.max_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == 'principal_axis':
            print('\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_k_harmonics))
            print('\tLargest wavector k_max = k_min * n_x')
            print('\tk_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_ka_value,
                                                              self.max_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print('\nTotal number of k values to calculate = {}'.format(len(self.k_list)))
        print('No. of unique ka values to calculate = {}'.format(len(self.ka_values)))


class DynamicStructureFactor(Observable):
    """Dynamic Structure factor.    """

    def setup(self, params, phase: str = None, **kwargs):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to compute. Default = 'production'.

        params : sarkas.base.Parameters
            Simulation's parameters.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        self.__name__ = 'dsf'
        self.__long_name__ = 'Dynamic Structure Factor'
        if phase:
            self.phase = phase.lower()

        super().setup_init(params, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'DynamicStructureFactor')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        # Create the phase directory
        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        # These calculation are needed for the io.postprocess_info().
        # This is a hack and we need to find a faster way to do it
        self.slice_steps = int((self.production_steps + 1) / (self.dump_step * self.no_slices))
        self.no_dumps = int(self.slice_steps / self.dump_step)
        dt_r = self.dt * self.dump_step

        self.frequencies = 2.0 * np.pi * np.fft.fftfreq(self.slice_steps, self.dt * self.dump_step)

        self.w_min = 2.0 * np.pi / (self.no_dumps * self.dt * self.dump_step)
        self.w_max = np.pi / dt_r  # Half because np.fft calculates negative and positive frequencies

        self.parse_k_data()

        self.filename_csv = os.path.join(self.saving_dir, "DynamicStructureFactor_" + self.job_id + '.csv')

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

    def compute(self, **kwargs):
        """
        Compute :math:`S_{ij} (k,\\omega)` and the array of :math:`\\omega` values.
        Shape = (``no_ws``, ``no_Sij``)

        Parameters
        ----------
        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

        # Parse nkt otherwise calculate it
        self.parse_k_data()  # repeat from setup in case parameters have been updated
        self.parse_kt_data(nkt_flag=True)
        # This re-initialization of the dataframe is needed to avoid len mismatch conflicts when re-calculating
        self.dataframe = pd.DataFrame()

        self.frequencies = 2.0 * np.pi * np.fft.fftfreq(self.slice_steps, self.dt * self.dump_step)
        self.dataframe["Frequencies"] = np.fft.fftshift(self.frequencies)

        temp_dataframe = pd.DataFrame()
        temp_dataframe["Frequencies"] = np.fft.fftshift(self.frequencies)

        Skw_tot = np.zeros((self.no_obs, len(self.k_counts), self.slice_steps))

        for isl in range(0, self.no_slices):
            nkt_data = np.load(self.nkt_file + '_slice_' + str(isl) + '.npz')
            nkt = nkt_data["nkt"]
            # Calculate Skw
            Skw = calc_Skw(nkt, self.k_list, self.k_counts, self.species_num, self.slice_steps, self.dt,
                           self.dump_step)

            Skw_tot += Skw / self.no_slices

            # Save Skw
            sp_indx = 0
            for i, sp1 in enumerate(self.species_names):
                for j, sp2 in enumerate(self.species_names[i:]):
                    for ik in range(len(self.k_counts)):
                        if ik == 0:
                            column = "{}-{} DSF ka_min".format(sp1, sp2)
                        else:
                            column = "{}-{} DSF {} ka_min".format(sp1, sp2, ik + 1)
                        temp_dataframe[column] = np.fft.fftshift(Skw[sp_indx, ik, :])
                    sp_indx += 1

            temp_dataframe.to_csv(self.filename_csv[:-4] + '_slice_' + str(isl) + '.csv', index=False, encoding='utf-8')

        # Repeat the saving procedure for the total Skw
        sp_indx = 0
        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:]):
                for ik in range(len(self.k_counts)):
                    if ik == 0:
                        column = "{}-{} DSF ka_min".format(sp1, sp2)
                    else:
                        column = "{}-{} DSF {} ka_min".format(sp1, sp2, ik + 1)
                    self.dataframe[column] = np.fft.fftshift(Skw_tot[sp_indx, ik, :])
                sp_indx += 1

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def pretty_print(self):
        """Print dynamic structure factor calculation parameters for help in choice of simulation parameters."""

        print('\n\n{:=^70} \n'.format(' ' + self.__long_name__ + ' '))
        print('k wavevector information saved in: ', self.k_file)
        print('n(k,t) data saved in: ', self.nkt_file)
        print('Data saved in: ', self.filename_csv)
        print('Data accessible at: self.k_list, self.k_counts, self.ka_values, self.frequencies, self.dataframe')

        print('Frequency Space Parameters:')
        print('\tNo. of slices = {}'.format(self.no_slices))
        print('\tNo. steps per slice = {}'.format(self.slice_steps))
        print('\tNo. dumps per slice = {}'.format(self.no_dumps))
        print('\tFrequency step dw = 2 pi (no_slices * prod_dump_step)/(production_steps * dt)')
        print('\tdw = {:1.4f} w_p = {:1.4e} [Hz]'.format(
            self.w_min / self.total_plasma_frequency, self.w_min))
        print('\tMaximum Frequency w_max = 2 pi /(prod_dump_step * dt)')
        print('\tw_max = {:1.4f} w_p = {:1.4e} [Hz]'.format(
            self.w_max / self.total_plasma_frequency, self.w_max))

        print('\n\nWavevector parameters:')
        print('Smallest wavevector k_min = 2 pi / L = 3.9 / N^(1/3)')
        print('k_min = {:.4f} / a_ws = {:.4e} '.format(self.ka_values[0], self.ka_values[0] / self.a_ws), end='')
        print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print('\nAngle averaging choice: {}'.format(self.angle_averaging))
        if self.angle_averaging == 'full':
            print('\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_aa_harmonics))
            print('\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)')
            print('\tk_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_aa_ka_value,
                                                              self.max_aa_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == 'custom':
            print('\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_aa_harmonics))
            print('\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)')
            print('\tAA k_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_aa_ka_value,
                                                                 self.max_aa_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

            print('\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_k_harmonics))
            print('\tLargest wavector k_max = k_min * n_x')
            print('\tk_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_ka_value,
                                                              self.max_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == 'principal_axis':
            print('\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_k_harmonics))
            print('\tLargest wavector k_max = k_min * n_x')
            print('\tk_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_ka_value,
                                                              self.max_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print('\nTotal number of k values to calculate = {}'.format(len(self.k_list)))
        print('No. of unique ka values to calculate = {}'.format(len(self.ka_values)))


class ElectricCurrent(Observable):
    """Electric Current Auto-correlation function."""

    def setup(self, params, phase: str = None, **kwargs):
        """
        Initialize the attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to compute. Default = 'production'.

        params : sarkas.base.Parameters
            Simulation's parameters.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """
        self.__name__ = 'ec'

        if phase:
            self.phase = phase.lower()

        super().setup_init(params, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'ElectricCurrent')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "ElectricCurrent_" + self.job_id + '.csv')

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

    def compute(self, **kwargs):
        """
        Compute the electric current and the corresponding auto-correlation functions.

        Parameters
        ----------
        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

        # Parse the particles from the dump files
        vel = np.zeros((self.no_dumps, 3, self.total_num_ptcls))
        #
        print("Parsing particles' velocities.")
        time = np.zeros(self.no_dumps)
        for it in tqdm(range(self.no_dumps), disable=(not self.verbose)):
            dump = int(it * self.dump_step)
            time[it] = dump * self.dt
            datap = load_from_restart(self.dump_dir, dump)
            vel[it, 0, :] = datap["vel"][:, 0]
            vel[it, 1, :] = datap["vel"][:, 1]
            vel[it, 2, :] = datap["vel"][:, 2]
        #
        print("Calculating Electric current quantities.")
        species_current, total_current = calc_elec_current(vel, self.species_charges, self.species_num)

        self.dataframe["Time"] = time

        self.dataframe["Total Current X"] = total_current[0, :]
        self.dataframe["Total Current Y"] = total_current[1, :]
        self.dataframe["Total Current Z"] = total_current[2, :]

        # Calculate the ACF in each direction
        cur_acf_xx = correlationfunction(total_current[0, :], total_current[0, :])
        cur_acf_yy = correlationfunction(total_current[1, :], total_current[1, :])
        cur_acf_zz = correlationfunction(total_current[2, :], total_current[2, :])
        # Total Current ACF
        tot_cur_acf = cur_acf_xx + cur_acf_yy + cur_acf_zz

        # Save
        self.dataframe["X Current ACF"] = cur_acf_xx
        self.dataframe["Y Current ACF"] = cur_acf_yy
        self.dataframe["Z Current ACF"] = cur_acf_zz
        self.dataframe["Total Current ACF"] = tot_cur_acf
        for i, sp in enumerate(self.species_names):
            acf_xx = correlationfunction(species_current[i, 0, :], species_current[i, 0, :])
            acf_yy = correlationfunction(species_current[i, 1, :], species_current[i, 1, :])
            acf_zz = correlationfunction(species_current[i, 2, :], species_current[i, 2, :])
            tot_acf = acf_xx + acf_yy + acf_zz

            self.dataframe["{} Total Current".format(sp)] = np.sqrt(
                species_current[i, 0, :] ** 2 + species_current[i, 1, :] ** 2 + species_current[i, 2, :] ** 2)
            self.dataframe["{} X Current".format(sp)] = species_current[i, 0, :]
            self.dataframe["{} Y Current".format(sp)] = species_current[i, 1, :]
            self.dataframe["{} Z Current".format(sp)] = species_current[i, 2, :]

            self.dataframe["{} Total Current ACF".format(sp)] = tot_acf
            self.dataframe["{} X Current ACF".format(sp)] = acf_xx
            self.dataframe["{} Y Current ACF".format(sp)] = acf_yy
            self.dataframe["{} Z Current ACF".format(sp)] = acf_zz

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')


# class HermiteCoefficients(Observable):
#     """
#     Hermite coefficients of the Hermite expansion.
#
#     Attributes
#     ----------
#     hermite_order: int
#         Order of the Hermite expansion.
#
#     no_bins: int
#         Number of bins used to calculate the velocity distribution.
#
#     plots_dir: str
#         Directory in which to store Hermite coefficients plots.
#
#     species_plots_dirs : list, str
#         Directory for each species where to save Hermite coefficients plots.
#
#     """
#
#     def setup(self, params, phase: str = None, **kwargs):
#         """
#         Assign attributes from simulation's parameters.
#
#         Parameters
#         ----------
#         phase : str
#             Phase to compute. Default = 'production'.
#
#         params : sarkas.base.Parameters
#             Simulation's parameters.
#
#         **kwargs :
#             These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
#             attributes and/or add new ones.
#
#         """
#         self.__name__ = 'hc'
#
#         self.phase = phase.lower() if phase.lower() != 'production' else 'production'
#
#         super().setup_init(params, self.phase)
#
#         # Create the directory where to store the computed data
#         saving_dir = os.path.join(self.postprocessing_dir, 'HermiteExpansion')
#         if not os.path.exists(saving_dir):
#             os.mkdir(saving_dir)
#
#         self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
#         if not os.path.exists(self.saving_dir):
#             os.mkdir(self.saving_dir)
#         self.filename_csv = os.path.join(self.saving_dir, "HermiteCoefficients_" + self.job_id + '.csv')
#
#         self.plots_dir = os.path.join(self.saving_dir, 'Hermite_Plots')
#         if not os.path.exists(self.plots_dir):
#             os.mkdir(self.plots_dir)
#
#         # Check that the existence of important attributes
#         if not hasattr(self, 'no_bins'):
#             self.no_bins = int(0.05 * params.total_num_ptcls)
#
#         if not hasattr(self, 'hermite_order'):
#             self.hermite_order = 7
#
#         self.species_plots_dirs = None
#
#         # Update the attribute with the passed arguments
#         self.__dict__.update(kwargs.copy())
#
#     def compute(self, **kwargs):
#         """
#         Calculate Hermite coefficients and save the pandas dataframe.
#
#         Parameters
#         ----------
#         **kwargs :
#             These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
#             attributes and/or add new ones.
#
#         """
#
#         self.__dict__.update(kwargs.copy())
#
#         self.dataframe = pd.DataFrame()
#         time = np.zeros(self.no_dumps)
#
#         # 2nd Dimension of the raw velocity array
#         dim = 1 if self.dimensional_average else self.dimensions
#         # range(inv_dim) for the loop over dimension
#         inv_dim = self.dimensions if self.dimensional_average else 1
#         # Array containing the start index of each species. The last value is equivalent to vel_raw.shape[-1]
#         species_index_start = np.array([0, *self.species_num], dtype=int) * inv_dim * self.runs
#         # Velocity array for storing simulation data
#         vel_raw = np.zeros((self.no_dumps, dim, self.runs * inv_dim * self.total_num_ptcls))
#
#         print("Collecting data from snapshots ...")
#
#         if self.dimensional_average:
#             # Loop over the runs
#             for r, dump_dir_r in enumerate(tqdm(self.adjusted_dump_dir, disable=(not self.verbose), desc='Runs Loop')):
#                 # Loop over the timesteps
#                 for it in tqdm(range(self.no_dumps), disable=(not self.verbose), desc='Timestep Loop'):
#                     # Read data from file
#                     dump = int(it * self.dump_step)
#                     datap = load_from_restart(dump_dir_r, dump)
#                     # Loop over the particles' species
#                     for sp_indx, (sp_name, sp_num) in enumerate(zip(self.species_names, self.species_num)):
#                         # Calculate the correct start and end index for storage
#                         start_indx = species_index_start[sp_indx] + inv_dim * sp_num * r
#                         end_indx = species_index_start[sp_indx] + inv_dim * sp_num * (r + 1)
#                         # Use a mask to grab only the selected species and flatten along the first axis
#                         # data = ( v1_x, v1_y, v1_z,
#                         #          v2_x, v2_y, v2_z,
#                         #          v3_x, v3_y, v3_z,
#                         #          ...)
#                         # The flatten array would like this
#                         # flattened = ( v1_x, v2_x, v3_x, ..., v1_y, v2_y, v3_y, ..., v1_z, v2_z, v3_z, ...)
#                         vel_raw[it, 0, start_indx: end_indx] = datap["vel"][datap["names"] == sp_name].flatten('F')
#
#                     time[it] = datap["time"]
#         else:  # Dimensional Average = False
#             # Loop over the runs
#             for r, dump_dir_r in enumerate(tqdm(self.adjusted_dump_dir, disable=(not self.verbose), desc='Runs Loop')):
#                 # Loop over the timesteps
#                 for it in tqdm(range(self.no_dumps), disable=(not self.verbose), desc='Timestep Loop'):
#                     # Read data from file
#                     dump = int(it * self.dump_step)
#                     datap = load_from_restart(dump_dir_r, dump)
#                     # Loop over the particles' species
#                     for sp_indx, (sp_name, sp_num) in enumerate(zip(self.species_names, self.species_num)):
#                         # Calculate the correct start and end index for storage
#                         start_indx = species_index_start[sp_indx] + inv_dim * sp_num * r
#                         end_indx = species_index_start[sp_indx] + inv_dim * sp_num * (r + 1)
#                         # Use a mask to grab only the selected species and transpose the array to put dimensions first
#                         vel_raw[it, :, start_indx: end_indx] = datap["vel"][datap["names"] == sp_name].transpose()
#
#                 time[it] = datap["time"]
#
#         self.dataframe["Time"] = time
#
#         xcoeff = np.zeros((self.num_species, self.no_dumps, self.hermite_order + 1))
#         ycoeff = np.zeros((self.num_species, self.no_dumps, self.hermite_order + 1))
#         zcoeff = np.zeros((self.num_species, self.no_dumps, self.hermite_order + 1))
#
#         time = np.zeros(self.no_dumps)
#         print("Computing Hermite Coefficients ...")
#         for it in range(self.no_dumps):
#             time[it] = it * self.dt * self.dump_step
#             dump = int(it * self.dump_step)
#             datap = load_from_restart(self.dump_dir, dump)
#             vel[0, :] = datap["vel"][:, 0]
#             vel[1, :] = datap["vel"][:, 1]
#             vel[2, :] = datap["vel"][:, 2]
#
#             sp_start = 0
#             sp_end = 0
#             for i, sp in enumerate(self.species_num):
#                 sp_end += sp
#                 x_hist, xbins = np.histogram(vel[0, sp_start:sp_end] * vscale, bins=self.no_bins, density=True)
#                 y_hist, ybins = np.histogram(vel[1, sp_start:sp_end] * vscale, bins=self.no_bins, density=True)
#                 z_hist, zbins = np.histogram(vel[2, sp_start:sp_end] * vscale, bins=self.no_bins, density=True)
#
#                 # Center the bins
#                 vx = 0.5 * (xbins[:-1] + xbins[1:])
#                 vy = 0.5 * (ybins[:-1] + ybins[1:])
#                 vz = 0.5 * (zbins[:-1] + zbins[1:])
#
#                 xcoeff[i, it, :] = calculate_herm_coeff(vx, x_hist, self.hermite_order)
#                 ycoeff[i, it, :] = calculate_herm_coeff(vy, y_hist, self.hermite_order)
#                 zcoeff[i, it, :] = calculate_herm_coeff(vz, z_hist, self.hermite_order)
#
#                 sp_start = sp_end
#
#         data = {"Time": time}
#         self.dataframe = pd.DataFrame(data)
#         for i, sp in enumerate(self.species_names):
#             for hi in range(self.hermite_order + 1):
#                 self.dataframe["{} Hermite x Coeff a{}".format(sp, hi)] = xcoeff[i, :, hi]
#                 self.dataframe["{} Hermite y Coeff a{}".format(sp, hi)] = ycoeff[i, :, hi]
#                 self.dataframe["{} Hermite z Coeff a{}".format(sp, hi)] = zcoeff[i, :, hi]
#
#         self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')
#
#     def plot(self, show=False):
#         """
#         Plot the Hermite coefficients and save the figure
#         """
#         try:
#             self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
#         except FileNotFoundError:
#             self.compute()
#
#         if not os.path.exists(self.plots_dir):
#             os.mkdir(self.plots_dir)
#
#         # Create a plots directory for each species for the sake of neatness
#         if self.num_species > 1:
#             self.species_plots_dirs = []
#             for i, name in enumerate(self.species_names):
#                 new_dir = os.path.join(self.plots_dir, "{}".format(name))
#                 self.species_plots_dirs.append(new_dir)
#                 if not os.path.exists(new_dir):
#                     os.mkdir(os.path.join(self.plots_dir, "{}".format(name)))
#         else:
#             self.species_plots_dirs = [self.plots_dir]
#
#         for sp, name in enumerate(self.species_names):
#             print("Species: {}".format(name))
#             fig, ax = plt.subplots(1, 2, sharex=True, constrained_layout=True, figsize=(16, 9))
#             for indx in range(self.hermite_order):
#                 xcolumn = "{} Hermite x Coeff a{}".format(name, indx)
#                 ycolumn = "{} Hermite y Coeff a{}".format(name, indx)
#                 zcolumn = "{} Hermite z Coeff a{}".format(name, indx)
#                 xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(self.dataframe["Time"], 1.0,
#                                                                        'Time', 'none', self.units)
#                 ia = int(indx % 2)
#                 ax[ia].plot(self.dataframe["Time"] * xmul, self.dataframe[xcolumn] + ia * (indx - 1),
#                             ls='-', label=r"$a_{" + str(indx) + " , x}$")
#                 ax[ia].plot(self.dataframe["Time"] * xmul, self.dataframe[ycolumn] + ia * (indx - 1),
#                             ls='--', label=r"$a_{" + str(indx) + " , y}$")
#                 ax[ia].plot(self.dataframe["Time"] * xmul, self.dataframe[zcolumn] + ia * (indx - 1),
#                             ls='-.', label=r"$a_{" + str(indx) + " , z}$")
#
#             ax[0].set_title(r'Even Coefficients')
#             ax[1].set_title(r'Odd Coefficients')
#
#             ax[0].set_xlabel(r'$t$' + xlbl)
#             ax[1].set_xlabel(r'$t$' + xlbl)
#
#             sigma = np.sqrt(self.kB * self.species_temperatures[sp] / self.species_masses[sp])
#             sigma /= (self.a_ws * self.total_plasma_frequency)  # Rescale
#
#             for i in range(0, self.hermite_order, 2):
#                 coeff = np.zeros(i + 1)
#                 coeff[-1] = 1.0
#                 print("Equilibrium a{} = {:1.2f} ".format(i, np.polynomial.hermite_e.hermeval(sigma, coeff)))
#
#             # t_end = self.dataframe["Time"].iloc[-1] * xmul/2
#             # ax[0].text(t_end, 1.1, r"$a_{0,\rm{eq}} = 1 $", transform=ax[0].transData)
#             # # ax[0].axhline(1, ls=':', c='k', label=r"$a_{0,\rm{eq}}$")
#             #
#             # ax[0].text(t_end, a2_eq * 0.97, r"$a_{2,\rm{eq}} = " + "{:1.2f}".format(a2_eq) +"$",
#             #            transform=ax[0].transData)
#             #
#             # if self.hermite_order > 3:
#             #     ax[0].text(t_end, a4_eq * 1.1, r"$a_{4,\rm{eq}} = " + "{:1.2f}".format(a4_eq) + "$",
#             #                transform=ax[0].transData)
#             #
#             # if self.hermite_order > 5:
#             #     ax[0].text(t_end, a6_eq * .98, r"$a_{6,\rm{eq}} = " + "{:1.2f}".format(a6_eq) + "$",
#             #                transform=ax[0].transData)
#
#             ax[0].legend(loc='best', ncol=int(self.hermite_order / 2 + self.hermite_order % 2))
#             ax[1].legend(loc='best', ncol=int(self.hermite_order / 2))
#             yt = np.arange(0, self.hermite_order + self.hermite_order % 2, 2)
#             ax[1].set_yticks(yt)
#             ax[1].set_yticklabels(np.zeros(len(yt)))
#             fig.suptitle("Hermite Coefficients of {}".format(name) + '  Phase: ' + self.phase.capitalize())
#             plot_name = os.path.join(self.species_plots_dirs[sp], '{}_HermCoeffPlot_'.format(name)
#                                      + self.job_id + '.png')
#             fig.savefig(plot_name)
#             if show:
#                 fig.show()
#

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

    def setup(self, params, phase: str = None, **kwargs):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to compute. Default = 'production'.

        params : sarkas.base.Parameters
            Simulation's parameters.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        self.__name__ = 'rdf'
        self.__long_name__ = 'Radial Distribution Function'
        if phase:
            self.phase = phase.lower()

        super().setup_init(params, self.phase)

        saving_dir = os.path.join(self.postprocessing_dir, 'RadialDistributionFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir,
                                         "RadialDistributionFunction_" + self.job_id + ".csv")

        # These definitions are needed for the print out.
        self.rc = self.cutoff_radius
        self.no_bins = self.rdf_nbins
        self.dr_rdf = self.rc / self.no_bins

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

    def compute(self, rdf_hist=None, **kwargs):
        """
        Parameters
        ----------
        rdf_hist : numpy.ndarray
            Histogram of the radial distribution function.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

        # initialize temporary arrays
        r_values = np.zeros(self.no_bins)
        bin_vol = np.zeros(self.no_bins)
        pair_density = np.zeros((self.num_species, self.num_species))
        gr = np.zeros((self.no_bins, self.no_obs))

        if not isinstance(rdf_hist, np.ndarray):
            # Find the last dump by looking for the largest number in the checkpoints filenames
            dumps_list = os.listdir(self.dump_dir)
            dumps_list.sort(key=num_sort)
            name, ext = os.path.splitext(dumps_list[-1])
            _, number = name.split('_')
            data = load_from_restart(self.dump_dir, int(number))
            rdf_hist = data["rdf_hist"]

        # Make sure you are getting the right number of bins and redefine dr_rdf.
        self.no_bins = rdf_hist.shape[0]
        self.dr_rdf = self.rc / self.no_bins

        t0 = self.timer.current()
        # No. of pairs per volume
        for i, sp1 in enumerate(self.species_num):
            pair_density[i, i] = sp1 * (sp1 - 1) / self.box_volume
            if self.num_species > 1:
                for j, sp2 in enumerate(self.species_num[i + 1:], i + 1):
                    pair_density[i, j] = sp1 * sp2 / self.box_volume

        # Calculate the volume of each bin
        sphere_shell_const = 4.0 * np.pi / 3.0
        bin_vol[0] = sphere_shell_const * self.dr_rdf ** 3
        for ir in range(1, self.no_bins):
            r1 = ir * self.dr_rdf
            r2 = (ir + 1) * self.dr_rdf
            bin_vol[ir] = sphere_shell_const * (r2 ** 3 - r1 ** 3)
            r_values[ir] = (ir + 0.5) * self.dr_rdf

        self.ra_values = r_values / self.a_ws

        self.dataframe["Distance"] = r_values
        gr_ij = 0
        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:], i):
                denom_const = (pair_density[i, j] * self.production_steps)
                gr[:, gr_ij] = (rdf_hist[:, i, j] + rdf_hist[:, j, i]) / denom_const / bin_vol[:]

                self.dataframe['{}-{} RDF'.format(sp1, sp2)] = gr[:, gr_ij]

                gr_ij += 1

        tend = self.timer.current()
        self.time_stamp('Radial Distribution Function Calculation', self.timer.time_division(tend - t0))
        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def pretty_print(self):
        """Print radial distribution function calculation parameters for help in choice of simulation parameters."""

        print('\n\n{:=^70} \n'.format(' ' + self.__long_name__ + ' '))
        print('Data saved in: \n\t', self.filename_csv)
        print('Data accessible at: self.ra_values, self.dataframe')
        print('\nNo. bins = {}'.format(self.no_bins))
        print('dr = {:1.4f} a_ws = {:1.4e} '.format(self.dr_rdf / self.a_ws, self.dr_rdf), end='')
        print("[cm]" if self.units == "cgs" else "[m]")
        print('Maximum Distance (i.e. potential.rc)= {:1.4f} a_ws = {:1.4e} '.format(
            self.rc / self.a_ws, self.rc), end='')
        print("[cm]" if self.units == "cgs" else "[m]")


class StaticStructureFactor(Observable):
    """
    Static Structure Factors :math:`S_{ij}(k)`.

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

    def setup(self, params, phase: str = None, **kwargs):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to compute. Default = 'production'.

        params : sarkas.base.Parameters
            Simulation's parameters.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """
        self.__name__ = 'ssf'
        self.__long_name__ = 'Static Structure Function'
        if phase:
            self.phase = phase.lower()

        self.k_observable = True
        super().setup_init(params, self.phase)

        saving_dir = os.path.join(self.postprocessing_dir, 'StaticStructureFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        # These calculation are needed for the io.postprocess_info().
        # This is a hack and we need to find a faster way to do it
        self.slice_steps = int((self.production_steps + 1) / (self.dump_step * self.no_slices))
        self.no_dumps = int(self.slice_steps / self.prod_dump_step)

        self.parse_k_data()

        self.filename_csv = os.path.join(self.saving_dir, "StaticStructureFunction_" + self.job_id + ".csv")

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

    def compute(self, **kwargs):
        """
        Calculate all :math:`S_{ij}(k)`, save them into a Pandas dataframe, and write them to a csv.

        Parameters
        ----------
        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

        # Parse nkt otherwise calculate it
        self.parse_k_data()  # repeat from setup in case parameters have been updated
        self.parse_kt_data(nkt_flag=True)
        # This re-initialization of the dataframe is needed to avoid len mismatch conflicts when re-calculating
        self.dataframe = pd.DataFrame()
        self.dataframe["ka values"] = self.ka_values

        no_dumps_calculated = self.slice_steps * self.no_slices
        Sk_all = np.zeros((self.no_obs, len(self.k_counts), no_dumps_calculated))

        print("Calculating S(k) ...")

        tinit = self.timer.current()
        for isl in tqdm(range(self.no_slices)):
            nkt_data = np.load(self.nkt_file + '_slice_' + str(isl) + '.npz')
            nkt = nkt_data["nkt"]
            init = isl * self.slice_steps
            fin = (isl + 1) * self.slice_steps
            Sk_all[:, :, init:fin] = calc_Sk(nkt, self.k_list, self.k_counts, self.species_num, self.slice_steps)

        Sk = np.mean(Sk_all, axis=-1)
        Sk_err = np.std(Sk_all, axis=-1)
        tend = self.timer.current()

        self.time_stamp("Static Structure Factor Calculation", self.timer.time_division(tend - tinit))

        sp_indx = 0
        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:]):
                column = "{}-{} SSF".format(sp1, sp2)
                err_column = "{}-{} SSF Errorbar".format(sp1, sp2)
                self.dataframe[column] = Sk[sp_indx, :]
                self.dataframe[err_column] = Sk_err[sp_indx, :]

                sp_indx += 1

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def pretty_print(self):
        """Print static structure factor calculation parameters for help in choice of simulation parameters."""

        print('\n\n{:=^70} \n'.format(' ' + self.__long_name__ + ' '))
        print('k wavevector information saved in: \n\t', self.k_file)
        print('n(k,t) data saved in: \n\t', self.nkt_file)
        print('Data saved in: \n\t', self.filename_csv)
        print('Data accessible at: self.k_list, self.k_counts, self.ka_values, self.dataframe')
        print('\nSmallest wavevector k_min = 2 pi / L = 3.9 / N^(1/3)')
        print('k_min = {:.4f} / a_ws = {:.4e} '.format(self.ka_values[0], self.ka_values[0] / self.a_ws), end='')
        print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print('\nAngle averaging choice: {}'.format(self.angle_averaging))
        if self.angle_averaging == 'full':
            print('\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_aa_harmonics))
            print('\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)')
            print('\tk_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_aa_ka_value,
                                                              self.max_aa_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == 'custom':
            print('\tMaximum angle averaged k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_aa_harmonics))
            print('\tLargest angle averaged k_max = k_min * sqrt( n_x^2 + n_y^2 + n_z^2)')
            print('\tAA k_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_aa_ka_value,
                                                                 self.max_aa_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

            print('\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_k_harmonics))
            print('\tLargest wavector k_max = k_min * n_x')
            print('\tk_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_ka_value,
                                                              self.max_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")
        elif self.angle_averaging == 'principal_axis':
            print('\tMaximum k harmonics = n_x, n_y, n_z = {}, {}, {}'.format(*self.max_k_harmonics))
            print('\tLargest wavector k_max = k_min * n_x')
            print('\tk_max = {:.4f} / a_ws = {:1.4e} '.format(self.max_ka_value,
                                                              self.max_ka_value / self.a_ws), end='')
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

        print('\nTotal number of k values to calculate = {}'.format(len(self.k_list)))
        print('No. of unique ka values to calculate = {}'.format(len(self.ka_values)))


class Thermodynamics(Observable):
    """
    Thermodynamic functions.
    """

    def setup(self, params, phase: str = None, **kwargs):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to compute. Default = 'production'.

        params : sarkas.base.Parameters
            Simulation's parameters.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        self.__name__ = 'therm'

        if phase:
            self.phase = phase.lower()

        super().setup_init(params, self.phase)
        self.dataframe = pd.DataFrame()

        if params.load_method == "restart":
            self.restart_sim = True
        else:
            self.restart_sim = False

        if self.phase.lower() == 'production':
            self.saving_dir = self.production_dir
        elif self.phase.lower() == 'equilibration':
            self.saving_dir = self.equilibration_dir
        elif self.phase.lower() == 'magnetization':
            self.saving_dir = self.magnetization_dir

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

    def compute_pressure_quantities(self):
        """
        Calculate Pressure, Pressure Tensor, Pressure Tensor Auto Correlation Function.
        """
        self.parse('production')
        pos = np.zeros((self.dimensions, self.total_num_ptcls))
        vel = np.zeros((self.dimensions, self.total_num_ptcls))
        acc = np.zeros((self.dimensions, self.total_num_ptcls))

        pressure = np.zeros(self.no_dumps)
        pressure_tensor_temp = np.zeros((self.dimensions, self.dimensions, self.no_dumps))

        # Collect particles' positions, velocities and accelerations
        for it in range(int(self.no_dumps)):
            dump = int(it * self.dump_step)

            data = load_from_restart(self.dump_dir, dump)
            pos[0, :] = data["pos"][:, 0]
            pos[1, :] = data["pos"][:, 1]
            pos[2, :] = data["pos"][:, 2]

            vel[0, :] = data["vel"][:, 0]
            vel[1, :] = data["vel"][:, 1]
            vel[2, :] = data["vel"][:, 2]

            acc[0, :] = data["acc"][:, 0]
            acc[1, :] = data["acc"][:, 1]
            acc[2, :] = data["acc"][:, 2]

            pressure[it], pressure_tensor_temp[:, :, it] = calc_pressure_tensor(pos, vel, acc, self.species_masses,
                                                                                self.species_num, self.box_volume)

        self.dataframe["Pressure"] = pressure
        self.dataframe["Pressure ACF"] = correlationfunction(pressure, pressure)

        if self.dimensions == 3:
            dim_lbl = ['x', 'y', 'z']
        elif self.dimensions == 2:
            dim_lbl = ['x', 'y']

        # Calculate the acf of the pressure tensor
        for i, ax1 in enumerate(dim_lbl):
            for j, ax2 in enumerate(dim_lbl):
                self.dataframe["Pressure Tensor {}{}".format(ax1, ax2)] = pressure_tensor_temp[i, j, :]
                pressure_tensor_acf_temp = correlationfunction(pressure_tensor_temp[i, j, :],
                                                               pressure_tensor_temp[i, j, :])
                self.dataframe["Pressure Tensor ACF {}{}".format(ax1, ax2)] = pressure_tensor_acf_temp

        # Save the pressure acf to file
        self.dataframe.to_csv(self.prod_energy_filename, index=False, encoding='utf-8')

    def compute_pressure_from_rdf(self, r, gr, potential, potential_matrix, **kwargs):
        """
        Calculate the Pressure using the radial distribution function

        Parameters
        ----------
        potential: str
            Potential used in the simulation.

        potential_matrix: numpy.ndarray
            Potential parameters.

        r : numpy.ndarray
            Particles' distances.

        gr : numpy.ndarray
            Pair distribution function.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        Returns
        -------
        pressure : float
            Pressure divided by :math:`k_BT`.

        """
        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

        r2 = r * r
        r3 = r2 * r

        if potential == "Coulomb":
            dv_dr = - 1.0 / r2
            # Check for finiteness of first element when r[0] = 0.0
            if not np.isfinite(dv_dr[0]):
                dv_dr[0] = dv_dr[1]
            gr -= 1
        elif potential == "Yukawa":
            pass
        elif potential == "QSP":
            pass
        else:
            raise ValueError('Unknown potential')

        # No. of independent g(r)
        T = np.mean(self.dataframe["Temperature"])
        pressure = self.kB * T - 2.0 / 3.0 * np.pi * self.species_num_dens[0] \
                   * potential_matrix[0, 0, 0] * np.trapz(dv_dr * r3 * gr, x=r)
        pressure *= self.species_num_dens[0]

        return pressure

    def parse(self, phase=None):
        """
        Grab the pandas dataframe from the saved csv file.
        """
        if phase:
            self.phase = phase.lower()

        if self.phase == 'equilibration':
            self.dataframe = pd.read_csv(self.eq_energy_filename, index_col=False)
            self.fldr = self.equilibration_dir
        elif self.phase == 'production':
            self.dataframe = pd.read_csv(self.prod_energy_filename, index_col=False)
            self.fldr = self.production_dir
        elif self.phase == 'magnetization':
            self.dataframe = pd.read_csv(self.mag_energy_filename, index_col=False)
            self.fldr = self.magnetization_dir

    def statistics(self, quantity="Total Energy", max_no_divisions=100, show=False):
        """
        ToDo:
        Parameters
        ----------
        quantity
        max_no_divisions
        show

        Returns
        -------

        """
        self.parse()
        run_avg = self.dataframe[quantity].mean()
        run_std = self.dataframe[quantity].std()

        observable = np.array(self.dataframe[quantity])
        # Loop over the blocks
        tau_blk, sigma2_blk, statistical_efficiency = calc_statistical_efficiency(observable,
                                                                                  run_avg, run_std,
                                                                                  max_no_divisions, self.no_dumps)
        # Plot the statistical efficiency
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.plot(1 / tau_blk[2:], statistical_efficiency[2:], '--o', label=quantity)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_xscale('log')
        ax.set_ylabel(r'$s(\tau_{\rm{blk}})$')
        ax.set_xlabel(r'$1/\tau_{\rm{blk}}$')
        fig.tight_layout()
        fig.savefig(os.path.join(self.postproc_dir, quantity + 'StatisticalEfficiency_' + self.job_id + '.png'))

        if show:
            fig.show()

    def temp_energy_plot(self, process,
                         phase: str = None,
                         show: bool = False,
                         publication: bool = False,
                         figname: str = None):
        """
        Plot Temperature and Energy as a function of time with their cumulative sum and average.

        Parameters
        ----------
        process : sarkas.processes.PostProcess
            Sarkas Process.

        phase: str
            Phase to plot. "equilibration" or "production".

        show: bool
            Flag for displaying the figure.

        publication: bool
            Flag for publication style plotting.

        figname: str
            Name with which to save the plot.

        """

        if phase:
            phase = phase.lower()
            self.phase = phase
            if self.phase == 'equilibration':
                self.no_dumps = self.eq_no_dumps
                self.dump_dir = self.eq_dump_dir
                self.dump_step = self.eq_dump_step
                self.saving_dir = self.equilibration_dir
                self.no_steps = self.equilibration_steps
                self.parse(self.phase)
                self.dataframe = self.dataframe.iloc[1:, :]

            elif self.phase == 'production':
                self.no_dumps = self.prod_no_dumps
                self.dump_dir = self.prod_dump_dir
                self.dump_step = self.prod_dump_step
                self.saving_dir = self.production_dir
                self.no_steps = self.production_steps
                self.parse(self.phase)

            elif self.phase == 'magnetization':
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

            plt.style.use('PUBstyle')
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
        plt.rc('font', size=fsz)  # controls default text sizes
        plt.rc('axes', titlesize=fsz)  # fontsize of the axes title
        plt.rc('axes', labelsize=fsz)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=fsz - 2)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=fsz - 2)  # fontsize of the tick labels

        # Grab the color line list from the plt cycler. I will used this in the hist plots
        color_from_cycler = plt.rcParams['axes.prop_cycle'].by_key()["color"]

        # ------------------------------------------- Temperature -------------------------------------------#
        # Calculate Temperature plot's labels and multipliers
        time_mul, temp_mul, time_prefix, temp_prefix, time_lbl, temp_lbl = plot_labels(self.dataframe["Time"],
                                                                                       self.dataframe["Temperature"],
                                                                                       "Time",
                                                                                       "Temperature", self.units)
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
        T_main_plot.plot(time, T_cumavg, label='Moving Average')
        T_main_plot.axhline(T_desired, ls='--', c='r', alpha=0.7, label='Desired T')
        T_main_plot.legend(loc='best')
        T_main_plot.set(ylabel="Temperature" + temp_lbl, xlabel="Time" + time_lbl)

        # Temperature Deviation plot
        T_delta_plot.plot(time, Delta_T, alpha=0.5)
        T_delta_plot.plot(time, Delta_T_cum_avg, alpha=0.8)
        T_delta_plot.set(xticks=[], ylabel=r'Deviation [%]')

        # This was a failed attempt to calculate the theoretical Temperature distribution.
        # Gaussian
        T_dist = scp_stats.norm(loc=T_desired, scale=Temperature.std())
        # Histogram plot
        sns.histplot(y=Temperature,
                     bins='auto',
                     stat='density',
                     alpha=0.75,
                     legend='False',
                     ax=T_hist_plot)
        T_hist_plot.set(ylabel=None, xlabel=None, xticks=[], yticks=[])
        T_hist_plot.plot(T_dist.pdf(Temperature), Temperature, color=color_from_cycler[1])

        # ------------------------------------------- Total Energy -------------------------------------------#
        # Calculate Energy plot's labels and multipliers
        time_mul, energy_mul, _, _, time_lbl, energy_lbl = plot_labels(self.dataframe["Time"],
                                                                       self.dataframe["Total Energy"],
                                                                       "Time",
                                                                       "Energy",
                                                                       self.units)

        Energy = energy_mul * self.dataframe["Total Energy"]
        # Total Energy moving average
        E_cumavg = Energy.expanding().mean()
        # Total Energy Deviation and its moving average
        Delta_E = (Energy - Energy.iloc[0]) * 100 / Energy.iloc[0]
        Delta_E_cum_avg = Delta_E.expanding().mean()

        # Energy main plot
        E_main_plot.plot(time, Energy, alpha=0.7)
        E_main_plot.plot(time, E_cumavg, label='Moving Average')
        E_main_plot.axhline(Energy.mean(), ls='--', c='r', alpha=0.7, label='Avg')
        E_main_plot.legend(loc='best')
        E_main_plot.set(ylabel="Total Energy" + energy_lbl, xlabel="Time" + time_lbl)

        # Deviation Plot
        E_delta_plot.plot(time, Delta_E, alpha=0.5)
        E_delta_plot.plot(time, Delta_E_cum_avg, alpha=0.8)
        E_delta_plot.set(xticks=[], ylabel=r'Deviation [%]')

        # (Failed) Attempt to calculate the theoretical Energy distribution
        # In an NVT ensemble Energy fluctuation are given by sigma(E) = sqrt( k_B T^2 C_v)
        # where C_v is the isothermal heat capacity
        # Since this requires a lot of prior calculation I skip it and just make a Gaussian
        E_dist = scp_stats.norm(loc=Energy.mean(), scale=Energy.std())
        # Histogram plot
        sns.histplot(y=Energy,
                     bins='auto',
                     stat='density',
                     alpha=0.75,
                     legend='False',
                     ax=E_hist_plot)
        # Grab the second color since the first is used for histplot
        E_hist_plot.plot(E_dist.pdf(Energy), Energy, color=color_from_cycler[1])

        E_hist_plot.set(ylabel=None, xlabel=None, xticks=[], yticks=[])

        if not publication:
            dt_mul, _, _, _, dt_lbl, _ = plot_labels(process.integrator.dt,
                                                     self.dataframe["Total Energy"],
                                                     "Time",
                                                     "Energy",
                                                     self.units)
            # Information section
            Info_plot.axis([0, 10, 0, 10])
            Info_plot.grid(False)

            Info_plot.text(0., 10, "Job ID: {}".format(self.job_id))
            Info_plot.text(0., 9.5, "Phase: {}".format(self.phase.capitalize()))
            Info_plot.text(0., 9.0, "No. of species = {}".format(len(self.species_num)))
            y_coord = 8.5
            for isp, sp in enumerate(process.species):
                Info_plot.text(0., y_coord, "Species {} : {}".format(isp + 1, sp.name))
                Info_plot.text(0.0, y_coord - 0.5, "  No. of particles = {} ".format(sp.num))
                Info_plot.text(0.0, y_coord - 1.,
                               "  Temperature = {:.2f} {}".format(temp_mul * sp.temperature, temp_lbl))
                y_coord -= 1.5

            y_coord -= 0.25
            Info_plot.text(0., y_coord, "Total $N$ = {}".format(process.parameters.total_num_ptcls))
            Info_plot.text(0., y_coord - 0.5, "Thermostat: {}".format(process.thermostat.type))
            Info_plot.text(0., y_coord - 1., "Berendsen rate = {:1.2f}".format(process.thermostat.relaxation_rate))
            Info_plot.text(0., y_coord - 1.5, "Potential: {}".format(process.potential.type))
            Info_plot.text(0., y_coord - 2., "Tot Force Error = {:1.2e}".format(process.parameters.force_error))
            delta_t = dt_mul * process.integrator.dt
            Info_plot.text(0., y_coord - 2.5, "$\Delta t$ = {:.2f} {}".format(delta_t, dt_lbl))
            Info_plot.text(0., y_coord - 3., "Step interval = {}".format(self.dump_step))
            Info_plot.text(0., y_coord - 3.5,
                           "Step interval time = {:.2f} {}".format(self.dump_step * delta_t, dt_lbl))
            Info_plot.text(0., y_coord - 4., "Completed steps = {}".format(completed_steps))
            Info_plot.text(0., y_coord - 4.5,
                           "Completed time = {:.2f} {}".format(completed_steps * delta_t / dt_mul * time_mul, time_lbl))
            Info_plot.text(0., y_coord - 5., "Total timesteps = {}".format(self.no_steps))
            Info_plot.text(0., y_coord - 5.5,
                           "Total time = {:.2f} {}".format(self.no_steps * delta_t / dt_mul * time_mul, time_lbl))
            Info_plot.text(0., y_coord - 6.,
                           "{:1.2f} % Completed".format(100 * completed_steps / self.no_steps))
            Info_plot.axis('off')

        if not publication:
            fig.tight_layout()

        # Saving
        if figname:
            fig.savefig(os.path.join(self.saving_dir, figname + '_' + self.job_id + '.png'))
        else:
            fig.savefig(os.path.join(self.saving_dir, 'Plot_EnsembleCheck_' + self.job_id + '.png'))

        if show:
            fig.show()

        # Restore the previous rcParams
        plt.rcParams = current_rcParams


class VelocityAutoCorrelationFunction(Observable):
    """Velocity Auto-correlation function."""

    def setup(self, params,
              phase: str = None,
              time_averaging: bool = False,
              timesteps_to_skip: int = 100,
              **kwargs):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to compute. Default = 'production'.

        params : sarkas.base.Parameters
            Simulation's parameters.

        time_averaging: bool
            Flag for species diffusion flux time averaging. Default = False.

        timesteps_to_skip: int
            Number of timesteps to skip for time_averaging. Default = 100.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        self.__name__ = 'vacf'

        if phase:
            self.phase = phase.lower()

        self.time_averaging = time_averaging
        self.timesteps_to_skip = timesteps_to_skip

        super().setup_init(params, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'VelocityAutoCorrelationFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "VelocityACF_" + self.job_id + '.csv')

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

    def compute(self, **kwargs):
        """
        Compute the velocity auto-correlation functions.

        Parameters
        ----------
        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

        # Parse the particles from the dump files
        vel = np.zeros((self.dimensions, self.total_num_ptcls, self.no_dumps))
        #
        print("Parsing particles' velocities.")
        time = np.zeros(self.no_dumps)
        for it in tqdm(range(self.no_dumps), disable=(not self.verbose)):
            dump = int(it * self.dump_step)
            time[it] = dump * self.dt
            datap = load_from_restart(self.dump_dir, dump)
            vel[0, :, it] = datap["vel"][:, 0]
            vel[1, :, it] = datap["vel"][:, 1]
            vel[2, :, it] = datap["vel"][:, 2]
        #
        self.dataframe["Time"] = time
        message = "Calculating velocity acf with time averaging "
        ta = "on" if self.time_averaging else "off"
        print('Please wait. ' + message + ta + ' ...')

        t0 = self.timer.current()
        vacf = calc_vacf(vel, self.species_num, self.time_averaging, self.timesteps_to_skip)
        tend = self.timer.current()

        self.time_stamp("VACF Calculation", self.timer.time_division(tend - t0))

        for i, sp1 in enumerate(self.species_names):
            self.dataframe["{} X Velocity ACF".format(sp1)] = vacf[i, 0, :]
            self.dataframe["{} Y Velocity ACF".format(sp1)] = vacf[i, 1, :]
            self.dataframe["{} Z Velocity ACF".format(sp1)] = vacf[i, 2, :]
            self.dataframe["{} Total Velocity ACF".format(sp1)] = vacf[i, 3, :]

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')


class FluxAutoCorrelationFunction(Observable):
    """Species Diffusion Flux Auto-correlation function."""

    def setup(self,
              params,
              phase: str = None,
              time_averaging: bool = False,
              timesteps_to_skip: int = 100,
              **kwargs):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to compute. Default = 'production'.

        params : sarkas.base.Parameters
            Simulation's parameters.

        time_averaging: bool
            Flag for species diffusion flux time averaging. Default = False.

        timesteps_to_skip: int
            Number of timesteps to skip for time_averaging. Default = 100.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        self.__name__ = 'facf'

        if phase:
            self.phase = phase.lower()

        self.time_averaging = time_averaging
        self.timesteps_to_skip = timesteps_to_skip

        super().setup_init(params, self.phase)

        if not hasattr(self, 'species_mass_densities'):
            self.species_mass_densities = self.species_num_dens * self.species_masses

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'DiffusionFluxAutoCorrelationFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "DiffusionFluxACF_" + self.job_id + '.csv')

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

    def compute(self, **kwargs):
        """
        Compute the velocity auto-correlation functions.

        Parameters
        ----------
        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """
        # Update the attribute with the passed arguments. e.g time_averaging and timesteps_to_skip
        self.__dict__.update(kwargs.copy())

        # Parse the particles from the dump files
        vel = np.zeros((self.dimensions, self.no_dumps, self.total_num_ptcls))
        #
        print("Parsing particles' velocities.")
        time = np.zeros(self.no_dumps)
        for it in tqdm(range(self.no_dumps), disable=(not self.verbose)):
            dump = int(it * self.dump_step)
            datap = load_from_restart(self.dump_dir, dump)
            time[it] = datap["time"]
            vel[0, it, :] = datap["vel"][:, 0]
            vel[1, it, :] = datap["vel"][:, 1]
            vel[2, it, :] = datap["vel"][:, 2]
        #
        self.dataframe["Time"] = time
        message = "Calculating diffusion flux acf with time averaging "
        ta = "on" if self.time_averaging else "off"
        print('Please wait. ' + message + ta + ' ...')
        t0 = self.timer.current()
        df_acf = calc_diff_flux_acf(vel,
                                    self.species_num,
                                    self.species_num_dens,
                                    self.species_masses,
                                    self.time_averaging,
                                    self.timesteps_to_skip)
        tend = self.timer.current()

        self.time_stamp("Diffusion Flux ACF Calculation", self.timer.time_division(tend - t0))

        # Store the data
        v_ij = 0
        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:], i):
                self.dataframe["{}-{} X Diffusion Flux ACF".format(sp1, sp2)] = df_acf[v_ij, 0, :]
                self.dataframe["{}-{} Y Diffusion Flux ACF".format(sp1, sp2)] = df_acf[v_ij, 1, :]
                self.dataframe["{}-{} Z Diffusion Flux ACF".format(sp1, sp2)] = df_acf[v_ij, 2, :]
                self.dataframe["{}-{} Total Diffusion Flux ACF".format(sp1, sp2)] = df_acf[v_ij, 3, :]
                v_ij += 1

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')


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

    def setup(self,
              params,
              phase: str = None,
              hist_kwargs: dict = None,
              max_no_moment: int = 6,
              curve_fit_kwargs: dict = None,
              **kwargs):

        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        hist_kwargs : dict, optional
            Dictionary of keyword arguments to pass to ``np.histogram`` for the calculation of the distributions.

        phase : str
            Phase to compute. Default = 'production'.

        max_no_moment : int
            Maximum number of moments to calculate. Default = 6.

        params : sarkas.base.Parameters
            Simulation's parameters.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        self.__name__ = 'vd'
        self.__long_name__ = 'Velocity Distribution'
        if curve_fit_kwargs:
            self.curve_fit_kwargs = curve_fit_kwargs

        if phase:
            self.phase = phase.lower()

        super().setup_init(params, self.phase)
        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())
        # Check on hist_kwargs
        if hist_kwargs:
            # Is it a dictionary ?
            assert isinstance(hist_kwargs, dict), 'hist_kwargs not a dictionary. Please pass a dictionary.'
            self.hist_kwargs = hist_kwargs
        # Default number of moments to calculate
        self.max_no_moment = max_no_moment

        # Create the directory where to store the computed data
        # First the name of the observable
        saving_dir = os.path.join(self.postprocessing_dir, 'VelocityDistribution')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)
        # then the phase
        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())

        self.adjusted_dump_dir = []

        if self.multi_run_average:
            for r in range(self.runs):
                # Direct to the correct dumps directory
                dump_dir = os.path.join('run{}'.format(r), os.path.join('Simulation',
                                                                        os.path.join(self.phase.capitalize(), 'dumps')))
                dump_dir = os.path.join(self.simulations_dir, dump_dir)
                self.adjusted_dump_dir.append(dump_dir)
            # Re-path the saving directory
            saving_dir = os.path.join(self.simulations_dir, 'PostProcessing')
            if not os.path.exists(saving_dir):
                os.mkdir(saving_dir)
            saving_dir = os.path.join(saving_dir, self.phase.capitalize())
            if not os.path.exists(saving_dir):
                os.mkdir(saving_dir)
            self.saving_dir = os.path.join(saving_dir, 'VelocityDistribution')
        else:
            self.adjusted_dump_dir = [self.dump_dir]

        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        # Directories in which to store plots
        self.plots_dir = os.path.join(self.saving_dir, 'Plots')
        if not os.path.exists(self.plots_dir):
            os.mkdir(self.plots_dir)

        # Paths where to store the dataframes
        self.filename_csv = os.path.join(self.saving_dir, "VelocityDistribution_" + self.job_id + '.csv')
        self.filename_hdf = os.path.join(self.saving_dir, "VelocityDistribution_" + self.job_id + '.h5')

        if hasattr(self, 'max_no_moment'):
            self.moments_dataframe = None
            self.mom_df_filename_csv = os.path.join(self.saving_dir, "Moments_" + self.job_id + '.csv')

        if hasattr(self, 'max_hermite_order'):
            self.hermite_dataframe = None
            self.herm_df_filename_csv = os.path.join(self.saving_dir, "HermiteCoefficients_" + self.job_id + '.csv')
            # some checks
            if not hasattr(self, 'hermite_rms_tol'):
                self.hermite_rms_tol = 0.05

        self.species_plots_dirs = None

        # Need this for pretty print
        # Calculate the dimension of the velocity container
        # 2nd Dimension of the raw velocity array
        self.dim = 1 if self.dimensional_average else self.dimensions
        # range(inv_dim) for the loop over dimension
        self.inv_dim = self.dimensions if self.dimensional_average else 1

        # Array containing the start index of each species. The last value is equivalent to vel_raw.shape[-1]
        self.species_index_start = np.array([0, *np.cumsum(self.species_num)], dtype=int) * self.inv_dim * self.runs

        # Check if arguments have been passed
        if hist_kwargs:
            # Did you pass a single dictionary for multispecies?
            for key, value in hist_kwargs.items():
                # The elements of hist_kwargs should be lists
                if not isinstance(hist_kwargs[key], list):
                    hist_kwargs[key] = [value for i in range(self.num_species)]

            # Override whatever was passed via YAML or setup
            self.hist_kwargs.update(hist_kwargs.copy())

        self.prepare_histogram_args()

    def grab_sim_data(self):
        """Read in velocity data"""

        # Velocity array for storing simulation data
        vel_raw = np.zeros((self.no_dumps, self.dim, self.runs * self.inv_dim * self.total_num_ptcls))
        time = np.zeros(self.no_dumps)

        print("\nCollecting data from snapshots ...")
        if self.dimensional_average:
            # Loop over the runs
            for r, dump_dir_r in enumerate(tqdm(self.adjusted_dump_dir, disable=(not self.verbose), desc='Runs Loop')):
                # Loop over the timesteps
                for it in tqdm(range(self.no_dumps), disable=(not self.verbose), desc='Timestep Loop'):
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
                        vel_raw[it, 0, start_indx: end_indx] = datap["vel"][datap["names"] == sp_name].flatten('F')

                    time[it] = datap["time"]
        else:  # Dimensional Average = False
            # Loop over the runs
            for r, dump_dir_r in enumerate(tqdm(self.adjusted_dump_dir, disable=(not self.verbose), desc='Runs Loop')):
                # Loop over the timesteps
                for it in tqdm(range(self.no_dumps), disable=(not self.verbose), desc='Timestep Loop'):
                    # Read data from file
                    dump = int(it * self.dump_step)
                    datap = load_from_restart(dump_dir_r, dump)
                    # Loop over the particles' species
                    for sp_indx, (sp_name, sp_num) in enumerate(zip(self.species_names, self.species_num)):
                        # Calculate the correct start and end index for storage
                        start_indx = self.species_index_start[sp_indx] + self.inv_dim * sp_num * r
                        end_indx = self.species_index_start[sp_indx] + self.inv_dim * sp_num * (r + 1)
                        # Use a mask to grab only the selected species and transpose the array to put dimensions first
                        vel_raw[it, :, start_indx: end_indx] = datap["vel"][datap["names"] == sp_name].transpose()

                time[it] = datap["time"]

        return time, vel_raw

    def compute(self, hist_kwargs: dict = None, **kwargs):
        """
        Calculate the moments of the velocity distributions and save them to a pandas dataframes and csv.

        Parameters
        ----------
        hist_kwargs : dict, optional
            Dictionary with arguments to pass to ``numpy.histogram``.

        **kwargs :
            These are will overwrite any ``sarkas.base.Parameters`` or default ``sarkas.tools.observables.Observable``
            attributes and/or add new ones.

        """

        # Update the attribute with the passed arguments
        self.__dict__.update(kwargs.copy())

        # Check if arguments have been passed
        if hist_kwargs:
            # Did you pass a single dictionary for multispecies?
            for key, value in hist_kwargs.items():
                # The elements of hist_kwargs should be lists
                if not isinstance(hist_kwargs[key], list):
                    hist_kwargs[key] = [value for i in range(self.num_species)]

            # Override whatever was passed via YAML or setup
            self.hist_kwargs.update(hist_kwargs.copy())

        # Make the histogram arguments
        self.prepare_histogram_args()

        # Print info to screen
        self.pretty_print()

        # Ok let's do it.
        self.dataframe = pd.DataFrame()

        # Calculate the dimension of the velocity container

        # 2nd Dimension of the raw velocity array
        self.dim = 1 if self.dimensional_average else self.dimensions
        # range(inv_dim) for the loop over dimension
        self.inv_dim = self.dimensions if self.dimensional_average else 1
        # Array containing the start index of each species. The last value is equivalent to vel_raw.shape[-1]
        self.species_index_start = np.array([0, *np.cumsum(self.species_num)], dtype=int) * self.inv_dim * self.runs

        # Grab simulation data
        time, vel_raw = self.grab_sim_data()

        # Make the velocity distribution
        self.create_distribution(vel_raw, time)

        # Calculate velocity moments
        if hasattr(self, "max_no_moment"):
            self.compute_moments(False, vel_raw, time)
        #
        if hasattr(self, "max_hermite_order"):
            self.compute_hermite_expansion(False)

    def prepare_histogram_args(self):

        # Initialize histograms arguments
        if not hasattr(self, 'hist_kwargs'):
            self.hist_kwargs = {'density': [],
                                'bins': [],
                                'range': []
                                }
        # Default values
        bin_width = 0.05
        # Range of the histogram = (-wid * vth, wid * vth)
        wid = 5
        # The number of bins is calculated from default values of bin_width and wid
        no_bins = int(2.0 * wid / bin_width)
        # Calculate thermal speed from energy/temperature data.
        try:
            energy_fle = self.prod_energy_filename if self.phase == 'production' else self.eq_energy_filename
            energy_df = pd.read_csv(energy_fle, index_col=False, encoding='utf-8')
            if self.num_species > 1:
                vth = np.zeros(self.num_species)
                for sp, (sp_mass, sp_name) in enumerate(zip(self.species_masses, self.species_names)):
                    vth[sp] = np.sqrt(energy_df["{} Temperature".format(sp_name)].mean() * self.kB / sp_mass)
            else:
                vth = np.sqrt(energy_df["Temperature"].mean() * self.kB / self.species_masses)

        except FileNotFoundError:
            vth = np.sqrt(self.kB * self.T_desired / self.species_masses)

        self.vth = np.copy(vth)

        # Create the default dictionary of histogram args
        default_hist_kwargs = {'density': [],
                               'bins': [],
                               'range': []}
        if self.num_species > 1:
            for sp in range(self.num_species):
                default_hist_kwargs['density'].append(True)
                default_hist_kwargs['bins'].append(no_bins)
                default_hist_kwargs['range'].append((-wid * vth[sp], wid * vth[sp]))
        else:
            default_hist_kwargs['density'].append(True)
            default_hist_kwargs['bins'].append(no_bins)
            default_hist_kwargs['range'].append((-wid * vth[0], wid * vth[0]))

        # Now do some checks.
        # Check for numpy.histogram args in kwargs
        must_have_keys = ['bins', 'range', 'density']

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
                        full_df_columns.append(
                            ["{}_{}_{:6e}".format(sp_name, ds, be) for be in bin_loc])
                        self.species_bin_edges[sp_name][ds] = bin_edges
                        # Ok. Now I have created the bins columns' names
                    # Time to insert in the huge matrix.
                    indx_1 = indx_0 + 1 + len(bin_count)
                    dist_matrix[it, indx_0] = time[it]
                    dist_matrix[it, indx_0 + 1:indx_1] = bin_count
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
        self.dataframe.to_csv(self.filename_csv, encoding='utf-8', index=False)

        # Hierarchical DataFrame
        self.hierarchical_dataframe = self.dataframe.copy()
        self.hierarchical_dataframe.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("_")) for c in self.hierarchical_dataframe.columns])
        self.hierarchical_dataframe.to_hdf(self.filename_hdf, key='velocity_distribution', encoding='utf-8')

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
                    self.moments_hdf_dataframe["{}_{}_Time".format(sp,ds)] = time
                    for m in range(self.max_no_moment):
                        self.moments_dataframe["{} {} moment axis {}".format(sp, m + 1, ds)] = moments[i, :, d, m][:, 0]
                        self.moments_hdf_dataframe["{}_{}_{} moment".format(sp, ds, m + 1)] = moments[i, :, :, m][:, 0]

                for d, ds in zip(range(self.dim), ["X", "Y", "Z"]):
                    self.moments_hdf_dataframe["{}_{}_Time".format(sp, ds)] = time
                    for m in range(self.max_no_moment):
                        self.moments_dataframe[
                            "{} {} moment ratio axis {}".format(sp, m + 1, ds)] = ratios[i, :, d, m][:, 0]
                        self.moments_hdf_dataframe[
                            "{}_{}_{}-2 ratio ".format(sp, ds, m + 1)] = ratios[i, :, d, m][:, 0]

        self.moments_dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')
        # Hierarchical DF Save
        # Make the columns
        self.moments_hdf_dataframe.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("_")) for c in self.moments_hdf_dataframe.columns])
        # Save the df in the hierarchical df with a new key/group
        self.moments_hdf_dataframe.to_hdf(
            self.filename_hdf,
            mode='a',
            key='velocity_moments',
            encoding='utf-8'
        )

    def compute_hermite_expansion(self, calc_moments: bool = False):
        """
        Calculate and save Hermite coefficients of the Grad expansion.

        Parameters
        ----------
        calc_moments: bool
            Flag for calculating velocity moments. These are needed for the hermite calculation.
            Default = False.

        """
        from scipy.optimize import curve_fit

        self.hermite_dataframe = pd.DataFrame()
        self.hermite_hdf_dataframe = pd.DataFrame()

        if calc_moments:
            self.compute_moments(parse_data=True)

        if not hasattr(self,'hermite_rms_tol'):
            self.hermite_rms_tol = 0.05

        self.hermite_dataframe["Time"] = np.copy(self.moments_dataframe["Time"])
        self.hermite_sigmas = np.zeros((self.num_species, self.dim, len(self.hermite_dataframe["Time"])))
        self.hermite_epochs = np.zeros((self.num_species, self.dim, len(self.hermite_dataframe["Time"])))
        hermite_coeff = np.zeros(
            (self.num_species, self.dim, self.max_hermite_order + 1, len(self.hermite_dataframe["Time"])))

        print("\nCalculating Hermite coefficients ...")
        tinit = self.timer.current()

        for sp, sp_name in enumerate(tqdm(self.species_names, desc='Species')):
            for it, t in enumerate(tqdm(self.hermite_dataframe["Time"], desc='Time')):
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
                            maxfev=1000)  # TODO: let the user pass curve_fit arguments.

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
                        self.hermite_hdf_dataframe[
                            "{}_{}_Time".format(sp_name, ds, h)] = hermite_coeff[sp, d, h, :]
                        self.hermite_hdf_dataframe[
                            "{}_{}_RMS".format(sp_name, ds, h)] = self.hermite_sigmas[sp, d, :]
                        self.hermite_hdf_dataframe[
                            "{}_{}_epoch".format(sp_name, ds, h)] = self.hermite_epochs[sp, d, :]
                    else:
                        self.hermite_hdf_dataframe["{}_{}_{} coeff".format(sp_name, ds, h)] = hermite_coeff[sp, d, h, :]

        # Save the CSV
        self.hermite_dataframe.to_csv(self.herm_df_filename_csv, index=False, encoding='utf-8')
        # Make the columns
        self.hermite_hdf_dataframe.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("_")) for c in self.hermite_hdf_dataframe.columns])
        # Save the df in the hierarchical df with a new key/group
        self.hermite_hdf_dataframe.to_hdf(
            self.filename_hdf,
            mode='a',
            key='hermite_coefficients',
            encoding='utf-8'
        )

        self.time_stamp("Hermite expansion calculation", self.timer.time_division(tend - tinit))

    def pretty_print(self):
        """Print information in a user-friendly way."""

        print('\n\n{:=^70} \n'.format(' ' + self.__long_name__ + ' '))
        print('CSV dataframe saved in:\n\t ', self.filename_csv)
        print('HDF5 dataframe saved in:\n\t ', self.filename_hdf)
        print('Data accessible at: self.dataframe, self.hierarchical_dataframe, self.species_bin_edges')
        print('\nMulti run average: ', self.multi_run_average)
        print('No. of runs: ', self.runs)
        print("Size of the parsed velocity array: {} x {} x {}".format(self.no_dumps,
                                                                       self.dim,
                                                                       self.runs * self.inv_dim * self.total_num_ptcls))
        print('\nHistograms Information:')
        for sp, (sp_name, dics) in enumerate(zip(self.species_names, self.list_hist_kwargs)):
            if sp == 0:
                print("Species: {}".format(sp_name))
            else:
                print("\nSpecies: {}".format(sp_name))
            print("No. of samples = {}".format(self.species_num[sp] * self.inv_dim * self.runs))
            print("Thermal speed: v_th = {:.6e} ".format(self.vth[sp]), end='')
            print("[cm/s]" if self.units == "cgs" else "[m/s]")
            for key, values in dics.items():
                if key == 'range':
                    print("{} : ( {:.2f}, {:.2f} ) v_th,"
                          "\n\t( {:.4e}, {:.4e} ) ".format(key, *values / self.vth[sp], *values), end='')
                    print("[cm/s]" if self.units == "cgs" else "[m/s]")
                else:
                    print("{}: {}".format(key, values))
            bin_width = abs(dics["range"][1] - dics["range"][0]) / (self.vth[sp] * dics["bins"])
            print("Bin Width = {:.4f}".format(bin_width))

        if hasattr(self,"max_no_moment"):
            print('\nMoments Information:')
            print('CSV dataframe saved in:\n\t ', self.mom_df_filename_csv)
            print('Data accessible at: self.moments_dataframe, self.moments_hdf_dataframe')
            print('Highest moment to calculate: {}'.format(self.max_no_moment))

        if hasattr(self,"max_hermite_order"):
            print('\nGrad Expansion Information:')
            print('CSV dataframe saved in:\n\t ', self.herm_df_filename_csv)
            print('Data accessible at: self.hermite_dataframe, self.hermite_hdf_dataframe')
            print('Highest order to calculate: {}'.format(self.max_hermite_order))
            print('RMS Tolerance: {:.3f}'.format(self.hermite_rms_tol))


@njit
def calc_Sk(nkt, ka_list, ka_counts, species_np, no_dumps):
    """
    Calculate :math:`S_{ij}(k)` at each saved timestep.

    Parameters
    ----------
    nkt : numpy.ndarray, complex
        Density fluctuations of all species. Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    ka_list :
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``ka_counts``.
        Shape=(``no_ka_values``, 5)

    ka_counts : numpy.ndarray
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
    Sk_all = np.zeros((no_sk, len(ka_counts), no_dumps))

    pair_indx = 0
    for ip, si in enumerate(species_np):
        for jp in range(ip, len(species_np)):
            sj = species_np[jp]
            for it in range(no_dumps):
                for ik, ka in enumerate(ka_list):
                    indx = int(ka[-1])
                    nk_i = nkt[ip, it, ik]
                    nk_j = nkt[jp, it, ik]
                    Sk_all[pair_indx, indx, it] += np.real(np.conj(nk_i) * nk_j) / (ka_counts[indx] * np.sqrt(si * sj))
            pair_indx += 1

    return Sk_all


def calc_Skw(nkt, ka_list, ka_counts, species_np, no_dumps, dt, dump_step):
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
    Skw = np.zeros((no_skw, len(ka_counts), no_dumps))

    pair_indx = 0
    for ip, si in enumerate(species_np):
        for jp in range(ip, len(species_np)):
            sj = species_np[jp]
            for ik, ka in enumerate(ka_list):
                indx = int(ka[-1])
                nkw_i = np.fft.fft(nkt[ip, :, ik]) * norm
                nkw_j = np.fft.fft(nkt[jp, :, ik]) * norm
                Skw[pair_indx, indx, :] += np.real(np.conj(nkw_i) * nkw_j) / (ka_counts[indx] * np.sqrt(si * sj))
            pair_indx += 1
    return Skw


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
    num_species = len(sp_num)
    no_dumps = vel.shape[0]

    Js = np.zeros((num_species, 3, no_dumps))
    Jtot = np.zeros((3, no_dumps))

    for it in range(no_dumps):
        sp_start = 0
        sp_end = 0
        for s, q_sp, n_sp in zip(range(len(num_species)), sp_charge, num_species):
            # Find the index of the last particle of species s
            sp_end += n_sp
            # Calculate the current of each species
            Js[s, :, it] = q_sp * np.sum(vel[it, :, sp_start:sp_end], axis=1)
            # Add to the total current
            Jtot[:, it] += Js[s, :, it]

            sp_start += sp_end

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
        ratios[:, :, :, mom] = moments[:, :, :, mom] / (const * moments[:, :, :, 1] ** (pwr / 2.))

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
        Timestep interval saving.

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
def calc_pressure_tensor(pos, vel, acc, species_mass, species_np, box_volume):
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
    # Rescale vel and acc of each particle by their individual mass
    for sp, num in enumerate(species_np):
        sp_end += num
        vel[:, sp_start: sp_end] *= np.sqrt(species_mass[sp])
        acc[:, sp_start: sp_end] *= species_mass[sp]  # force
        sp_start += num

    pressure_tensor = (vel @ np.transpose(vel) + pos @ np.transpose(acc)) / box_volume
    pressure = np.trace(pressure_tensor) / 3.0

    return pressure, pressure_tensor


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
        sigma2_blk[i] /= (i - 1)
        statistical_efficiency[i] = tau_blk[i] * sigma2_blk[i] / run_std ** 2

    return tau_blk, sigma2_blk, statistical_efficiency


# @jit Numba doesn't like scipy.signal
def calc_diff_flux_acf(vel, sp_num, sp_dens, sp_mass, time_averaging, timesteps_to_skip):
    """
    Calculate the diffusion flux autocorrelation function of each species and in each direction.

    Parameters
    ----------
    time_averaging: bool
        Flag for time averaging.

    timesteps_to_skip: int
        Number of timesteps to skip for time_averaging.

    vel : numpy.ndarray
        Particles' velocities. Shape = (``dimensions``, ``no_dumps``, ``total_num_ptcls``)

    sp_num: numpy.ndarray
        Number of particles of each species.

    sp_dens: numpy.ndarray
        Number densities of each species.

    sp_mass: numpy.ndarray
        Particle's mass of each species.

    Returns
    -------
    jc_acf: numpy.ndarray
        Diffusion flux autocorrelation function. Shape = ( Ns*(Ns +1)/2, Ndim + 1 , Nt)
        where Ns = number of species, Ndim = Number of cartesian dimensions, Nt = Number of dumps.
    """

    no_dim = vel.shape[0]
    no_dumps = vel.shape[1]
    no_species = len(sp_num)
    no_vacf = int(no_species * (no_species + 1) / 2.)

    mass_densities = sp_dens * sp_mass
    tot_mass_dens = np.sum(mass_densities)
    # Center of mass velocity field of each species in each direction and at each timestep
    com_vel = np.zeros((no_species, no_dim, no_dumps))
    # Total center of mass velocity field, see eq.(18) in
    # Haxhimali T. et al., Diffusivity of Mixtures in Warm Dense Matter Regime.In: Graziani F., et al. (eds)
    # Frontiers and Challenges in Warm Dense Matter. Lecture Notes in Computational Science and Engineering, vol 96.
    # Springer (2014)
    tot_com_vel = np.zeros((no_dim, no_dumps))

    sp_start = 0
    sp_end = 0
    # Calculate the total center of mass velocity (tot_com_vel)
    # and the center of mass velocity of each species (com_vel)
    for i, ns in enumerate(sp_num):
        sp_end += ns
        com_vel[i, :, :] = np.sum(vel[:, :, sp_start: sp_end], axis=-1)
        tot_com_vel += mass_densities[i] * com_vel[i, :, :] / tot_mass_dens
        sp_start = sp_end

    jc_acf = np.zeros((no_vacf, no_dim + 1, no_dumps))
    it_skip = timesteps_to_skip if time_averaging else no_dumps
    indx = 0
    # the flux is given by eq.(19) of the above reference
    for i, rho1 in enumerate(mass_densities):
        # Flux of species i
        sp1_flux = rho1 * (com_vel[i] - tot_com_vel)
        for j, rho2 in enumerate(mass_densities[i:], i):
            # this sign seems to be an issue in the calculation of the flux
            sign = (1 - 2 * (i != j))
            # Flux of species j
            sp2_flux = sign * rho2 * (com_vel[j] - tot_com_vel)
            # Calculate the correlation function in each direction
            for d in range(no_dim):
                # Counter for time origins
                norm_counter = np.zeros(no_dumps)
                # temporary storage of correlation function
                temp = np.zeros(no_dumps)
                # Grab the correct time intervals
                for it in range(0, no_dumps, it_skip):
                    # Calculate the correlation function and add it to the array
                    temp[:no_dumps - it] += correlationfunction(sp1_flux[d, it:], sp2_flux[d, it:])
                    # Note that norm_counter = 1 if time_averaging is false
                    norm_counter[:(no_dumps - it)] += 1.0
                # Divide by the number time origins
                jc_acf[indx, d, :] = temp / norm_counter
                # Calculate the total correlation function by summing the three directions
                jc_acf[indx, -1, :] += temp / norm_counter
            indx += 1

    return jc_acf


# @jit Numba doesn't like Scipy
def calc_vacf(vel, sp_num, time_averaging, timesteps_to_skip):
    """
    Calculate the velocity autocorrelation function of each species and in each direction.

    Parameters
    ----------
    time_averaging: bool
        Flag for time averaging.

    timesteps_to_skip: int
        Number of timesteps to skip for time_averaging.

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

    it_skip = timesteps_to_skip if time_averaging else no_dumps
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
                # temporary vacf
                ptcl_vacf = np.zeros(no_dumps)
                # Counter of time origins
                norm_counter = np.zeros(no_dumps)
                # Grab the correct time interval
                for it in range(0, no_dumps, it_skip):
                    # Calculate the correlation function and add it to the array
                    ptcl_vacf[:no_dumps - it] += correlationfunction(vel[i, ptcl, it:], vel[i, ptcl, it:])
                    # Note that norm_counter = 1 if time_averaging is false
                    norm_counter[:(no_dumps - it)] += 1.0

                # Add this particle vacf to the species vacf and normalize by the time origins
                sp_vacf += ptcl_vacf / norm_counter

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
                pos[sp_start:sp_end, :], vel[sp_start:sp_end], k_list)
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
    gaussian = np.exp(- 0.5 * (x / rms) ** 2) / (np.sqrt(2.0 * np.pi * rms ** 2))

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
        List of all possible :math:`k` vectors with their corresponding magnitudes and indexes.

    k_counts : numpy.ndarray
        Number of occurrences of each :math:`k` magnitude.

    k_unique : numpy.ndarray
        Magnitude of each allowed :math:`k` vector.
    """
    if angle_averaging == 'full':
        # The first value of k_arr = [0, 0, 0]
        first_non_zero = 1
        # Obtain all possible permutations of the wave number arrays
        k_arr = [np.array([i / box_lengths[0], j / box_lengths[1], k / box_lengths[2]])
                 for i in range(max_k_harmonics[0] + 1)
                 for j in range(max_k_harmonics[1] + 1)
                 for k in range(max_k_harmonics[2] + 1)]
    elif angle_averaging == 'principal_axis':
        # The first value of k_arr = [1, 0, 0]
        first_non_zero = 0
        # Calculate the k vectors along the principal axis only
        k_arr = [np.array([i / box_lengths[0], 0, 0]) for i in range(1, max_k_harmonics[0] + 1)]
        k_arr = np.append(k_arr,
                          [np.array([0, i / box_lengths[1], 0]) for i in range(1, max_k_harmonics[1] + 1)],
                          axis=0)
        k_arr = np.append(k_arr,
                          [np.array([0, 0, i / box_lengths[2]]) for i in range(1, max_k_harmonics[2] + 1)],
                          axis=0)
    elif angle_averaging == 'custom':
        # The first value of k_arr = [0, 0, 0]
        first_non_zero = 1
        # Obtain all possible permutations of the wave number arrays up to max_aa_harmonics included
        k_arr = [np.array([i / box_lengths[0],
                           j / box_lengths[1],
                           k / box_lengths[2]]) for i in range(max_aa_harmonics[0] + 1)
                 for j in range(max_aa_harmonics[1] + 1)
                 for k in range(max_aa_harmonics[2] + 1)]
        # Append the rest of k values calculated from principal axis
        k_arr = np.append(
            k_arr,
            [np.array([i / box_lengths[0], 0, 0]) for i in range(max_aa_harmonics[0] + 1, max_k_harmonics[0] + 1)],
            axis=0)
        k_arr = np.append(
            k_arr,
            [np.array([0, i / box_lengths[1], 0]) for i in range(max_aa_harmonics[1] + 1, max_k_harmonics[1] + 1)],
            axis=0)
        k_arr = np.append(
            k_arr,
            [np.array([0, 0, i / box_lengths[2]]) for i in range(max_aa_harmonics[2] + 1, max_k_harmonics[2] + 1)],
            axis=0)

    # Compute wave number magnitude - don't use |k| (skipping first entry in k_arr)
    k_mag = np.sqrt(np.sum(np.array(k_arr) ** 2, axis=1)[..., None])

    # Add magnitude to wave number array
    k_arr = np.concatenate((k_arr, k_mag), 1)

    # Sort from lowest to highest magnitude
    ind = np.argsort(k_arr[:, -1])
    k_arr = k_arr[ind]

    # Count how many times a |k| value appears
    k_unique, k_counts = np.unique(k_arr[first_non_zero:, -1], return_counts=True)

    # Generate a 1D array containing index to be used in S array
    k_index = np.repeat(range(len(k_counts)), k_counts)[..., None]

    # Add index to k_array
    k_arr = np.concatenate((k_arr[int(first_non_zero):, :], k_index), 1)
    return k_arr, k_counts, k_unique


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
    units_dict = UNITS[1] if units == 'cgs' else UNITS[0]

    if units == 'cgs' and xlbl == 'Length':
        xmax *= 1e2

    if units == 'cgs' and ylbl == 'Length':
        ymax *= 1e2

    # Use scientific notation. This returns a string
    x_str = np.format_float_scientific(xmax)
    y_str = np.format_float_scientific(ymax)

    # Grab the exponent
    x_exp = 10.0 ** (float(x_str[x_str.find('e') + 1:]))
    y_exp = 10.0 ** (float(y_str[y_str.find('e') + 1:]))

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
    ymul = - 1.5
    i = 1.0
    while ymul < 0:
        for key, value in PREFIXES.items():
            ratio = i * y_exp / value
            if abs(ratio - 1) < 1.0e-6:
                yprefix = key
                ymul = 1 / value
        i /= 10.

    if "Energy" in ylbl:
        yname = "Energy"
    else:
        yname = ylbl

    if "Pressure" in ylbl:
        yname = "Pressure"
    else:
        yname = ylbl

    if yname in units_dict:
        ylabel = ' [' + yprefix + units_dict[yname] + ']'
    else:
        ylabel = ''

    if "Energy" in xlbl:
        xname = "Energy"
    else:
        xname = xlbl

    if "Pressure" in xlbl:
        xname = "Pressure"
    else:
        xname = xlbl

    if xname in units_dict:
        xlabel = ' [' + xprefix + units_dict[xname] + ']'
    else:
        xlabel = ''

    return xmul, ymul, xprefix, yprefix, xlabel, ylabel


def correlationfunction(At, Bt):
    """
    Calculate the correlation function :math:`\mathbf{A}(t)` and :math:`\mathbf{B}(t)` using
    ``scipy.signal.correlate``

    .. math::
        C_{AB}(\\tau) =  \sum_j^D \sum_i^T A_j(t_i)B_j(t_i + \\tau)

    where :math:`D` (= ``no_dim``) is the number of dimensions and :math:`T` (= ``no_steps``) is the total length
    of the simulation.

    Parameters
    ----------
    At : numpy.ndarray
        Observable to correlate. Shape=(``no_steps``).

    Bt : numpy.ndarray
        Observable to correlate. Shape=(``no_steps``).

    Returns
    -------
    CF : numpy.ndarray
        Correlation function :math:`C_{AB}(\\tau)`
    """
    no_steps = At.size

    # Calculate the full correlation function.
    full_corr = scp_signal.correlate(At, Bt, mode='full')
    # Normalization of the full correlation function, Similar to norm_counter
    norm_corr = np.array([no_steps - ii for ii in range(no_steps)])
    # Find the mid point of the array
    mid = full_corr.size // 2
    # I want only the second half of the array, i.e. the positive lags only
    return full_corr[mid:] / norm_corr

# These are old functions that should not be trusted.
# @njit
# def autocorrelationfunction_slow(At):
#     """
#     Calculate the autocorrelation function of the array input.
#
#     .. math::
#         A(\\tau) =  \sum_j^D \sum_i^T A_j(t_i)A_j(t_i + \\tau)
#
#     where :math:`D` is the number of dimensions and :math:`T` is the total length
#     of the simulation.
#
#     Parameters
#     ----------
#     At : numpy.ndarray
#         Observable to autocorrelate. Shape=(``no_dim``, ``no_steps``).
#
#     Returns
#     -------
#     ACF : numpy.ndarray
#         Autocorrelation function of ``At``.
#     """
#     no_steps = At.shape[1]
#     no_dim = At.shape[0]
#
#     ACF = np.zeros(no_steps)
#     Norm_counter = np.zeros(no_steps)
#
#     for it in range(no_steps):
#         for dim in range(no_dim):
#             ACF[: no_steps - it] += At[dim, it] * At[dim, it:no_steps]
#         Norm_counter[: no_steps - it] += 1.0
#
#     return ACF / Norm_counter
#
#
# @njit
# def autocorrelationfunction_1D_slow(At):
#     """
#     Calculate the autocorrelation function of the input.
#
#     .. math::
#         A(\\tau) =  \sum_i^T A(t_i)A(t_i + \\tau)
#
#     where :math:`T` is the total length of the simulation.
#
#     Parameters
#     ----------
#     At : numpy.ndarray
#         Array to autocorrelate. Shape=(``no_steps``).
#
#     Returns
#     -------
#     ACF : numpy.ndarray
#         Autocorrelation function of ``At``.
#     """
#     no_steps = At.shape[0]
#     ACF = np.zeros(no_steps)
#     Norm_counter = np.zeros(no_steps)
#
#     for it in range(no_steps):
#         ACF[: no_steps - it] += At[it] * At[it:no_steps]
#         Norm_counter[: no_steps - it] += 1.0
#
#     return ACF / Norm_counter
#
#
# @jit
# def autocorrelationfunction(At):
#     """
#     Calculate the autocorrelation function of the array input.
#
#     .. math::
#         A(\\tau) =  \sum_j^D \sum_i^T A_j(t_i)A_j(t_i + \\tau)
#
#     where :math:`D` is the number of dimensions and :math:`T` is the total length
#     of the simulation.
#
#     Parameters
#     ----------
#     At : numpy.ndarray
#         Observable to autocorrelate. Shape=(``no_dim``, ``no_steps``).
#
#     Returns
#     -------
#     ACF : numpy.ndarray
#         Autocorrelation function of ``At``.
#     """
#     no_steps = At.shape[1]
#     no_dim = At.shape[0]
#
#     norm_counter = np.zeros(no_steps)
#     # Number of time origins for each time step
#     for it in range(no_steps):
#         norm_counter[: no_steps - it] += 1.0
#
#     ACF_total = np.zeros(no_steps)
#     for i in range(no_dim):
#         At2 = np.zeros(int(2 * no_steps), dtype=np.complex128)
#         At2[:no_steps] = At[i, :] + 1j * 0.0
#         # Prepare for FFTW
#         fftw_obj = pyfftw.builders.fft(At2)
#         # Calculate FFTW
#         Aw = fftw_obj()
#         # ACF in frequency space
#         Aw2 = np.conj(Aw) * Aw
#         # Prepare for IFFTW
#         ifftw_obj = pyfftw.builders.ifft(Aw2)
#         # IFFTW to get ACF
#         At2 = ifftw_obj()
#         # Normalization associated with FFTW
#         # At2 /= 2 * no_steps
#         ACF_total += np.real(At2[:no_steps]) / norm_counter
#
#     return ACF_total
#
#
# @jit
# def autocorrelationfunction_1D(At):
#     """
#     Calculate the autocorrelation function of the input using the FFT method.
#
#     .. math::
#         A(\\tau) =  \sum_i^T A(t_i)A(t_i + \\tau)
#
#     where :math:`T` is the total length of the simulation.
#
#     Parameters
#     ----------
#     At : numpy.ndarray
#         Array to autocorrelate. Shape=(``no_steps``).
#
#     Returns
#     -------
#     ACF : numpy.ndarray
#         Autocorrelation function of ``At``.
#
#     Notes
#     -----
#     This code is a reproduction of Allen-Tildesley
#     `code <https://github.com/Allen-Tildesley/examples/blob/master/python_examples/corfun.py>`_
#
#     """
#     no_steps = At.shape[0]
#     # Normalization. Number of time origins for each time step
#     norm_counter = np.zeros(no_steps)
#     for it in range(no_steps):
#         norm_counter[: no_steps - it] += 1.0
#
#     # Create the arrays for FFTW
#     At2 = pyfftw.empty_aligned(int(2 * no_steps), dtype=np.complex128)
#     Aw = pyfftw.empty_aligned(int(2 * no_steps), dtype=np.complex128)
#     # Append an array of zeros to the function A(t)
#     At2[:no_steps] = At + 1j * 0.0
#     # Prepare for FFTW
#     fftw_obj = pyfftw.builders.fft(At2, Aw)
#     # Calculate FFTW
#     Aw = fftw_obj()
#     # ACF in frequency space
#     Aw2 = np.conj(Aw) * Aw
#     # Prepare for IFFTW, Aw2 At2 is the out_array
#     ifftw_obj = pyfftw.builders.fft(Aw2, At2)
#     # IFFTW to get ACF
#     At2 = ifftw_obj()
#     return np.real(At2[:no_steps]) / norm_counter
#
#
# @njit

#
#
# @njit
# def correlationfunction_1D_slow(At, Bt):
#     """
#     Calculate the correlation function between :math:`A(t)` and :math:`B(t)`
#
#     .. math::
#         C_{AB}(\\tau) =  \sum_i^T A(t_i)B(t_i + \\tau)
#
#     where :math:`T` (= ``no_steps``) is the total length of the simulation.
#
#     Parameters
#     ----------
#     At : numpy.ndarray
#         Observable to correlate. Shape=(``no_steps``).
#
#     Bt : numpy.ndarray
#         Observable to correlate. Shape=(``no_steps``).
#
#     Returns
#     -------
#     CF : numpy.ndarray
#         Correlation function :math:`C_{AB}(\\tau)`
#     """
#     no_steps = At.shape[0]
#     CF = np.zeros(no_steps)
#     Norm_counter = np.zeros(no_steps)
#
#     for it in range(no_steps):
#         CF[: no_steps - it] += At[it] * Bt[it:no_steps]
#         Norm_counter[: no_steps - it] += 1.0
#
#     return CF / Norm_counter
#
#
# @jit
# def correlationfunction(At, Bt):
#     """
#     Calculate the autocorrelation function of the array input.
#
#     .. math::
#         A(\\tau) =  \sum_j^D \sum_i^T A_j(t_i)A_j(t_i + \\tau)
#
#     where :math:`D` is the number of dimensions and :math:`T` is the total length
#     of the simulation.
#
#     Parameters
#     ----------
#     At : numpy.ndarray
#         Observable to autocorrelate. Shape=(``no_dim``, ``no_steps``).
#
#     Returns
#     -------
#     ACF : numpy.ndarray
#         Autocorrelation function of ``At``.
#     """
#     no_steps = At.shape[1]
#     no_dim = At.shape[0]
#
#     norm_counter = np.zeros(no_steps)
#     # Number of time origins for each time step
#     for it in range(no_steps):
#         norm_counter[: no_steps - it] += 1.0
#
#     CF_total = np.zeros(no_steps)
#     for i in range(no_dim):
#         # Create a larger array for storing A(t)
#         At2 = np.zeros(int(2 * no_steps), dtype=np.complex128)
#         At2[:no_steps] = At[i, :] + 1j * 0.0
#         # Create a larger array for storing B(t)
#         Bt2 = np.zeros(int(2 * no_steps), dtype=np.complex128)
#         Bt2[:no_steps] = Bt[i, :] + 1j * 0.0
#         # Prepare for FFTW
#         A_fftw_obj = pyfftw.builders.fft(At2)
#         # Calculate FFTW
#         Aw = A_fftw_obj()
#         # Prepare for B FFTW
#         B_fftw_obj = pyfftw.builders.fft(Bt2)
#         # Calculate FFTW
#         Bw = B_fftw_obj()
#         # ACF in frequency space
#         CFw2 = np.conj(Aw) * Bw
#         # Prepare for IFFTW
#         ifftw_obj = pyfftw.builders.ifft(CFw2)
#         # IFFTW to get ACF
#         CF_t = ifftw_obj()
#
#         # Normalization associated with FFTW
#         # At2 /= 2 * no_steps
#         CF_total += np.real(CF_t[:no_steps]) / norm_counter
#
#     return CF_total
#
#
# @jit
# def correlationfunction_1D(At, Bt):
#     """
#     Calculate the autocorrelation function of the input.
#
#     .. math::
#         A(\\tau) =  \sum_i^T A(t_i)A(t_i + \\tau)
#
#     where :math:`T` is the total length of the simulation.
#
#     Parameters
#     ----------
#     At : numpy.ndarray
#         Array to autocorrelate. Shape=(``no_steps``).
#
#     Returns
#     -------
#     ACF : numpy.ndarray
#         Autocorrelation function of ``At``.
#     """
#     no_steps = At.shape[0]
#     norm_counter = np.zeros(no_steps)
#     # Create a larger array for storing A(t)
#     At2 = np.zeros(int(2 * no_steps), dtype=np.complex128)
#     At2[:no_steps] = At + 1j * 0.0
#     # Create a larger array for storing B(t)
#     Bt2 = np.zeros(int(2 * no_steps), dtype=np.complex128)
#     Bt2[:no_steps] = Bt + 1j * 0.0
#     # Prepare for FFTW
#     A_fftw_obj = pyfftw.builders.fft(At2)
#     # Calculate FFTW
#     Aw = A_fftw_obj()
#     # Prepare for B FFTW
#     B_fftw_obj = pyfftw.builders.fft(Bt2)
#     # Calculate FFTW
#     Bw = B_fftw_obj()
#
#     # ACF in frequency space
#     ACFw2 = np.conj(Aw) * Bw
#     # Prepare for IFFTW
#     ifftw_obj = pyfftw.builders.ifft(ACFw2)
#     # IFFTW to get ACF
#     CF_t = ifftw_obj()
#     # Normalization associated with FFTW
#     # At2 /= 2 * no_steps
#     # Number of time origins for each time step
#     for it in range(no_steps):
#         norm_counter[: no_steps - it] += 1.0
#
#     return np.real(CF_t[:no_steps]) / norm_counter
