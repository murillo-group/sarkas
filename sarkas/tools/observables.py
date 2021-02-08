"""
Module for calculating physical quantities from Sarkas checkpoints.
"""
from IPython import get_ipython

if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import os
import numpy as np
from numba import njit
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

UNITS = [
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
     "Conductivity": "mho/cm",
     "Diffusion": r"cm$^2$/s",
     "Viscosity": r"Ba s",
     "none": ""}
]

PREFIXES = {
    "Y": 1e24,
    "Z": 1e21,
    "E": 1e18,
    "P": 1e15,
    "T": 1e12,
    "G": 1e9,
    "M": 1e6,
    "k": 1e3,
    "": 1e0,
    "c": 1.0e-2,
    "m": 1.0e-3,
    r"$\mu$": 1.0e-6,
    "n": 1.0e-9,
    "p": 1.0e-12,
    "f": 1.0e-15,
    "a": 1.0e-18,
    "z": 1.0e-21,
    "y": 1.0e-24
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

    no_ka_harmonics : list
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

    species : list
        List of ``sarkas.base.Species`` indicating simulation's species.

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

    """

    def __init__(self):
        self.dataframe = pd.DataFrame()
        self.dataframe_longitudinal = pd.DataFrame()
        self.dataframe_transverse = pd.DataFrame()
        self.saving_dir = None
        self.filename_csv = None
        self.filename_csv_longitudinal = None
        self.filename_csv_transverse = None
        self.phase = None

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
        self.__dict__.update(params.__dict__)

        # Create the lists of k vectors
        if hasattr(self, 'no_ka_harmonics'):
            if isinstance(self.no_ka_harmonics, np.ndarray) == 0:
                self.no_ka_harmonics = np.ones(3, dtype=int) * self.no_ka_harmonics
        else:
            self.no_ka_harmonics = [5, 5, 5]

        if not hasattr(self, 'all_k_values'):
            self.all_k_values = False

        self.k_space_dir = os.path.join(self.postprocessing_dir, "k_space_data")
        self.k_file = os.path.join(self.k_space_dir, "k_arrays.npz")
        self.nkt_file = os.path.join(self.k_space_dir, "nkt")
        self.vkt_file = os.path.join(self.k_space_dir, "vkt")

        self.no_obs = int(self.num_species * (self.num_species + 1) / 2)
        self.prod_no_dumps = len(os.listdir(self.prod_dump_dir))
        self.eq_no_dumps = len(os.listdir(self.eq_dump_dir))
        if self.magnetized and self.electrostatic_equilibration:
            self.mag_no_dumps = len(os.listdir(self.mag_dump_dir))

        if phase == 'equilibration':
            self.no_dumps = self.eq_no_dumps
            self.dump_dir = self.eq_dump_dir
            self.dump_step = self.eq_dump_step
            self.no_steps = self.equilibration_steps
        elif self.phase == 'production':
            self.no_dumps = self.prod_no_dumps
            self.dump_dir = self.prod_dump_dir
            self.dump_step = self.prod_dump_step
            self.no_steps = self.production_steps
        elif self.phase == 'magnetization':

            self.no_dumps = self.mag_no_dumps
            self.dump_dir = self.mag_dump_dir
            self.dump_step = self.mag_dump_step
            self.no_steps = self.magnetization_steps

        if not hasattr(self, 'no_slices'):
            self.no_slices = 1
        self.slice_steps = int(self.no_dumps / self.no_slices)

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

        elif self.__class__.__name__[-15:] == 'StructureFactor':
            try:
                self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
                k_data = np.load(self.k_file)
                self.k_list = k_data["k_list"]
                self.k_counts = k_data["k_counts"]
                self.ka_values = k_data["ka_values"]

            except FileNotFoundError:
                print("\nFile {} not found!".format(self.filename_csv))
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
            if self.all_k_values == k_data["all_k_values"]:
                # Check for the correct max harmonics
                comp = self.no_ka_harmonics == k_data["max_harmonics"]
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

        self.k_list, self.k_counts, k_unique = kspace_setup(self.no_ka_harmonics, self.box_lengths,
                                                            self.all_k_values)
        self.ka_values = 2.0 * np.pi * k_unique * self.a_ws
        self.no_ka_values = len(self.ka_values)

        if not (os.path.exists(self.k_space_dir)):
            os.mkdir(self.k_space_dir)

        np.savez(self.k_file,
                 k_list=self.k_list,
                 k_counts=self.k_counts,
                 ka_values=self.ka_values,
                 max_harmonics=self.no_ka_harmonics,
                 all_k_values=self.all_k_values)

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
                if self.all_k_values == nkt_data["all_k_values"]:
                    # Check for the correct max harmonics
                    comp = self.no_ka_harmonics == nkt_data["max_harmonics"]
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
                if self.all_k_values == vkt_data["all_k_values"]:
                    # Check for the correct max harmonics
                    comp = self.no_ka_harmonics == vkt_data["max_harmonics"]
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
                print("Calculating n(k,t) for slice {}/{}.".format(isl, self.no_slices))
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
                         max_harmonics=self.no_ka_harmonics,
                         all_k_values=self.all_k_values)
        if vkt_flag:
            for isl in range(self.no_slices):
                print("Calculating longitudinal and transverse "
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
                         max_harmonics=self.no_ka_harmonics,
                         all_k_values=self.all_k_values)

    def plot(self, normalization=None, figname=None, show=False, acf=False, longitudinal=True, **kwargs):
        """
        Plot the observable by calling the pandas.DataFrame.plot() function and save the figure.

        Parameters
        ----------
        longitudinal : bool
            Flag for longitudinal plot in case of CurrenCurrelationFunction

        acf : bool
            Flag for renormalizing the autocorrelation functions. Default= False

        figname : str
            Name with which to save the file. It automaticall saves it in the correct directory.

        normalization: float
            Factor by which to divide the distance array.

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
        self.parse()
        # Make a copy of the dataframe for plotting
        if self.__class__.__name__ != 'CurrentCorrelationFunction':
            plot_dataframe = self.dataframe.copy()
        elif longitudinal:
            plot_dataframe = self.dataframe_longitudinal.copy()
        else:
            plot_dataframe = self.dataframe_transverse.copy()

        if normalization:
            plot_dataframe.iloc[:, 0] /= normalization

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

    def setup(self, params, phase=None):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to analyze.

        params : sarkas.base.Parameters
            Simulation's parameters.

        species : list
            List of sarkas.base.Species.

        """

        self.phase = phase if phase else 'production'

        super().setup_init(params, self.phase)

        saving_dir = os.path.join(self.postprocessing_dir, 'CurrentCorrelationFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv_longitudinal = os.path.join(self.saving_dir,
                                                      "LongitudinalCurrentCorrelationFunction_" + self.job_id + '.csv')
        self.filename_csv_transverse = os.path.join(self.saving_dir,
                                                    "TransverseCurrentCorrelationFunction_" + self.job_id + '.csv')

    def compute(self):
        """
        Calculate the velocity fluctuations correlation functions.
        """

        # Parse vkt otherwise calculate them
        self.parse_k_data()
        self.parse_kt_data(nkt_flag=False, vkt_flag=True)
        # Initialize dataframes and add frequencies to it.
        # This re-initialization of the dataframe is needed to avoid len mismatch conflicts when re-calculating
        self.dataframe = pd.DataFrame()
        frequencies = 2.0 * np.pi * np.fft.fftfreq(self.slice_steps, self.dt * self.dump_step)
        self.dataframe_longitudinal["Frequencies"] = np.fft.fftshift(frequencies)
        self.dataframe_transverse["Frequencies"] = np.fft.fftshift(frequencies)

        temp_dataframe_longitudinal = pd.DataFrame()
        temp_dataframe_longitudinal["Frequencies"] = np.fft.fftshift(frequencies)

        temp_dataframe_transverse = pd.DataFrame()
        temp_dataframe_transverse["Frequencies"] = np.fft.fftshift(frequencies)

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


class DynamicStructureFactor(Observable):
    """
    Dynamic Structure factor.

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

    def setup(self, params, phase=None):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : (optional), str
            Phase to analyze.

        params : sarkas.base.Parameters
            Simulation's parameters.

        species : list
            List of ``sarkas.base.Species``.

        """

        self.phase = phase if phase else 'production'
        super().setup_init(params, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'DynamicStructureFactor')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "DynamicStructureFactor_" + self.job_id + '.csv')

    def compute(self):
        """
        Compute :math:`S_{ij} (k,\\omega)` and the array of :math:`\\omega` values.
        ``self.Skw``. Shape = (``no_ws``, ``no_Sij``)
        """

        # Parse nkt otherwise calculate it
        self.parse_k_data()
        self.parse_kt_data(nkt_flag=True)
        # This re-initialization of the dataframe is needed to avoid len mismatch conflicts when re-calculating
        self.dataframe = pd.DataFrame()

        frequencies = 2.0 * np.pi * np.fft.fftfreq(self.slice_steps, self.dt * self.dump_step)
        self.dataframe["Frequencies"] = np.fft.fftshift(frequencies)

        temp_dataframe = pd.DataFrame()
        temp_dataframe["Frequencies"] = np.fft.fftshift(frequencies)

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


class ElectricCurrent(Observable):
    """
    Electric Current Auto-correlation function.

    Attributes
    ----------
    dataframe : pandas.DataFrame
        Dataframe of the longitudinal velocity correlation functions.

    saving_dir : str
        Path to directory where computed data is stored.

    filename_csv : str
        Name of file for the longitudinal velocities fluctuation correlation function.

    """

    def setup(self, params, phase=None):
        """
        Initialize the attributes from simulation's parameters.

        Parameters
        ----------
        phase : (optional), str
            Phase to analyze.

        params: sarkas.base.Parameters
            Simulation's parameters.

        species : list
            List of ``sarkas.base.Species``.

        """
        self.phase = phase if phase else 'production'

        super().setup_init(params, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'ElectricCurrent')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "ElectricCurrent_" + self.job_id + '.csv')

    def compute(self):
        """
        Compute the electric current and the corresponding auto-correlation functions.
        """

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

        cur_acf_xx = autocorrelationfunction_1D(total_current[0, :])
        cur_acf_yy = autocorrelationfunction_1D(total_current[1, :])
        cur_acf_zz = autocorrelationfunction_1D(total_current[2, :])

        tot_cur_acf = autocorrelationfunction(total_current)
        # Normalize and save
        self.dataframe["X Current ACF"] = cur_acf_xx
        self.dataframe["Y Current ACF"] = cur_acf_yy
        self.dataframe["Z Current ACF"] = cur_acf_zz
        self.dataframe["Total Current ACF"] = tot_cur_acf
        for i, sp in enumerate(self.species_names):
            tot_acf = autocorrelationfunction(species_current[i, :, :])
            acf_xx = autocorrelationfunction_1D(species_current[i, 0, :])
            acf_yy = autocorrelationfunction_1D(species_current[i, 1, :])
            acf_zz = autocorrelationfunction_1D(species_current[i, 2, :])

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


class HermiteCoefficients(Observable):
    """
    Hermite coefficients of the Hermite expansion.

    Attributes
    ----------
    hermite_order: int
        Order of the Hermite expansion.

    no_bins: int
        Number of bins used to calculate the velocity distribution.

    plots_dir: str
        Directory in which to store Hermite coefficients plots.

    species_plots_dirs : list, str
        Directory for each species where to save Hermite coefficients plots.

    """

    def setup(self, params, phase=None):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : (optional), str
            Phase to analyze.

        params : sarkas.base.Parameters
            Simulation's parameters.

        """

        self.phase = phase if phase else 'equilibration'

        super().setup_init(params, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'HermiteExpansion')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)
        self.filename_csv = os.path.join(self.saving_dir, "HermiteCoefficients_" + self.job_id + '.csv')

        self.plots_dir = os.path.join(self.saving_dir, 'Hermite_Plots')
        if not os.path.exists(self.plots_dir):
            os.mkdir(self.plots_dir)

        # Check that the existence of important attributes
        if not hasattr(self, 'no_bins'):
            self.no_bins = int(0.05 * params.total_num_ptcls)

        if not hasattr(self, 'hermite_order'):
            self.hermite_order = 7

        self.species_plots_dirs = None

    def compute(self):
        """
        Calculate Hermite coefficients and save the pandas dataframe.
        """
        vscale = 1.0 / (self.a_ws * self.total_plasma_frequency)
        vel = np.zeros((self.dimensions, self.total_num_ptcls))

        xcoeff = np.zeros((self.num_species, self.no_dumps, self.hermite_order + 1))
        ycoeff = np.zeros((self.num_species, self.no_dumps, self.hermite_order + 1))
        zcoeff = np.zeros((self.num_species, self.no_dumps, self.hermite_order + 1))

        time = np.zeros(self.no_dumps)
        print("Computing Hermite Coefficients ...")
        for it in range(self.no_dumps):
            time[it] = it * self.dt * self.dump_step
            dump = int(it * self.dump_step)
            datap = load_from_restart(self.dump_dir, dump)
            vel[0, :] = datap["vel"][:, 0]
            vel[1, :] = datap["vel"][:, 1]
            vel[2, :] = datap["vel"][:, 2]

            sp_start = 0
            sp_end = 0
            for i, sp in enumerate(self.species_num):
                sp_end += sp
                x_hist, xbins = np.histogram(vel[0, sp_start:sp_end] * vscale, bins=self.no_bins, density=True)
                y_hist, ybins = np.histogram(vel[1, sp_start:sp_end] * vscale, bins=self.no_bins, density=True)
                z_hist, zbins = np.histogram(vel[2, sp_start:sp_end] * vscale, bins=self.no_bins, density=True)

                # Center the bins
                vx = 0.5 * (xbins[:-1] + xbins[1:])
                vy = 0.5 * (ybins[:-1] + ybins[1:])
                vz = 0.5 * (zbins[:-1] + zbins[1:])

                xcoeff[i, it, :] = calculate_herm_coeff(vx, x_hist, self.hermite_order)
                ycoeff[i, it, :] = calculate_herm_coeff(vy, y_hist, self.hermite_order)
                zcoeff[i, it, :] = calculate_herm_coeff(vz, z_hist, self.hermite_order)

                sp_start = sp_end

        data = {"Time": time}
        self.dataframe = pd.DataFrame(data)
        for i, sp in enumerate(self.species_names):
            for hi in range(self.hermite_order + 1):
                self.dataframe["{} Hermite x Coeff a{}".format(sp, hi)] = xcoeff[i, :, hi]
                self.dataframe["{} Hermite y Coeff a{}".format(sp, hi)] = ycoeff[i, :, hi]
                self.dataframe["{} Hermite z Coeff a{}".format(sp, hi)] = zcoeff[i, :, hi]

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def plot(self, show=False):
        """
        Plot the Hermite coefficients and save the figure
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            self.compute()

        if not os.path.exists(self.plots_dir):
            os.mkdir(self.plots_dir)

        # Create a plots directory for each species for the sake of neatness
        if self.num_species > 1:
            self.species_plots_dirs = []
            for i, name in enumerate(self.species_names):
                new_dir = os.path.join(self.plots_dir, "{}".format(name))
                self.species_plots_dirs.append(new_dir)
                if not os.path.exists(new_dir):
                    os.mkdir(os.path.join(self.plots_dir, "{}".format(name)))
        else:
            self.species_plots_dirs = [self.plots_dir]

        for sp, name in enumerate(self.species_names):
            print("Species: {}".format(name))
            fig, ax = plt.subplots(1, 2, sharex=True, constrained_layout=True, figsize=(16, 9))
            for indx in range(self.hermite_order):
                xcolumn = "{} Hermite x Coeff a{}".format(name, indx)
                ycolumn = "{} Hermite y Coeff a{}".format(name, indx)
                zcolumn = "{} Hermite z Coeff a{}".format(name, indx)
                xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(self.dataframe["Time"], 1.0,
                                                                       'Time', 'none', self.units)
                ia = int(indx % 2)
                ax[ia].plot(self.dataframe["Time"] * xmul, self.dataframe[xcolumn] + ia * (indx - 1),
                            ls='-', label=r"$a_{" + str(indx) + " , x}$")
                ax[ia].plot(self.dataframe["Time"] * xmul, self.dataframe[ycolumn] + ia * (indx - 1),
                            ls='--', label=r"$a_{" + str(indx) + " , y}$")
                ax[ia].plot(self.dataframe["Time"] * xmul, self.dataframe[zcolumn] + ia * (indx - 1),
                            ls='-.', label=r"$a_{" + str(indx) + " , z}$")

            ax[0].set_title(r'Even Coefficients')
            ax[1].set_title(r'Odd Coefficients')

            ax[0].set_xlabel(r'$t$' + xlbl)
            ax[1].set_xlabel(r'$t$' + xlbl)

            sigma = np.sqrt(self.kB * self.species_temperatures[sp] / self.species_masses[sp])
            sigma /= (self.a_ws * self.total_plasma_frequency)  # Rescale

            for i in range(0, self.hermite_order, 2):
                coeff = np.zeros(i + 1)
                coeff[-1] = 1.0
                print("Equilibrium a{} = {:1.2f} ".format(i, np.polynomial.hermite_e.hermeval(sigma, coeff)))

            # t_end = self.dataframe["Time"].iloc[-1] * xmul/2
            # ax[0].text(t_end, 1.1, r"$a_{0,\rm{eq}} = 1 $", transform=ax[0].transData)
            # # ax[0].axhline(1, ls=':', c='k', label=r"$a_{0,\rm{eq}}$")
            #
            # ax[0].text(t_end, a2_eq * 0.97, r"$a_{2,\rm{eq}} = " + "{:1.2f}".format(a2_eq) +"$",
            #            transform=ax[0].transData)
            #
            # if self.hermite_order > 3:
            #     ax[0].text(t_end, a4_eq * 1.1, r"$a_{4,\rm{eq}} = " + "{:1.2f}".format(a4_eq) + "$",
            #                transform=ax[0].transData)
            #
            # if self.hermite_order > 5:
            #     ax[0].text(t_end, a6_eq * .98, r"$a_{6,\rm{eq}} = " + "{:1.2f}".format(a6_eq) + "$",
            #                transform=ax[0].transData)

            ax[0].legend(loc='best', ncol=int(self.hermite_order / 2 + self.hermite_order % 2))
            ax[1].legend(loc='best', ncol=int(self.hermite_order / 2))
            yt = np.arange(0, self.hermite_order + self.hermite_order % 2, 2)
            ax[1].set_yticks(yt)
            ax[1].set_yticklabels(np.zeros(len(yt)))
            fig.suptitle("Hermite Coefficients of {}".format(name) + '  Phase: ' + self.phase.capitalize())
            plot_name = os.path.join(self.species_plots_dirs[sp], '{}_HermCoeffPlot_'.format(name)
                                     + self.job_id + '.png')
            fig.savefig(plot_name)
            if show:
                fig.show()


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

    def setup(self, params, phase=None):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : (optional), str
            Phase to analyze.

        params : sarkas.base.Parameters
            Simulation's parameters.

        """
        self.phase = phase if phase else 'production'

        super().setup_init(params, self.phase)

        saving_dir = os.path.join(self.postprocessing_dir, 'RadialDistributionFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir,
                                         "RadialDistributionFunction_" + self.job_id + ".csv")
        self.rc = self.cutoff_radius

    def compute(self, rdf_hist=None):
        """
        Parameters
        ----------
        rdf_hist : numpy.ndarray
            Histogram of the radial distribution function.

        """
        if not isinstance(rdf_hist, np.ndarray):
            # Find the last dump by looking for the longest filename in
            dumps_list = os.listdir(self.dump_dir)
            last = 0
            for file in dumps_list:
                name, ext = os.path.splitext(file)
                _, number = name.split('_')
                if int(number) > last:
                    last = int(number)
            data = load_from_restart(self.dump_dir, int(last))
            rdf_hist = data["rdf_hist"]

        self.no_bins = rdf_hist.shape[0]
        self.dr_rdf = self.rc / self.no_bins

        r_values = np.zeros(self.no_bins)
        bin_vol = np.zeros(self.no_bins)
        pair_density = np.zeros((self.num_species, self.num_species))
        gr = np.zeros((self.no_bins, self.no_obs))
        # No. of pairs per volume
        for i, sp1 in enumerate(self.species_num):
            pair_density[i, i] = 0.5 * sp1 * (sp1 - 1) / self.box_volume
            if self.num_species > 1:
                for j, sp2 in enumerate(self.species_num[i:], i):
                    pair_density[i, j] = sp1 * sp2 / self.box_volume
        # Calculate each bin's volume
        sphere_shell_const = 4.0 * np.pi / 3.0
        bin_vol[0] = sphere_shell_const * self.dr_rdf ** 3
        for ir in range(1, self.no_bins):
            r1 = ir * self.dr_rdf
            r2 = (ir + 1) * self.dr_rdf
            bin_vol[ir] = sphere_shell_const * (r2 ** 3 - r1 ** 3)
            r_values[ir] = (ir + 0.5) * self.dr_rdf

        self.dataframe["distance"] = r_values
        gr_ij = 0
        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:], i):
                denom_const = (pair_density[i, j] * self.production_steps)
                gr[:, gr_ij] = (rdf_hist[:, i, j] + rdf_hist[:, j, i]) / denom_const / bin_vol[:]

                self.dataframe['{}-{} RDF'.format(sp1, sp2)] = gr[:, gr_ij]

                gr_ij += 1
        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')


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

    def setup(self, params, phase=None):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : (optional), str
            Phase to analyze.

        params : sarkas.base.Parameters
            Simulation's parameters.

        """

        self.phase = phase if phase else 'production'
        super().setup_init(params, self.phase)

        saving_dir = os.path.join(self.postprocessing_dir, 'StaticStructureFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "StaticStructureFunction_" + self.job_id + ".csv")

    def compute(self):
        """
        Calculate all :math:`S_{ij}(k)`, save them into a Pandas dataframe, and write them to a csv.
        """
        # Parse nkt otherwise calculate it
        self.parse_k_data()
        self.parse_kt_data(nkt_flag=True)
        # This re-initialization of the dataframe is needed to avoid len mismatch conflicts when re-calculating
        self.dataframe = pd.DataFrame()
        self.dataframe["ka values"] = self.ka_values

        no_dumps_calculated = self.slice_steps * self.no_slices
        Sk_all = np.zeros((self.no_obs, len(self.k_counts), no_dumps_calculated))

        print("Calculating S(k)")

        for isl in tqdm(range(self.no_slices)):
            nkt_data = np.load(self.nkt_file + '_slice_' + str(isl) + '.npz')
            nkt = nkt_data["nkt"]
            init = isl * self.slice_steps
            fin = (isl + 1) * self.slice_steps
            Sk_all[:, :, init:fin] = calc_Sk(nkt, self.k_list, self.k_counts, self.species_num, self.slice_steps)

        Sk = np.mean(Sk_all, axis=-1)
        Sk_err = np.std(Sk_all, axis=-1)

        sp_indx = 0
        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:]):
                column = "{}-{} SSF".format(sp1, sp2)
                err_column = "{}-{} SSF Errorbar".format(sp1, sp2)
                self.dataframe[column] = Sk[sp_indx, :]
                self.dataframe[err_column] = Sk_err[sp_indx, :]

                sp_indx += 1

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')


class Thermodynamics(Observable):
    """
    Thermodynamic functions.
    """

    def setup(self, params, phase=None):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : (optional), str
            Phase to analyze.

        params : sarkas.base.Parameters
            Simulation's parameters.

        """
        if not hasattr(self, 'phase'):
            self.phase = phase.lower() if phase else 'production'

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

    def compute_pressure_quantities(self):
        """
        Calculate Pressure, Pressure Tensor, Pressure Tensor Auto Correlation Function.
        """
        self.parse('production')
        pos = np.zeros((self.dimensions, self.total_num_ptcls))
        vel = np.zeros((self.dimensions, self.total_num_ptcls))
        acc = np.zeros((self.dimensions, self.total_num_ptcls))

        pressure = np.zeros(self.no_dumps)
        pressure_tensor_temp = np.zeros((3, 3, self.no_dumps))

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
        self.dataframe["Pressure ACF"] = autocorrelationfunction_1D(pressure)

        if self.dimensions == 3:
            dim_lbl = ['x', 'y', 'z']

        # Calculate the acf of the pressure tensor
        for i, ax1 in enumerate(dim_lbl):
            for j, ax2 in enumerate(dim_lbl):
                self.dataframe["Pressure Tensor {}{}".format(ax1, ax2)] = pressure_tensor_temp[i, j, :]
                pressure_tensor_acf_temp = autocorrelationfunction_1D(pressure_tensor_temp[i, j, :])
                self.dataframe["Pressure Tensor ACF {}{}".format(ax1, ax2)] = pressure_tensor_acf_temp

        # Save the pressure acf to file
        self.dataframe.to_csv(self.prod_energy_filename, index=False, encoding='utf-8')

    def compute_pressure_from_rdf(self, r, gr, potential, potential_matrix):
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

        Returns
        -------
        pressure : float
            Pressure divided by :math:`k_BT`.

        """
        r *= self.a_ws
        r2 = r * r
        r3 = r2 * r

        if potential == "Coulomb":
            dv_dr = - 1.0 / r2
            # Check for finiteness of first element when r[0] = 0.0
            if not np.isfinite(dv_dr[0]):
                dv_dr[0] = dv_dr[1]
        elif potential == "Yukawa":
            pass
        elif potential == "QSP":
            pass
        else:
            raise ValueError('Unknown potential')

        # No. of independent g(r)
        T = np.mean(self.dataframe["Temperature"])
        pressure = self.kB * T - 2.0 / 3.0 * np.pi * self.species_dens[0] \
                   * potential_matrix[1, 0, 0] * np.trapz(dv_dr * r3 * gr, x=r)
        pressure *= self.species_dens[0]

        return pressure

    # def plot(self, quantity="Total Energy", phase=None, show=False):
    #     """
    #     Plot ``quantity`` vs time and save the figure with appropriate name.
    #
    #     Parameters
    #     ----------
    #     phase
    #     show : bool
    #         Flag for displaying figure.
    #
    #     quantity : str
    #         Quantity to plot. Default = Total Energy.
    #     """
    #
    #     if phase:
    #         self.phase = phase
    #         self.parse(phase)
    #
    #     if quantity[:8] == "Pressure":
    #         if not "Pressure" in self.dataframe.columns:
    #             print("Calculating Pressure quantities ...")
    #             self.compute_pressure_quantities()
    #
    #     xmul, ymul, xpref, ypref, xlbl, ylbl = plot_labels(self.dataframe["Time"],
    #                                                        self.dataframe[quantity],
    #                                                        "Time", quantity, self.units)
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    #     yq = {"Total Energy": r"$E_{tot}(t)$", "Kinetic Energy": r"$K_{tot}(t)$", "Potential Energy": r"$U_{tot}(t)$",
    #           "Temperature": r"$T(t)$",
    #           "Pressure Tensor ACF": r'$P_{\alpha\beta} = \langle P_{\alpha\beta}(0)P_{\alpha\beta}(t)\rangle$',
    #           "Pressure Tensor": r"$P_{\alpha\beta}(t)$", "Gamma": r"$\Gamma(t)$", "Pressure": r"$P(t)$"}
    #     dim_lbl = ['x', 'y', 'z']
    #
    #     if quantity == "Pressure Tensor ACF":
    #         for i, dim1 in enumerate(dim_lbl):
    #             for j, dim2 in enumerate(dim_lbl):
    #                 ax.plot(self.dataframe["Time"] * xmul,
    #                         self.dataframe["Pressure Tensor ACF {}{}".format(dim1, dim2)] /
    #                         self.dataframe["Pressure Tensor ACF {}{}".format(dim1, dim2)][0],
    #                         label=r'$P_{' + dim1 + dim2 + '} (t)$')
    #         ax.set_xscale('log')
    #         ax.legend(loc='best', ncol=3)
    #         ax.set_ylim(-1, 1.5)
    #
    #     elif quantity == "Pressure Tensor":
    #         for i, dim1 in enumerate(dim_lbl):
    #             for j, dim2 in enumerate(dim_lbl):
    #                 ax.plot(self.dataframe["Time"] * xmul,
    #                         self.dataframe["Pressure Tensor {}{}".format(dim1, dim2)] * ymul,
    #                         label=r'$P_{' + dim1 + dim2 + '} (t)$')
    #         ax.set_xscale('log')
    #         ax.legend(loc='best', ncol=3)
    #
    #     elif quantity == 'Temperature' and self.num_species > 1:
    #         for sp in self.species_names:
    #             qstr = "{} Temperature".format(sp)
    #             ax.plot(self.dataframe["Time"] * xmul, self.dataframe[qstr] * ymul, label=qstr)
    #         ax.plot(self.dataframe["Time"] * xmul, self.dataframe["Temperature"] * ymul, label='Total Temperature')
    #         ax.legend(loc='best')
    #     else:
    #         ax.plot(self.dataframe["Time"] * xmul, self.dataframe[quantity] * ymul)
    #
    #     ax.grid(True, alpha=0.3)
    #     ax.set_ylabel(yq[quantity] + ylbl)
    #     ax.set_xlabel(r'Time' + xlbl)
    #     fig.tight_layout()
    #     fig.savefig(os.path.join(self.fldr, quantity + '_' + self.job_id + '.png'))
    #     if show:
    #         fig.show()

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

    def temp_energy_plot(self, simulation, phase=None, show=False):
        """
        Plot Temperature and Energy as a function of time with their cumulative sum and average.

        Parameters
        ----------
        simulation : sarkas.processes.PostProcess
            Wrapper class.

        phase: str
            Phase to plot. "equilibration" or "production".

        show: bool
            Flag for displaying the figure.

        """

        if phase:
            phase = phase.lower()
            self.phase = phase
            if self.phase == 'equilibration':
                self.no_dumps = self.eq_no_dumps
                self.dump_dir = self.eq_dump_dir
                self.dump_step = self.eq_dump_step
                self.fldr = self.equilibration_dir
                self.no_steps = self.equilibration_steps
                self.parse(self.phase)
                self.dataframe = self.dataframe.iloc[1:, :]

            elif self.phase == 'production':
                self.no_dumps = self.prod_no_dumps
                self.dump_dir = self.prod_dump_dir
                self.dump_step = self.prod_dump_step
                self.fldr = self.production_dir
                self.no_steps = self.production_steps
                self.parse(self.phase)

            elif self.phase == 'magnetization':
                self.no_dumps = self.mag_no_dumps
                self.dump_dir = self.mag_dump_dir
                self.dump_step = self.mag_dump_step
                self.fldr = self.magnetization_dir
                self.no_steps = self.magnetization_steps
                self.parse(self.phase)

        else:
            self.parse()

        completed_steps = self.dump_step * (self.no_dumps - 1)

        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(4, 8)
        fsz = 16

        nbins = int(0.05 * self.no_dumps)

        Info_plot = fig.add_subplot(gs[0:4, 0:2])

        T_hist_plot = fig.add_subplot(gs[1:4, 2])
        T_delta_plot = fig.add_subplot(gs[0, 3:5])
        T_main_plot = fig.add_subplot(gs[1:4, 3:5])

        E_delta_plot = fig.add_subplot(gs[0, 5:7])
        E_main_plot = fig.add_subplot(gs[1:4, 5:7])
        E_hist_plot = fig.add_subplot(gs[1:4, 7])

        # Temperature plots
        xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(self.dataframe["Time"],
                                                               self.dataframe["Temperature"], "Time",
                                                               "Temperature", self.units)
        T_cumavg = self.dataframe["Temperature"].expanding().mean()

        T_main_plot.plot(xmul * self.dataframe["Time"], ymul * self.dataframe["Temperature"], alpha=0.7)
        T_main_plot.plot(xmul * self.dataframe["Time"], ymul * T_cumavg, label='Cum Avg')
        T_main_plot.axhline(ymul * self.T_desired, ls='--', c='r', alpha=0.7, label='Desired T')

        Delta_T = (self.dataframe["Temperature"] - self.T_desired) * 100 / self.T_desired
        Delta_T_cum_avg = Delta_T.expanding().mean()
        T_delta_plot.plot(self.dataframe["Time"] * xmul, Delta_T, alpha=0.5)
        T_delta_plot.plot(self.dataframe["Time"] * xmul, Delta_T_cum_avg, alpha=0.8)

        T_delta_plot.get_xaxis().set_ticks([])
        T_delta_plot.set_ylabel(r'Deviation [%]')
        T_delta_plot.tick_params(labelsize=fsz - 2)
        T_main_plot.tick_params(labelsize=fsz)
        T_main_plot.legend(loc='best')
        T_main_plot.set_ylabel("Temperature" + ylbl, fontsize=fsz)
        T_main_plot.set_xlabel("Time" + xlbl, fontsize=fsz)
        T_hist_plot.hist(self.dataframe['Temperature'], bins=nbins, density=True, orientation='horizontal',
                         alpha=0.75)
        T_hist_plot.get_xaxis().set_ticks([])
        T_hist_plot.get_yaxis().set_ticks([])
        T_hist_plot.set_xlim(T_hist_plot.get_xlim()[::-1])

        # Energy plots
        xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(self.dataframe["Time"],
                                                               self.dataframe["Total Energy"],
                                                               "Time",
                                                               "Energy",
                                                               self.units)

        E_cumavg = self.dataframe["Total Energy"].expanding().mean()

        E_main_plot.plot(xmul * self.dataframe["Time"], ymul * self.dataframe["Total Energy"], alpha=0.7)
        E_main_plot.plot(xmul * self.dataframe["Time"], ymul * E_cumavg, label='Cum Avg')
        E_main_plot.axhline(ymul * self.dataframe["Total Energy"].mean(), ls='--', c='r', alpha=0.7, label='Avg')

        Delta_E = (self.dataframe["Total Energy"] - self.dataframe["Total Energy"].iloc[0]) * 100 / \
                  self.dataframe["Total Energy"].iloc[0]
        Delta_E_cum_avg = Delta_E.expanding().mean()

        E_delta_plot.plot(self.dataframe["Time"] * xmul, Delta_E, alpha=0.5)
        E_delta_plot.plot(self.dataframe["Time"] * xmul, Delta_E_cum_avg, alpha=0.8)
        E_delta_plot.get_xaxis().set_ticks([])
        E_delta_plot.set_ylabel(r'Deviation [%]')
        E_delta_plot.tick_params(labelsize=fsz - 2)

        E_main_plot.tick_params(labelsize=fsz)
        E_main_plot.legend(loc='best')
        E_main_plot.set_ylabel("Total Energy" + ylbl, fontsize=fsz)
        E_main_plot.set_xlabel("Time" + xlbl, fontsize=fsz)

        E_hist_plot.hist(xmul * self.dataframe['Total Energy'], bins=nbins, density=True,
                         orientation='horizontal', alpha=0.75)
        E_hist_plot.get_xaxis().set_ticks([])
        E_hist_plot.get_yaxis().set_ticks([])

        xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(self.dt,
                                                               self.dataframe["Temperature"],
                                                               "Time",
                                                               "Temperature",
                                                               self.units)
        Info_plot.axis([0, 10, 0, 10])
        Info_plot.grid(False)

        Info_plot.text(0., 10, "Job ID: {}".format(self.job_id), fontsize=fsz)
        Info_plot.text(0., 9.5, "Phase: {}".format(self.phase.capitalize()), fontsize=fsz)
        Info_plot.text(0., 9.0, "No. of species = {}".format(len(self.species_num)), fontsize=fsz)
        y_coord = 8.5
        for isp, sp in enumerate(simulation.species):
            Info_plot.text(0., y_coord, "Species {} : {}".format(isp + 1, sp.name), fontsize=fsz)
            Info_plot.text(0.0, y_coord - 0.5, "  No. of particles = {} ".format(sp.num), fontsize=fsz)
            Info_plot.text(0.0, y_coord - 1., "  Temperature = {:.4e} {}".format(ymul * sp.temperature, ylbl),
                           fontsize=fsz)
            y_coord -= 1.5

        y_coord -= 0.25
        Info_plot.text(0., y_coord,
                       "Total $N$ = {}".format(simulation.parameters.total_num_ptcls), fontsize=fsz)
        Info_plot.text(0., y_coord - 0.5,
                       "Thermostat: {}".format(simulation.thermostat.type), fontsize=fsz)
        Info_plot.text(0., y_coord - 1.,
                       "Berendsen rate = {:1.2f}".format(simulation.thermostat.relaxation_rate), fontsize=fsz)
        Info_plot.text(0., y_coord - 1.5,
                       "Potential: {}".format(simulation.potential.type), fontsize=fsz)
        Info_plot.text(0., y_coord - 2.,
                       "Tot Force Error = {:1.4e}".format(simulation.parameters.force_error), fontsize=fsz)

        Info_plot.text(0., y_coord - 2.5,
                       "Timestep = {:1.4f} {}".format(xmul * simulation.integrator.dt, xlbl), fontsize=fsz)
        Info_plot.text(0., y_coord - 3., "{} step interval = {}".format(self.phase, self.dump_step), fontsize=fsz)
        Info_plot.text(0., y_coord - 3.5,
                       "{} completed steps = {}".format(self.phase, completed_steps),
                       fontsize=fsz)
        Info_plot.text(0., y_coord - 4., "Tot {} steps = {}".format(self.phase.capitalize(), self.no_steps),
                       fontsize=fsz)
        Info_plot.text(0., y_coord - 4.5, "{:1.2f} % {} Completed".format(
            100 * completed_steps / self.no_steps, self.phase.capitalize()), fontsize=fsz)

        Info_plot.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.fldr, 'EnsembleCheckPlot_' + self.job_id + '.png'))
        if show:
            fig.show()


class VelocityAutoCorrelationFunction(Observable):
    """Velocity Auto-correlation function."""

    def setup(self, params, phase=None):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : (optional), str
            Phase to analyze.

        params : sarkas.base.Parameters
            Simulation's parameters.

        species : list
            List of ``sarkas.base.Species``.

        """
        self.phase = phase if phase else 'production'

        super().setup_init(params, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'VelocityAutoCorrelationFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "VelocityACF_" + self.job_id + '.csv')

    def compute(self, time_averaging=False, it_skip=100):
        """
        Compute the velocity auto-correlation functions.

        Parameters
        ----------
        time_averaging: bool
            Flag for species diffusion flux time averaging. Default = False.

        it_skip: int
            Timestep interval for species diffusion flux time averaging. Default = 100

        """

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
        if time_averaging:
            print("Calculating vacf with time averaging on...")
        else:
            print("Calculating vacf with time averaging off...")

        vacf = calc_vacf(vel, self.species_num, time_averaging, it_skip)

        for i, sp1 in enumerate(self.species_names):
            self.dataframe["{} X Velocity ACF".format(sp1)] = vacf[i, 0, :]
            self.dataframe["{} Y Velocity ACF".format(sp1)] = vacf[i, 1, :]
            self.dataframe["{} Z Velocity ACF".format(sp1)] = vacf[i, 2, :]
            self.dataframe["{} Total Velocity ACF".format(sp1)] = vacf[i, 3, :]

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')


class FluxAutoCorrelationFunction(Observable):
    """Species Diffusion Flux Auto-correlation function."""

    def setup(self, params, phase=None):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : (optional), str
            Phase to analyze.

        params : sarkas.base.Parameters
            Simulation's parameters.

        """
        self.phase = phase if phase else 'production'

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

    def compute(self, time_averaging=False, it_skip=100):
        """
        Compute the velocity auto-correlation functions.

        Parameters
        ----------
        time_averaging: bool
            Flag for species diffusion flux time averaging. Default = False.

        it_skip: int
            Timestep interval for species diffusion flux time averaging. Default = 100

        """

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
        if time_averaging:
            print("Calculating diffusion flux acf with time averaging on ...")
        else:
            print("Calculating diffusion flux acf with time averaging off ...")

        df_acf = calc_diff_flux_acf(vel,
                                    self.species_num,
                                    self.species_num_dens,
                                    self.species_masses,
                                    time_averaging,
                                    it_skip)

        v_ij = 0
        for i, sp1 in enumerate(self.species_names):
            for j, sp2 in enumerate(self.species_names[i:], i):
                self.dataframe["{}-{} X Diffusion Flux ACF".format(sp1, sp2)] = df_acf[v_ij, 0, :]
                self.dataframe["{}-{} Y Diffusion Flux ACF".format(sp1, sp2)] = df_acf[v_ij, 1, :]
                self.dataframe["{}-{} Z Diffusion Flux ACF".format(sp1, sp2)] = df_acf[v_ij, 2, :]
                self.dataframe["{}-{} Total Diffusion Flux ACF".format(sp1, sp2)] = df_acf[v_ij, 3, :]
                v_ij += 1

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')


class VelocityMoments(Observable):
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

    def setup(self, params, phase=None):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to compute.

        params : sarkas.base.Parameters
            Simulation's parameters.

        """
        self.phase = phase if phase else 'production'
        super().setup_init(params, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'VelocityMoments')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.plots_dir = os.path.join(self.saving_dir, 'Plots')
        if not os.path.exists(self.plots_dir):
            os.mkdir(self.plots_dir)

        self.filename_csv = os.path.join(self.saving_dir, "VelocityMoments_" + self.job_id + '.csv')

        if not hasattr(self, 'no_bins'):
            self.no_bins = int(0.05 * params.total_num_ptcls)

        if not hasattr(self, 'max_no_moment'):
            self.max_no_moment = 3

        self.species_plots_dirs = None

    def compute(self):
        """
        Calculate the moments of the velocity distributions and save them to a pandas dataframes and csv.
        """
        vscale = 1. / (self.a_ws * self.total_plasma_frequency)
        vel = np.zeros((self.no_dumps, self.total_num_ptcls, 3))

        time = np.zeros(self.no_dumps)
        for it in range(self.no_dumps):
            dump = int(it * self.dump_step)
            datap = load_from_restart(self.dump_dir, dump)
            vel[it, :, 0] = datap["vel"][:, 0] * vscale
            vel[it, :, 1] = datap["vel"][:, 1] * vscale
            vel[it, :, 2] = datap["vel"][:, 2] * vscale
            time[it] = datap["time"]

        data = {"Time": time}
        self.dataframe = pd.DataFrame(data)
        print("Calculating velocity moments ...")
        moments = calc_moments(vel, self.no_bins, self.species_num)

        print("Calculating ratios ...")
        ratios = calc_moment_ratios(moments, self.species_num, self.no_dumps)
        # Save the dataframe
        for i, sp in enumerate(self.species_names):
            self.dataframe["{} vx 2nd moment".format(sp)] = moments[:, int(9 * i)]
            self.dataframe["{} vx 4th moment".format(sp)] = moments[:, int(9 * i) + 1]
            self.dataframe["{} vx 6th moment".format(sp)] = moments[:, int(9 * i) + 2]

            self.dataframe["{} vy 2nd moment".format(sp)] = moments[:, int(9 * i) + 3]
            self.dataframe["{} vy 4th moment".format(sp)] = moments[:, int(9 * i) + 4]
            self.dataframe["{} vy 6th moment".format(sp)] = moments[:, int(9 * i) + 5]

            self.dataframe["{} vz 2nd moment".format(sp)] = moments[:, int(9 * i) + 6]
            self.dataframe["{} vz 4th moment".format(sp)] = moments[:, int(9 * i) + 7]
            self.dataframe["{} vz 6th moment".format(sp)] = moments[:, int(9 * i) + 8]

            self.dataframe["{} 4-2 moment ratio".format(sp)] = ratios[i, 0, :]
            self.dataframe["{} 6-2 moment ratio".format(sp)] = ratios[i, 1, :]

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def plot_ratios(self, show=False):
        """
        Plot the moment ratios and save the figure.

        Parameters
        ----------
        show : bool
            Flag for displaying the figure.
        """
        self.parse()

        if not os.path.exists(self.plots_dir):
            os.mkdir(self.plots_dir)

        if self.num_species > 1:
            self.species_plots_dirs = []
            for i, sp in enumerate(self.species_names):
                new_dir = os.path.join(self.plots_dir, "{}".format(sp))
                self.species_plots_dirs.append(new_dir)
                if not os.path.exists(new_dir):
                    os.mkdir(os.path.join(self.plots_dir, "{}".format(sp)))
        else:
            self.species_plots_dirs = [self.plots_dir]

        for i, sp in enumerate(self.species_names):
            fig, ax = plt.subplots(1, 1)
            xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(self.dataframe["Time"],
                                                                   np.array([1]), 'Time', 'none', self.units)
            ax.plot(self.dataframe["Time"] * xmul,
                    self.dataframe["{} 4-2 moment ratio".format(sp)], label=r"4/2 ratio")
            ax.plot(self.dataframe["Time"] * xmul,
                    self.dataframe["{} 6-2 moment ratio".format(sp)], label=r"6/2 ratio")

            ax.axhline(1.0, ls='--', c='k', label='Equilibrium')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            #
            ax.set_xscale('log')
            if self.phase == 'equilibration':
                ax.set_yscale('log')
            ax.set_xlabel(r'$t$' + xlbl)
            #
            ax.set_title("Moments ratios of {}".format(sp) + '  Phase: ' + self.phase.capitalize())
            fig.savefig(os.path.join(self.species_plots_dirs[i], "MomentRatios_" + self.job_id + '.png'))
            if show:
                fig.show()
            else:
                # this is useful because it will have too many figures open.
                plt.close(fig)


@njit
def autocorrelationfunction(At):
    """
    Calculate the autocorrelation function of the array input.

    .. math::
        A(\\tau) =  \sum_j^D \sum_i^T A_j(t_i)A_j(t_i + \\tau)

    where :math:`D` is the number of dimensions and :math:`T` is the total length
    of the simulation.

    Parameters
    ----------
    At : numpy.ndarray
        Observable to autocorrelate. Shape=(``no_dim``, ``no_steps``).

    Returns
    -------
    ACF : numpy.ndarray
        Autocorrelation function of ``At``.
    """
    no_steps = At.shape[1]
    no_dim = At.shape[0]

    ACF = np.zeros(no_steps)
    Norm_counter = np.zeros(no_steps)

    for it in range(no_steps):
        for dim in range(no_dim):
            ACF[: no_steps - it] += At[dim, it] * At[dim, it:no_steps]
        Norm_counter[: no_steps - it] += 1.0

    return ACF / Norm_counter


@njit
def autocorrelationfunction_1D(At):
    """
    Calculate the autocorrelation function of the input.

    .. math::
        A(\\tau) =  \sum_i^T A(t_i)A(t_i + \\tau)

    where :math:`T` is the total length of the simulation.

    Parameters
    ----------
    At : numpy.ndarray
        Array to autocorrelate. Shape=(``no_steps``).

    Returns
    -------
    ACF : numpy.ndarray
        Autocorrelation function of ``At``.
    """
    no_steps = At.shape[0]
    ACF = np.zeros(no_steps)
    Norm_counter = np.zeros(no_steps)

    for it in range(no_steps):
        ACF[: no_steps - it] += At[it] * At[it:no_steps]
        Norm_counter[: no_steps - it] += 1.0

    return ACF / Norm_counter


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

    norm = dt / np.sqrt(no_dumps * dt * dump_step)
    no_skw = int(len(species_np) * (len(species_np) + 1) / 2)
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
        for s in range(num_species):
            sp_end = sp_start + sp_num[s]
            # Calculate the current of each species
            Js[s, :, it] = sp_charge[s] * np.sum(vel[it, :, sp_start:sp_end], axis=1)
            Jtot[:, it] += Js[s, :, it]

            sp_start = sp_end

    return Js, Jtot


@njit
def calc_moment_ratios(moments, species_np, no_dumps):
    """
    Take the ratio of the velocity moments.

    Parameters
    ----------
    moments: numpy.ndarray
        Velocity moments of each species per direction at each time step.

    no_dumps: int
        Number of saved timesteps.

    species_np: numpy.ndarray
        Number of particles of each species.

    Returns
    -------
    ratios: numpy.ndarray
        Ratios of high order velocity moments with respoect the 2nd moment.
        Shape=(``no_species``,2, ``no_dumps``)
    """

    no_ratios = 2
    ratios = np.zeros((len(species_np), no_ratios, no_dumps))

    sp_start = 0
    for sp, nsp in enumerate(species_np):
        sp_end = sp_start + nsp

        vx2_mom = moments[:, int(9 * sp)]
        vx4_mom = moments[:, int(9 * sp) + 1]
        vx6_mom = moments[:, int(9 * sp) + 2]

        vy2_mom = moments[:, int(9 * sp) + 3]
        vy4_mom = moments[:, int(9 * sp) + 4]
        vy6_mom = moments[:, int(9 * sp) + 5]

        vz2_mom = moments[:, int(9 * sp) + 6]
        vz4_mom = moments[:, int(9 * sp) + 7]
        vz6_mom = moments[:, int(9 * sp) + 8]

        ratios[sp, 0, :] = (vx4_mom / vx2_mom ** 2) * (vy4_mom / vy2_mom ** 2) * (vz4_mom / vz2_mom ** 2) / 27.0
        ratios[sp, 1, :] = (vx6_mom / vx2_mom ** 3) * (vy6_mom / vy2_mom ** 3) * (vz6_mom / vz2_mom ** 3) / 15.0 ** 3

        sp_start = sp_end

    print('Done')
    return ratios


def calc_moments(vel, nbins, species_np):
    """
    Calculate the even moments of the velocity distributions.

    Parameters
    ----------
    vel: numpy.ndarray
        Particles' velocity at each time step.

    nbins: int
        Number of bins to be used for the distribution.

    species_np: numpy.ndarray
        Number of particles of each species.

    Returns
    -------
    moments: numpy.ndarray
        2nd, 4th, 8th moment of the velocity distributions.
        Shape=( ``no_dumps``, ``9 * len(species_np)``)
    """

    no_dumps = vel.shape[0]
    moments = np.empty((no_dumps, int(9 * len(species_np))))

    for it in range(no_dumps):
        sp_start = 0
        for sp, nsp in enumerate(species_np):
            sp_end = sp_start + int(nsp)

            xdist, xbins = np.histogram(vel[it, sp_start:sp_end, 0], bins=nbins, density=True)
            ydist, ybins = np.histogram(vel[it, sp_start:sp_end, 1], bins=nbins, density=True)
            zdist, zbins = np.histogram(vel[it, sp_start:sp_end, 2], bins=nbins, density=True)

            vx = (xbins[:-1] + xbins[1:]) / 2.
            vy = (ybins[:-1] + ybins[1:]) / 2.
            vz = (zbins[:-1] + zbins[1:]) / 2.

            moments[it, int(9 * sp)] = np.sum(vx ** 2 * xdist) * abs(vx[1] - vx[0])
            moments[it, int(9 * sp) + 1] = np.sum(vx ** 4 * xdist) * abs(vx[1] - vx[0])
            moments[it, int(9 * sp) + 2] = np.sum(vx ** 6 * xdist) * abs(vx[1] - vx[0])

            moments[it, int(9 * sp) + 3] = np.sum(vy ** 2 * ydist) * abs(vy[1] - vy[0])
            moments[it, int(9 * sp) + 4] = np.sum(vy ** 4 * ydist) * abs(vy[1] - vy[0])
            moments[it, int(9 * sp) + 5] = np.sum(vy ** 6 * ydist) * abs(vy[1] - vy[0])

            moments[it, int(9 * sp) + 6] = np.sum(vz ** 2 * zdist) * abs(vz[1] - vz[0])
            moments[it, int(9 * sp) + 7] = np.sum(vz ** 4 * zdist) * abs(vz[1] - vz[0])
            moments[it, int(9 * sp) + 8] = np.sum(vz ** 6 * zdist) * abs(vz[1] - vz[0])

            sp_start = sp_end

    return moments


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
    # Rescale vel and acc of each particle by their individual mass
    for sp, num in enumerate(species_np):
        sp_end = sp_start + num
        vel[:, sp_start: sp_end] *= np.sqrt(species_mass[sp])
        acc[:, sp_start: sp_end] *= species_mass[sp]  # force
        sp_start = sp_end

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


@njit
def calc_diff_flux_acf(vel, sp_num, sp_dens, sp_mass, time_averaging, it_skip):
    """
    Calculate the diffusion flux autocorrelation function of each species and in each direction.

    Parameters
    ----------
    time_averaging: bool
        Flag for time averaging.

    it_skip: int
        Timestep interval for time averaging.

    vel : numpy.ndarray
        Particles' velocities.

    sp_num: numpy.ndarray
        Number of particles of each species.

    sp_dens: numpy.ndarray
        Number densities of each species.

    sp_mass: numpy.ndarray
        Particle's mass of each species.

    Returns
    -------
    jc_acf: numpy.ndarray
        Diffusion flux autocorrelation function. Shape Ns*(Ns +1)/2 x Ndim + 1 x Nt, where Ns = number of species,
        Ndim = Number of cartesian dimensions, Nt = Number of dumps.
    """
    no_dim = vel.shape[0]
    no_dumps = vel.shape[2]
    no_species = len(sp_num)
    no_vacf = int(no_species * (no_species + 1) / 2.)

    mass_densities = sp_dens * sp_mass
    tot_mass_dens = np.sum(mass_densities)
    # Center of mass velocity field of each species
    com_vel = np.zeros((no_species, no_dim, no_dumps))
    # Total center of mass velocity field, see eq.(18) in
    # Haxhimali T. et al., Diffusivity of Mixtures in Warm Dense Matter Regime.In: Graziani F., et al. (eds)
    # Frontiers and Challenges in Warm Dense Matter. Lecture Notes in Computational Science and Engineering, vol 96.
    # Springer (2014)
    tot_com_vel = np.zeros((no_dim, no_dumps))

    sp_start = 0
    sp_end = 0
    for i, ns in enumerate(sp_num):
        sp_end += ns
        com_vel[i, :, :] = np.sum(vel[:, sp_start: sp_end, :], axis=1)
        tot_com_vel += mass_densities[i] * com_vel[i, :, :] / tot_mass_dens
        sp_start = sp_end

    jc_acf = np.zeros((no_vacf, no_dim + 1, no_dumps))
    # the flux is given by eq.(19) of the above reference
    indx = 0
    if time_averaging:
        indx = 0
        for i in range(no_species):
            sp1_flux = mass_densities[i] * (com_vel[i] - tot_com_vel)
            for j in range(i, no_species):
                sp2_flux = mass_densities[j] * (com_vel[j] - tot_com_vel)
                for d in range(no_dim):
                    norm_counter = np.zeros(no_dumps)
                    temp = np.zeros(no_dumps)
                    for it in range(0, no_dumps, it_skip):
                        temp[:no_dumps - it] += correlationfunction_1D(sp1_flux[d, it:], sp2_flux[d, it:])
                        norm_counter[:(no_dumps - it)] += 1.0
                    jc_acf[indx, d, :] = temp / norm_counter

                norm_counter = np.zeros(no_dumps)
                temp = np.zeros(no_dumps)
                for it in range(0, no_dumps, it_skip):
                    temp[: no_dumps - it] += correlationfunction(sp1_flux[:, it:], sp2_flux[:, it:])
                    norm_counter[:(no_dumps - it)] += 1.0
                jc_acf[indx, no_dim, :] = temp / norm_counter
                indx += 1

    else:
        for i, rho1 in enumerate(mass_densities):
            sp1_flux = rho1 * (com_vel[i] - tot_com_vel)
            for j, rho2 in enumerate(mass_densities[i:], i):
                sign = (1 - 2 * (i != j))  # this sign seems to be an issue in the calculation of
                sp2_flux = sign * rho2 * (com_vel[j] - tot_com_vel)

                for d in range(no_dim):
                    jc_acf[indx, d, :] = correlationfunction_1D(sp1_flux[d, :], sp2_flux[d, :])

                jc_acf[indx, - 1, :] = correlationfunction(sp1_flux, sp2_flux)
                indx += 1
    return jc_acf


@njit
def calc_vacf(vel, sp_num, time_averaging, it_skip):
    """
    Calculate the velocity autocorrelation function of each species and in each direction.

    Parameters
    ----------
    time_averaging: bool
        Flag for time averaging.

    it_skip: int
        Timestep interval for time averaging.

    vel : numpy.ndarray
        Particles' velocities stored in a 3D array with shape D x Np x Nt. D = cartesian dimensions,
        Np = Number of particles, Nt = number of dumps.

    sp_num: numpy.ndarray
        Number of particles of each species.

    Returns
    -------
    vacf: numpy.ndarray
        Velocity autocorrelation functions.

    """
    no_dim = vel.shape[0]
    no_dumps = vel.shape[2]

    vacf = np.zeros((len(sp_num), no_dim + 1, no_dumps))

    if time_averaging:
        for d in range(no_dim):
            vacf_temp = np.zeros(no_dumps)
            norm_counter = np.zeros(no_dumps)

            for ptcl in range(sp_num[0]):
                for it in range(0, no_dumps, it_skip):
                    vacf_temp[: no_dumps - it] += autocorrelationfunction_1D(vel[d, ptcl, it:])
                    norm_counter[: no_dumps - it] += 1.0

            vacf[0, d, :] = vacf_temp / norm_counter

        vacf_temp = np.zeros(no_dumps)
        norm_counter = np.zeros(no_dumps)
        for ptcl in range(sp_num[0]):
            for it in range(0, no_dumps, it_skip):
                vacf_temp[: no_dumps - it] += autocorrelationfunction(vel[:, ptcl, it:])
                norm_counter[: no_dumps - it] += 1.0

        vacf[0, -1, :] = vacf_temp / norm_counter
    else:
        # Calculate the vacf of each species in each dimension
        for i in range(no_dim):
            vacf_temp = np.zeros(no_dumps)
            sp_start = 0
            sp_end = 0
            for sp, n_sp in enumerate(sp_num):
                sp_end += n_sp
                for ptcl in range(sp_start, sp_end):
                    vacf_temp += autocorrelationfunction_1D(vel[i, ptcl, :])
                vacf[sp, i, :] = vacf_temp / n_sp
                sp_start = sp_end

        vacf_temp = np.zeros(no_dumps)
        sp_start = 0
        sp_end = 0
        for sp, n_sp in enumerate(sp_num):
            sp_end += n_sp
            for ptcl in range(sp_start, sp_end):
                vacf_temp += autocorrelationfunction(vel[:, ptcl, :])
            vacf[sp, -1, :] = vacf_temp / n_sp
            sp_start = sp_end

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


@njit
def correlationfunction(At, Bt):
    """
    Calculate the correlation function :math:`\mathbf{A}(t)` and :math:`\mathbf{B}(t)`

    .. math::
        C_{AB}(\\tau) =  \sum_j^D \sum_i^T A_j(t_i)B_j(t_i + \\tau)

    where :math:`D` (= ``no_dim``) is the number of dimensions and :math:`T` (= ``no_steps``) is the total length
    of the simulation.

    Parameters
    ----------
    At : numpy.ndarray
        Observable to correlate. Shape=(``no_dim``, ``no_steps``).

    Bt : numpy.ndarray
        Observable to correlate. Shape=(``no_dim``, ``no_steps``).

    Returns
    -------
    CF : numpy.ndarray
        Correlation function :math:`C_{AB}(\\tau)`
    """
    no_steps = At.shape[1]
    no_dim = At.shape[0]

    CF = np.zeros(no_steps)
    Norm_counter = np.zeros(no_steps)

    for it in range(no_steps):
        for dim in range(no_dim):
            CF[: no_steps - it] += At[dim, it] * Bt[dim, it:no_steps]
        Norm_counter[: no_steps - it] += 1.0

    return CF / Norm_counter


@njit
def correlationfunction_1D(At, Bt):
    """
    Calculate the correlation function between :math:`A(t)` and :math:`B(t)`

    .. math::
        C_{AB}(\\tau) =  \sum_i^T A(t_i)B(t_i + \\tau)

    where :math:`T` (= ``no_steps``) is the total length of the simulation.

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
    no_steps = At.shape[0]
    CF = np.zeros(no_steps)
    Norm_counter = np.zeros(no_steps)

    for it in range(no_steps):
        CF[: no_steps - it] += At[it] * Bt[it:no_steps]
        Norm_counter[: no_steps - it] += 1.0

    return CF / Norm_counter


def kspace_setup(no_ka, box_lengths, full=False):
    """
    Calculate all allowed :math:`k` vectors.

    Parameters
    ----------
    no_ka : numpy.ndarray
        Number of harmonics in each direction.

    box_lengths : numpy.ndarray
        Length of each box's side.

    full : bool
        Flag for calculating all the possible `k` vector directions and magnitudes. Defauly = False

    Returns
    -------
    k_arr : list
        List of all possible :math:`k` vectors with their corresponding magnitudes and indexes.

    k_counts : numpy.ndarray
        Number of occurrences of each :math:`k` magnitude.

    k_unique : numpy.ndarray
        Magnitude of each allowed :math:`k` vector.
    """
    if full:
        # Obtain all possible permutations of the wave number arrays
        k_arr = [np.array([i / box_lengths[0], j / box_lengths[1], k / box_lengths[2]]) for i in range(no_ka[0] + 1)
                 for j in range(no_ka[1] + 1)
                 for k in range(no_ka[2] + 1)]
    else:
        # Obtain all possible permutations of the wave number arrays
        k_arr = [np.array([i / box_lengths[0], 0, 0]) for i in range(1, no_ka[0] + 1)]
        k_arr = np.append(k_arr,
                          [np.array([0, i / box_lengths[1], 0]) for i in range(1, no_ka[1] + 1)],
                          axis=0)
        k_arr = np.append(k_arr,
                          [np.array([0, 0, i / box_lengths[2]]) for i in range(1, no_ka[2] + 1)],
                          axis=0)

    # Compute wave number magnitude - don't use |k| (skipping first entry in k_arr)
    k_mag = np.sqrt(np.sum(np.array(k_arr) ** 2, axis=1)[..., None])

    # Add magnitude to wave number array
    k_arr = np.concatenate((k_arr, k_mag), 1)

    # Sort from lowest to highest magnitude
    ind = np.argsort(k_arr[:, -1])
    k_arr = k_arr[ind]

    # Count how many times a |k| value appears
    # int(full) = 1 if True else = 0. This is needed because if full==True
    # the first vector is k = [0, 0, 0] and needs not be counted
    k_unique, k_counts = np.unique(k_arr[int(full):, -1], return_counts=True)

    # Generate a 1D array containing index to be used in S array
    k_index = np.repeat(range(len(k_counts)), k_counts)[..., None]

    # Add index to k_array
    k_arr = np.concatenate((k_arr[int(full):, :], k_index), 1)
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

    x_str = np.format_float_scientific(xmax)
    y_str = np.format_float_scientific(ymax)

    x_exp = 10.0 ** (float(x_str[x_str.find('e') + 1:]))
    y_exp = 10.0 ** (float(y_str[y_str.find('e') + 1:]))

    # find the prefix
    xprefix = "none"
    xmul = -1.5
    i = 0.1
    while xmul < 0:
        i *= 10.
        for key, value in PREFIXES.items():
            ratio = i * x_exp / value
            if abs(ratio - 1) < 1.0e-6:
                xprefix = key
                xmul = i / value

    # find the prefix
    yprefix = "none"
    ymul = - 1.5
    i = 1.0
    while ymul < 0:
        for key, value in PREFIXES.items():
            ratio = i * y_exp / value
            if abs(ratio - 1) < 1.0e-6:
                yprefix = key
                ymul = i / value
        i *= 10.

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
