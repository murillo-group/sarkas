"""
Module for calculating physical quantities from Sarkas checkpoints.
"""
import os
from tqdm import tqdm
import numpy as np
from numba import njit
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

# Sarkas modules

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
     "Length": 'cm',
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
        self.species = list()
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
            disp += "\t{} : {}\n".format(key, value)
        disp += ')'
        return disp

    def setup_init(self, params, species, phase):
        self.__dict__.update(params.__dict__)

        if len(self.species) < params.num_species:
            for sp in species:
                self.species.append(sp)

        # Create the lists of k vectors
        if hasattr(self, 'no_ka_harmonics'):
            if isinstance(self.no_ka_harmonics, np.ndarray) == 0:
                self.no_ka_harmonics = np.ones(3, dtype=int) * self.no_ka_harmonics
        else:
            self.no_ka_harmonics = [5, 5, 5]

        self.k_space_dir = os.path.join(self.postprocessing_dir, "k_space_data")
        self.k_file = os.path.join(self.k_space_dir, "k_arrays.npz")
        self.nkt_file = os.path.join(self.k_space_dir, "nkt.npy")
        self.vkt_file = os.path.join(self.k_space_dir, "vkt.npz")

        self.no_obs = int(self.num_species * (self.num_species + 1) / 2)
        self.prod_no_dumps = len(os.listdir(self.prod_dump_dir))
        self.eq_no_dumps = len(os.listdir(self.eq_dump_dir))

        if self.phase == 'equilibration':
            self.no_dumps = self.eq_no_dumps
            self.dump_dir = self.eq_dump_dir
            self.dump_step = self.eq_dump_step

        else:
            self.no_dumps = self.prod_no_dumps
            self.dump_dir = self.prod_dump_dir
            self.dump_step = self.prod_dump_step

        if hasattr(params, 'mpl_style'):
            plt.style.use(params.mpl_style)


class CurrentCorrelationFunctions(Observable):
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

    def setup(self, params, species, phase=None):
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

        super().setup_init(params, species, self.phase)

        saving_dir = os.path.join(self.postprocessing_dir, 'CurrentCorrelationFunctions')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv_longitudinal = os.path.join(self.saving_dir,
                                                      "LongitudinalCurrentCorrelationFunction_" + self.job_id + '.csv')
        self.filename_csv_transverse = os.path.join(self.saving_dir,
                                                    "TransverseCurrentCorrelationFunction_" + self.job_id + '.csv')

    def parse(self):
        """
        Grab the pandas dataframe from the saved csv file. If file does not exist call ``compute``.
        """
        try:
            self.dataframe_longitudinal = pd.read_csv(self.filename_csv_longitudinal, index_col=False)
            self.dataframe_transverse = pd.read_csv(self.filename_csv_transverse, index_col=False)
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]
        except FileNotFoundError:
            print("\nCurrent Correlation files not found!")
            print("\nComputing CCF now ...")
            self.compute()

    def compute(self):
        """
        Calculate the velocity fluctuations correlation functions.
        """
        self.dataframe_longitudinal["Frequencies"] = 2.0 * np.pi * np.fft.fftfreq(self.no_dumps,
                                                                                  self.dt * self.dump_step)
        self.dataframe_transverse["Frequencies"] = 2.0 * np.pi * np.fft.fftfreq(self.no_dumps,
                                                                                self.dt * self.dump_step)
        # Parse vkt otherwise calculate them
        try:
            data = np.load(self.vkt_file)
            vkt = data["longitudinal"]
            vkt_i = data["transverse_i"]
            vkt_j = data["transverse_j"]
            vkt_k = data["transverse_k"]
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]
            self.no_ka_values = len(self.ka_values)

        except FileNotFoundError:
            self.k_list, self.k_counts, k_unique = kspace_setup(self.no_ka_harmonics, self.box_lengths)
            self.ka_values = 2.0 * np.pi * k_unique * self.a_ws
            self.no_ka_values = len(self.ka_values)

            if not (os.path.exists(self.k_space_dir)):
                os.mkdir(self.k_space_dir)

            np.savez(self.k_file,
                     k_list=self.k_list,
                     k_counts=self.k_counts,
                     ka_values=self.ka_values)

            vkt, vkt_i, vkt_j, vkt_k = calc_vkt(self.dump_dir, self.no_dumps, self.dump_step,
                                                self.species_num,
                                                self.k_list)
            np.savez(self.vkt_file,
                     longitudinal=vkt,
                     transverse_i=vkt_i,
                     transverse_j=vkt_j,
                     transverse_k=vkt_k)

        # Calculate Lkw
        Lkw = calc_Skw(vkt, self.k_list, self.k_counts, self.species_num, self.no_dumps, self.dt,
                       self.dump_step)
        Tkw_i = calc_Skw(vkt_i, self.k_list, self.k_counts, self.species_num, self.no_dumps, self.dt,
                         self.dump_step)
        Tkw_j = calc_Skw(vkt_j, self.k_list, self.k_counts, self.species_num, self.no_dumps, self.dt,
                         self.dump_step)
        Tkw_k = calc_Skw(vkt_k, self.k_list, self.k_counts, self.species_num, self.no_dumps, self.dt,
                         self.dump_step)
        Tkw = (Tkw_i + Tkw_j + Tkw_k) / 3.0
        print("Saving L(k,w) and T(k,w)")
        sp_indx = 0
        for i, sp1 in enumerate(self.species):
            for j, sp2 in enumerate(self.species[i:]):
                for ik in range(len(self.k_counts)):
                    if ik == 0:
                        column = "{}-{} CCF ka_min".format(sp1.name, sp2.name)
                    else:
                        column = "{}-{} CCF {} ka_min".format(sp1.name, sp2.name, ik + 1)

                    self.dataframe_longitudinal[column] = Lkw[sp_indx, ik, :]
                    self.dataframe_transverse[column] = Tkw[sp_indx, ik, :]
                sp_indx += 1

        self.dataframe_longitudinal.to_csv(self.filename_csv_transverse, index=False, encoding='utf-8')
        self.dataframe_transverse.to_csv(self.filename_csv_longitudinal, index=False, encoding='utf-8')

    def plot(self, show=False, longitudinal=True, dispersion=False):
        """
        Plot velocity fluctuations correlation functions and save the figure.

        Parameters
        ----------
        longitudinal : bool
            Flag for plotting longitudinal or transverse correlation function. Default=True.

        show: bool
            Flag for prompting the plots to screen. Default=False

        dispersion : bool
            Flag for plotting the collective mode dispersion. Default=False

        """
        try:
            if longitudinal:
                self.dataframe = pd.read_csv(self.filename_csv_longitudinal, index_col=False)
                lbl = "L"
            else:
                self.dataframe = pd.read_csv(self.filename_csv_transverse, index_col=False)
                lbl = "T"
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]
            self.no_ka_values = len(self.ka_values)
        except FileNotFoundError:
            print("Computing L(k,w), T(k,w)")
            self.compute()
            if longitudinal:
                self.dataframe = pd.read_csv(self.filename_csv_longitudinal, index_col=False)
                lbl = "L"
            else:
                self.dataframe = pd.read_csv(self.filename_csv_transverse, index_col=False)
                lbl = "T"

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        if self.num_species > 1:
            for i, sp1 in enumerate(self.species):
                for j, sp2 in enumerate(self.species[i:]):
                    column = "{}-{} CCF ka_min".format(sp1.name, sp2.name)
                    ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.species_wp[0],
                            np.fft.fftshift(self.dataframe[column]),
                            label=r'$' + lbl + '_{' + sp1.name + sp2.name + '}(k,\omega)$')
        else:
            column = "{}-{} CCF ka_min".format(self.species_names[0], self.species_names[0])
            ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.species_wp[0],
                    np.fft.fftshift(self.dataframe[column]),
                    label=r'$ka = {:1.4f}$'.format(self.ka_values[0]))
            for i in range(1, 5):
                column = "{}-{} CCF {} ka_min".format(self.species_names[0], self.species_names[0], i + 1)
                ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.total_plasma_frequency,
                        np.fft.fftshift(self.dataframe[column]),
                        label=r'$ka = {:1.4f}$'.format(self.ka_values[i]))

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=3)
        ax.set_yscale('log')
        if longitudinal:
            ax.set_ylabel(r'$L(k,\omega)$')
            fig_name = os.path.join(self.saving_dir, 'Lkw_' + self.job_id + '.png')
        else:
            ax.set_ylabel(r'$T(k,\omega)$')
            fig_name = os.path.join(self.saving_dir, 'Tkw_' + self.job_id + '.png')

        ax.set_xlabel(r'$\omega/\omega_p$')
        fig.tight_layout()
        fig.savefig(fig_name)
        if show:
            fig.show()

        if dispersion:
            w_array = np.array(self.dataframe["Frequencies"]) / self.total_plasma_frequency
            neg_indx = np.where(w_array < 0.0)[0][0]
            Skw = np.array(self.dataframe.iloc[:, 1:self.no_ka_values + 1])
            ka_vals, w = np.meshgrid(self.ka_values, w_array[:neg_indx])
            fig = plt.figure(figsize=(10, 7))
            plt.pcolor(ka_vals, w, Skw[neg_indx:, :], vmin=Skw[:, 1].min(), vmax=Skw[:, 1].max())
            cbar = plt.colorbar()
            cbar.set_ticks([])
            cbar.ax.tick_params(labelsize=14)
            plt.xlabel(r'$ka$')
            plt.ylabel(r'$\omega/\omega_p$')
            plt.ylim(0, 2)
            plt.tick_params(axis='both', which='major')
            fig.tight_layout()
            if longitudinal:
                fig.savefig(os.path.join(self.saving_dir, 'Lkw_Dispersion_' + self.job_id + '.png'))
            else:
                fig.savefig(os.path.join(self.saving_dir, 'Tkw_Dispersion_' + self.job_id + '.png'))
            if show:
                fig.show()


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

    def setup(self, params, species, phase=None):
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
        super().setup_init(params, species, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'DynamicStructureFactor')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "DynamicStructureFactor_" + self.job_id + '.csv')

    def parse(self):
        """
        Grab the pandas dataframe from the saved csv file. If file does not exist call ``compute``.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]

        except FileNotFoundError:
            print("\nFile {} not found!".format(self.filename_csv))
            print("\nComputing DSF now...")
            self.compute()
        return

    def compute(self):
        """
        Compute :math:`S_{ij} (k,\\omega)` and the array of :math:`\\omega` values.
        ``self.Skw``. Shape = (``no_ws``, ``no_Sij``)
        """

        # Parse nkt otherwise calculate it
        try:
            nkt = np.load(self.nkt_file)
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]
            self.no_ka_values = len(self.ka_values)

        except FileNotFoundError:
            self.k_list, self.k_counts, k_unique = kspace_setup(self.no_ka_harmonics, self.box_lengths)
            self.ka_values = 2.0 * np.pi * k_unique * self.a_ws
            self.no_ka_values = len(self.ka_values)

            if not (os.path.exists(self.k_space_dir)):
                os.mkdir(self.k_space_dir)

            np.savez(self.k_file,
                     k_list=self.k_list,
                     k_counts=self.k_counts,
                     ka_values=self.ka_values)

            nkt = calc_nkt(self.dump_dir, self.no_dumps, self.dump_step, self.species_num, self.k_list)
            np.save(self.nkt_file, nkt)

        self.dataframe["Frequencies"] = 2.0 * np.pi * np.fft.fftfreq(self.no_dumps, self.dt * self.dump_step)

        # Calculate Skw
        Skw = calc_Skw(nkt, self.k_list, self.k_counts, self.species_num, self.no_dumps, self.dt,
                       self.dump_step)
        print("Saving S(k,w)")
        sp_indx = 0
        for i, sp1 in enumerate(self.species):
            for j, sp2 in enumerate(self.species[i:]):
                for ik in range(len(self.k_counts)):
                    if ik == 0:
                        column = "{}-{} DSF ka_min".format(sp1.name, sp2.name)
                    else:
                        column = "{}-{} DSF {} ka_min".format(sp1.name, sp2.name, ik + 1)
                    self.dataframe[column] = Skw[sp_indx, ik, :]
                sp_indx += 1

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def plot(self, show=False, dispersion=False):
        """
        Plot :math:`S(k,\\omega)` and save the figure.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]
            self.no_ka_values = len(self.ka_values)
        except FileNotFoundError:
            print("Computing S(k,w)")
            self.compute()

        fig, ax = plt.subplots(1, 1)
        if self.num_species > 1:
            for i, sp1 in enumerate(self.species):
                for j, sp2 in enumerate(self.species[i:]):
                    column = "{}-{} DSF ka_min".format(sp1.name, sp2.name)
                    ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.species_wp[0],
                            np.fft.fftshift(self.dataframe[column]),
                            label=r'$S_{' + sp1.name + sp2.name + '}(k,\omega)$')
        else:
            column = "{}-{} DSF ka_min".format(self.species_names[0], self.species_names[0])
            ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.species_wp[0],
                    np.fft.fftshift(self.dataframe[column]),
                    label=r'$ka = {:1.4f}$'.format(self.ka_values[0]))
            for i in range(1, 5):
                column = "{}-{} DSF {} ka_min".format(self.species[0].name, self.species[0].name, i + 1)
                ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.total_plasma_frequency,
                        np.fft.fftshift(self.dataframe[column]),
                        label=r'$ka = {:1.4f}$'.format(self.ka_values[i]))

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=3)
        ax.set_yscale('log')
        ax.set_ylabel(r'$S(k,\omega)$')
        ax.set_xlabel(r'$\omega/\omega_p$')
        fig.tight_layout()
        fig.savefig(os.path.join(self.saving_dir, 'Skw_' + self.job_id + '.png'))
        if show:
            fig.show()

        if dispersion:
            w_array = np.array(self.dataframe["Frequencies"]) / self.total_plasma_frequency
            neg_indx = np.where(w_array < 0.0)[0][0]
            Skw = np.array(self.dataframe.iloc[:, 1:self.no_ka_values + 1])
            ka_vals, w = np.meshgrid(self.ka_values, w_array[: neg_indx])
            fig = plt.figure(figsize=(10, 7))
            plt.pcolor(ka_vals, w, Skw[: neg_indx, :], vmin=Skw[:, 1].min(), vmax=Skw[:, 1].max())
            cbar = plt.colorbar()
            cbar.set_ticks([])
            cbar.ax.tick_params(labelsize=14)
            plt.xlabel(r'$ka$')
            plt.ylabel(r'$\omega/\omega_p$')
            plt.ylim(0, 2)
            plt.tick_params(axis='both', which='major')
            plt.title(r"$S(k,\omega)$")
            fig.tight_layout()
            fig.savefig(os.path.join(self.saving_dir, 'Skw_Dispersion_' + self.job_id + '.png'))
            if show:
                fig.show()


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

    def setup(self, params, species, phase=None):
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

        super().setup_init(params, species, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'ElectricCurrent')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "ElectricCurrent_" + self.job_id + '.csv')

    def parse(self):
        """
        Grab the pandas dataframe from the saved csv file. If file does not exist call ``compute``.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            print("\nFile {} not found!".format(self.filename_csv))
            print("\nComputing Electric Current now ...")
            self.compute()

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
        for i, sp in enumerate(self.species):
            tot_acf = autocorrelationfunction(species_current[i, :, :])
            acf_xx = autocorrelationfunction_1D(species_current[i, 0, :])
            acf_yy = autocorrelationfunction_1D(species_current[i, 1, :])
            acf_zz = autocorrelationfunction_1D(species_current[i, 2, :])

            self.dataframe["{} Total Current".format(sp.name)] = np.sqrt(
                species_current[i, 0, :] ** 2 + species_current[i, 1, :] ** 2 + species_current[i, 2, :] ** 2)
            self.dataframe["{} X Current".format(sp.name)] = species_current[i, 0, :]
            self.dataframe["{} Y Current".format(sp.name)] = species_current[i, 1, :]
            self.dataframe["{} Z Current".format(sp.name)] = species_current[i, 2, :]

            self.dataframe["{} Total Current ACF".format(sp.name)] = tot_acf
            self.dataframe["{} X Current ACF".format(sp.name)] = acf_xx
            self.dataframe["{} Y Current ACF".format(sp.name)] = acf_yy
            self.dataframe["{} Z Current ACF".format(sp.name)] = acf_zz

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def plot(self, show=False):
        """
        Plot the electric current autocorrelation function and save the figure.

        Parameters
        ----------
        show: bool
            Prompt the plot to screen.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            self.compute()

        # plt.style.use(style)

        fig, ax = plt.subplots(1, 1)
        xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(self.dataframe["Time"],
                                                               self.dataframe["Total Current ACF"], "Time", "none",
                                                               self.units)
        ax.plot(xmul * self.dataframe["Time"],
                self.dataframe["Total Current ACF"] / self.dataframe["Total Current ACF"][0],
                '--o', label=r'$J_{tot} (t)$')

        ax.legend(loc='upper right')
        ax.set_ylabel(r'$J(t)$')
        ax.set_xlabel('Time' + xlbl)
        ax.set_xscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(self.saving_dir, 'TotalCurrentACF_' + self.job_id + '.png'))
        if show:
            fig.show()


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

    def setup(self, params, species, phase=None):
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

        self.phase = phase if phase else 'equilibration'

        super().setup_init(params, species, self.phase)

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
            for sp in range(self.num_species):
                sp_end = int(sp_start + self.species_num[sp])
                x_hist, xbins = np.histogram(vel[0, sp_start:sp_end] * vscale, bins=self.no_bins, density=True)
                y_hist, ybins = np.histogram(vel[1, sp_start:sp_end] * vscale, bins=self.no_bins, density=True)
                z_hist, zbins = np.histogram(vel[2, sp_start:sp_end] * vscale, bins=self.no_bins, density=True)

                # Center the bins
                vx = 0.5 * (xbins[:-1] + xbins[1:])
                vy = 0.5 * (ybins[:-1] + ybins[1:])
                vz = 0.5 * (zbins[:-1] + zbins[1:])

                xcoeff[sp, it, :] = calculate_herm_coeff(vx, x_hist, self.hermite_order)
                ycoeff[sp, it, :] = calculate_herm_coeff(vy, y_hist, self.hermite_order)
                zcoeff[sp, it, :] = calculate_herm_coeff(vz, z_hist, self.hermite_order)

                sp_start = sp_end

        data = {"Time": time}
        self.dataframe = pd.DataFrame(data)
        for i, sp in enumerate(self.species):
            for hi in range(self.hermite_order + 1):
                self.dataframe["{} Hermite x Coeff a{}".format(sp.name, hi)] = xcoeff[i, :, hi]
                self.dataframe["{} Hermite y Coeff a{}".format(sp.name, hi)] = ycoeff[i, :, hi]
                self.dataframe["{} Hermite z Coeff a{}".format(sp.name, hi)] = zcoeff[i, :, hi]

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def parse(self):
        """
        Grab the pandas dataframe from the saved csv file. If file does not exist call ``compute``.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            print("\nFile {} not found!".format(self.filename_csv))
            print("\nComputing Hermite Coefficients now ...")
            self.compute()

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

    def setup(self, params, species, phase=None):
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

        super().setup_init(params, species, self.phase)

        saving_dir = os.path.join(self.postprocessing_dir, 'RadialDistributionFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir,
                                         "RadialDistributionFunction_" + self.job_id + ".csv")
        self.rc = self.cutoff_radius

    def save(self, rdf_hist=None):
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
        for i, sp1 in enumerate(self.species):
            pair_density[i, i] = 0.5 * sp1.num * (sp1.num - 1) / self.box_volume
            if self.num_species > 1:
                for j, sp2 in enumerate(self.species[i:], i):
                    pair_density[i, j] = sp1.num * sp2.num / self.box_volume
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
        for i, sp1 in enumerate(self.species):
            for j, sp2 in enumerate(self.species[i:], i):
                denom_const = (2.0 - (i != j)) / (pair_density[i, j] * self.production_steps)
                gr[:, gr_ij] = rdf_hist[:, i, j] * denom_const / bin_vol[:]

                self.dataframe['{}-{} RDF'.format(sp1.name, sp2.name)] = gr[:, gr_ij]

                gr_ij += 1
        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def parse(self):
        """
        Grab the pandas dataframe from the saved csv file.
        """
        self.dataframe = pd.read_csv(self.filename_csv, index_col=False)

    def plot(self, normalized=False, show=False):
        """
        Plot :math: `g_{ij}(r)` and save the figure.

        Parameters
        ----------
        normalized: bool
            Flag for normalizing distances.

        show : bool
            Flag for prompting the plot to screen. Default=False
        """
        self.dataframe = pd.read_csv(self.filename_csv, index_col=False)

        indx = 0
        xmul, ymul, xpref, ypref, xlbl, ylbl = plot_labels(self.dataframe["distance"], 1, 'Length', 'none', self.units)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for i, sp1 in enumerate(self.species):
            for j, sp2 in enumerate(self.species[i:]):
                subscript = sp1.name + sp2.name
                if normalized:
                    ax.plot(self.dataframe["distance"] / self.a_ws,
                            self.dataframe["{}-{} RDF".format(sp1.name, sp2.name)],
                            label=r'$g_{' + subscript + '} (r)$')
                else:
                    ax.plot(self.dataframe["distance"] * xmul,
                            self.dataframe["{}-{} RDF".format(sp1.name, sp2.name)],
                            label=r'$g_{' + subscript + '} (r)$')

                indx += 1
        ax.grid(True, alpha=0.3)
        if self.num_species > 2:
            ax.legend(loc='best', ncol=(self.num_species - 1))
        else:
            ax.legend(loc='best')

        ax.set_ylabel(r'$g(r)$')
        if normalized:
            ax.set_xlabel(r'$r/a$')
        else:
            ax.set_xlabel(r'$r$' + xlbl)
        # ax.set_ylim(0, 5)
        fig.tight_layout()
        fig.savefig(os.path.join(self.saving_dir, 'RDF_' + self.job_id + '.png'))
        if show:
            fig.show()


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

    def setup(self, params, species, phase=None):
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
        super().setup_init(params, species, self.phase)

        saving_dir = os.path.join(self.postprocessing_dir, 'StaticStructureFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "StaticStructureFunction_" + self.job_id + ".csv")

    def parse(self):
        """
        Grab the pandas dataframe from the saved csv file. If file does not exist call ``compute``.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            print("\nFile {} not found!".format(self.filename_csv))
            print("\nComputing S(k) now")
            self.compute()

    def compute(self):
        """
        Calculate all :math:`S_{ij}(k)`, save them into a Pandas dataframe, and write them to a csv.
        """
        # Parse nkt otherwise calculate it
        try:
            nkt = np.load(self.nkt_file)
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]
            self.no_ka_values = len(self.ka_values)
            print("n(k,t) Loaded")
        except FileNotFoundError:
            self.k_list, self.k_counts, k_unique = kspace_setup(self.no_ka_harmonics, self.box_lengths)
            self.ka_values = 2.0 * np.pi * k_unique * self.a_ws
            self.no_ka_values = len(self.ka_values)

            if not (os.path.exists(self.k_space_dir)):
                os.mkdir(self.k_space_dir)

            np.savez(self.k_file,
                     k_list=self.k_list,
                     k_counts=self.k_counts,
                     ka_values=self.ka_values)

            nkt = calc_nkt(self.dump_dir, self.no_dumps, self.dump_step, self.species_num, self.k_list)
            np.save(self.nkt_file, nkt)

        self.dataframe["ka values"] = self.ka_values

        print("Calculating S(k) ...")
        Sk_all = calc_Sk(nkt, self.k_list, self.k_counts, self.species_num, self.no_dumps)
        Sk = np.mean(Sk_all, axis=-1)
        Sk_err = np.std(Sk_all, axis=-1)

        sp_indx = 0
        for i, sp1 in enumerate(self.species):
            for j, sp2 in enumerate(self.species[i:]):
                column = "{}-{} SSF".format(sp1.name, sp2.name)
                err_column = "{}-{} SSF Errorbar".format(sp1.name, sp2.name)
                self.dataframe[column] = Sk[sp_indx, :]
                self.dataframe[err_column] = Sk_err[sp_indx, :]

                sp_indx += 1

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def plot(self, show=False, errorbars=False):
        """
        Plot :math:`S_{ij}(k)` and save the figure.

        Parameters
        ----------
        show : bool
            Flag to prompt the figure to screen. Default=False.

        errorbars : bool
            Plot errorbars. Default = False.

        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            self.compute()

        fig, ax = plt.subplots(1, 1)
        for i, sp1 in enumerate(self.species):
            for j, sp2 in enumerate(self.species[i:]):
                subscript = sp1.name + sp2.name
                if errorbars:
                    ax.errorbar(self.dataframe["ka values"],
                                self.dataframe["{}-{} SSF".format(sp1.name, sp2.name)],
                                yerr=self.dataframe[
                                    "{}-{} SSF Errorbar".format(sp1.name, sp2.name)],
                                ls='--', marker='o', label=r'$S_{ ' + subscript + '} (k)$')
                else:
                    ax.plot(self.dataframe["ka values"],
                            self.dataframe["{}-{} SSF".format(sp1.name, sp2.name)],
                            label=r'$S_{ ' + subscript + '} (k)$')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylabel(r'$S(k)$')
        ax.set_xlabel(r'$ka$')
        fig.tight_layout()
        fig.savefig(os.path.join(self.saving_dir, 'StaticStructureFactor_' + self.job_id + '.png'))
        if show:
            fig.show()


class Thermodynamics(Observable):
    """
    Thermodynamic functions.
    """

    def setup(self, params, species, phase=None):
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
        if not hasattr(self, 'phase'):
            self.phase = phase if phase else 'production'

        super().setup_init(params, species, self.phase)
        self.dataframe = pd.DataFrame()

        if params.load_method == "restart":
            self.restart_sim = True
        else:
            self.restart_sim = False

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

        potential_matrix: ndarray
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

    def plot(self, quantity="Total Energy", phase=None, show=False):
        """
        Plot ``quantity`` vs time and save the figure with appropriate name.

        Parameters
        ----------
        phase
        show : bool
            Flag for displaying figure.

        quantity : str
            Quantity to plot. Default = Total Energy.
        """

        if phase:
            self.phase = phase
            self.parse(phase)

        # plt.style.use('MSUstyle')

        if quantity[:8] == "Pressure":
            if not "Pressure" in self.dataframe.columns:
                print("Calculating Pressure quantities ...")
                self.compute_pressure_quantities()

        xmul, ymul, xpref, ypref, xlbl, ylbl = plot_labels(self.dataframe["Time"],
                                                           self.dataframe[quantity],
                                                           "Time", quantity, self.units)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        yq = {"Total Energy": r"$E_{tot}(t)$", "Kinetic Energy": r"$K_{tot}(t)$", "Potential Energy": r"$U_{tot}(t)$",
              "Temperature": r"$T(t)$",
              "Pressure Tensor ACF": r'$P_{\alpha\beta} = \langle P_{\alpha\beta}(0)P_{\alpha\beta}(t)\rangle$',
              "Pressure Tensor": r"$P_{\alpha\beta}(t)$", "Gamma": r"$\Gamma(t)$", "Pressure": r"$P(t)$"}
        dim_lbl = ['x', 'y', 'z']

        if quantity == "Pressure Tensor ACF":
            for i, dim1 in enumerate(dim_lbl):
                for j, dim2 in enumerate(dim_lbl):
                    ax.plot(self.dataframe["Time"] * xmul,
                            self.dataframe["Pressure Tensor ACF {}{}".format(dim1, dim2)] /
                            self.dataframe["Pressure Tensor ACF {}{}".format(dim1, dim2)][0],
                            label=r'$P_{' + dim1 + dim2 + '} (t)$')
            ax.set_xscale('log')
            ax.legend(loc='best', ncol=3)
            ax.set_ylim(-1, 1.5)

        elif quantity == "Pressure Tensor":
            for i, dim1 in enumerate(dim_lbl):
                for j, dim2 in enumerate(dim_lbl):
                    ax.plot(self.dataframe["Time"] * xmul,
                            self.dataframe["Pressure Tensor {}{}".format(dim1, dim2)] * ymul,
                            label=r'$P_{' + dim1 + dim2 + '} (t)$')
            ax.set_xscale('log')
            ax.legend(loc='best', ncol=3)

        elif quantity == 'Temperature' and self.num_species > 1:
            for sp in self.species_names:
                qstr = "{} Temperature".format(sp)
                ax.plot(self.dataframe["Time"] * xmul, self.dataframe[qstr] * ymul, label=qstr)
            ax.plot(self.dataframe["Time"] * xmul, self.dataframe["Temperature"] * ymul, label='Total Temperature')
            ax.legend(loc='best')
        else:
            ax.plot(self.dataframe["Time"] * xmul, self.dataframe[quantity] * ymul)

        ax.grid(True, alpha=0.3)
        ax.set_ylabel(yq[quantity] + ylbl)
        ax.set_xlabel(r'Time' + xlbl)
        fig.tight_layout()
        fig.savefig(os.path.join(self.fldr, quantity + '_' + self.job_id + '.png'))
        if show:
            fig.show()

    def parse(self, phase=None):
        """
        Grab the pandas dataframe from the saved csv file.
        """
        if phase:
            self.phase = phase

        if self.phase == 'equilibration':
            self.dataframe = pd.read_csv(self.eq_energy_filename, index_col=False)
            self.fldr = self.equilibration_dir
        else:
            self.dataframe = pd.read_csv(self.prod_energy_filename, index_col=False)
            self.fldr = self.production_dir

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

        return

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
            self.phase = phase
            if self.phase == 'equilibration':
                self.no_dumps = self.eq_no_dumps
                self.dump_dir = self.eq_dump_dir
                self.dump_step = self.eq_dump_step
                self.fldr = self.equilibration_dir
                self.no_steps = self.equilibration_steps

            else:
                self.no_dumps = self.prod_no_dumps
                self.dump_dir = self.prod_dump_dir
                self.dump_step = self.eq_dump_step
                self.fldr = self.production_dir
                self.no_steps = self.production_steps

            self.parse(phase)
        else:
            self.parse()

        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(4, 8)
        fsz = 14

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
        T_cumavg = self.dataframe["Temperature"].cumsum() / [i for i in range(1, self.no_dumps + 1)]

        T_main_plot.plot(xmul * self.dataframe["Time"], ymul * self.dataframe["Temperature"], alpha=0.7)
        T_main_plot.plot(xmul * self.dataframe["Time"], ymul * T_cumavg, label='Cum Avg')
        T_main_plot.axhline(ymul * self.T_desired, ls='--', c='r', alpha=0.7, label='Desired T')

        Delta_T = (self.dataframe["Temperature"] - self.T_desired) * 100 / self.T_desired
        Delta_T_cum_avg = Delta_T.cumsum() / [i for i in range(1, self.no_dumps  + 1)]
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
                                                               "Total Energy",
                                                               self.units)

        E_cumavg = self.dataframe["Total Energy"].cumsum() / [i for i in range(1, self.no_dumps + 1)]

        E_main_plot.plot(xmul * self.dataframe["Time"], ymul * self.dataframe["Total Energy"], alpha=0.7)
        E_main_plot.plot(xmul * self.dataframe["Time"], ymul * E_cumavg, label='Cum Avg')
        E_main_plot.axhline(ymul * self.dataframe["Total Energy"].mean(), ls='--', c='r', alpha=0.7, label='Avg')

        Delta_E = (self.dataframe["Total Energy"] - self.dataframe["Total Energy"][0]) * 100 / \
                  self.dataframe["Total Energy"][0]
        Delta_E_cum_avg = Delta_E.cumsum() / [i for i in range(1, self.no_dumps + 1)]

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
        for isp, sp in enumerate(self.species):
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
                       "{} completed steps = {}".format(self.phase, self.dump_step * (self.no_dumps - 1)),
                       fontsize=fsz)
        Info_plot.text(0., y_coord - 4., "Tot {} steps = {}".format(self.phase.capitalize(), self.no_steps), fontsize=fsz)
        Info_plot.text(0., y_coord - 4.5, "{:1.2f} % {} Completed".format(
            100 * self.dump_step * (self.no_dumps - 1) / self.no_steps, self.phase.capitalize()), fontsize=fsz)

        Info_plot.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.fldr, 'EnsembleCheckPlot_' + self.job_id + '.png'))
        if show:
            fig.show()


class VelocityAutocorrelationFunctions(Observable):
    """Velocity Auto-correlation function."""

    def setup(self, params, species, phase=None):
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

        super().setup_init(params, species, self.phase)

        # Create the directory where to store the computed data
        saving_dir = os.path.join(self.postprocessing_dir, 'VelocityAutoCorrelationFunction')
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)

        self.saving_dir = os.path.join(saving_dir, self.phase.capitalize())
        if not os.path.exists(self.saving_dir):
            os.mkdir(self.saving_dir)

        self.filename_csv = os.path.join(self.saving_dir, "VelocityACF_" + self.job_id + '.csv')

    def parse(self):
        """
        Grab the pandas dataframe from the saved csv file. If file does not exist call ``compute``.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            print("\nFile {} not found!".format(self.filename_csv))
            print("\nComputing VACF now ...")
            self.compute()

    def compute(self, time_averaging=False, it_skip=100):
        """
        Compute the velocity auto-correlation functions.
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

        if self.num_species > 1:
            vacf = calc_vacf(vel, self.species_num, self.species_masses, time_averaging, it_skip)
        else:
            vacf = calc_vacf_single(vel, self.species_num, time_averaging, it_skip)
        # Save to csv
        v_ij = 0
        for i, sp1 in enumerate(self.species):
            self.dataframe["{} X Velocity ACF".format(sp1.name)] = vacf[v_ij, 0, :]
            self.dataframe["{} Y Velocity ACF".format(sp1.name)] = vacf[v_ij, 1, :]
            self.dataframe["{} Z Velocity ACF".format(sp1.name)] = vacf[v_ij, 2, :]
            self.dataframe["{} Total Velocity ACF".format(sp1.name)] = vacf[v_ij, 3, :]
            for j, sp2 in enumerate(self.species[i + 1:]):
                v_ij += 1
                self.dataframe["{}-{} X Current ACF".format(sp1.name, sp2.name)] = vacf[v_ij, 0, :]
                self.dataframe["{}-{} Y Current ACF".format(sp1.name, sp2.name)] = vacf[v_ij, 1, :]
                self.dataframe["{}-{} Z Current ACF".format(sp1.name, sp2.name)] = vacf[v_ij, 2, :]
                self.dataframe["{}-{} Total Current ACF".format(sp1.name, sp2.name)] = vacf[v_ij, 3, :]

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def plot(self, intercurrent=False, show=False):
        """
        Plot the velocity autocorrelation function and save the figure.

        Parameters
        ----------
        show: bool
            Flag for displaying the figure.

        intercurrent: bool
            Flag for plotting inter-species currents instead of vacf.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            self.compute()

        if intercurrent:
            fig, ax = plt.subplots(1, 1)
            for i, sp_name in enumerate(self.species_names):
                for j in range(i + 1, self.num_species):
                    J = self.dataframe["{}-{} Total Current ACF".format(sp_name, self.species_names[j])]
                    ax.plot(self.dataframe["Time"] * self.total_plasma_frequency,
                            J / J[0], label=r'$J_{' + sp_name + self.species_names[j] + '} (t)$')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_ylabel(r'$J(t)$', )
            ax.set_xlabel(r'$\omega_p t$')
            ax.set_xscale('log')
            ax.set_ylim(-0.2, 1.2)
            fig.tight_layout()
            fig.savefig(os.path.join(self.saving_dir, 'InterCurrentACF_' + self.job_id + '.png'))
            if show:
                fig.show()
        else:

            xmul, ymul, xpref, ypref, xlbl, ylbl = plot_labels(self.dataframe["Time"], 1.0,
                                                               "Time", 'none', self.units)
            fig, ax = plt.subplots(1, 1)
            for i, sp in enumerate(self.species):
                Z = self.dataframe["{} Total Velocity ACF".format(sp.name)]
                ax.plot(self.dataframe["Time"] * xmul, Z / Z[0], label=r'$Z_{' + sp.name + '} (t)$')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_ylabel(r'$Z(t)$')
            ax.set_xlabel(r'Time' + xlbl)
            ax.set_xscale('log')
            ax.set_ylim(-0.2, 1.2)
            fig.tight_layout()
            fig.savefig(os.path.join(self.saving_dir, 'TotalVelocityACF_' + self.job_id + '.png'))
            if show:
                fig.show()


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

    def setup(self, params, species, phase=None):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        phase : str
            Phase to compute.

        params : sarkas.base.Parameters
            Simulation's parameters.

        species : list
            List of sarkas.base.Species.

        """
        self.phase = phase if phase else 'production'
        super().setup_init(params, species, self.phase)

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
        for i, sp in enumerate(self.species):
            self.dataframe["{} vx 2nd moment".format(sp.name)] = moments[:, int(9 * i)]
            self.dataframe["{} vx 4th moment".format(sp.name)] = moments[:, int(9 * i) + 1]
            self.dataframe["{} vx 6th moment".format(sp.name)] = moments[:, int(9 * i) + 2]

            self.dataframe["{} vy 2nd moment".format(sp.name)] = moments[:, int(9 * i) + 3]
            self.dataframe["{} vy 4th moment".format(sp.name)] = moments[:, int(9 * i) + 4]
            self.dataframe["{} vy 6th moment".format(sp.name)] = moments[:, int(9 * i) + 5]

            self.dataframe["{} vz 2nd moment".format(sp.name)] = moments[:, int(9 * i) + 6]
            self.dataframe["{} vz 4th moment".format(sp.name)] = moments[:, int(9 * i) + 7]
            self.dataframe["{} vz 6th moment".format(sp.name)] = moments[:, int(9 * i) + 8]

            self.dataframe["{} 4-2 moment ratio".format(sp.name)] = ratios[i, 0, :]
            self.dataframe["{} 6-2 moment ratio".format(sp.name)] = ratios[i, 1, :]

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def parse(self):
        """
        Grab the pandas dataframe from the saved csv file. If file does not exist call ``compute``.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            print("\nFile {} not found!".format(self.filename_csv))
            print("\nComputing Moments now ...")
            self.compute()

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
            for i, sp in enumerate(self.species):
                new_dir = os.path.join(self.plots_dir, "{}".format(sp.name))
                self.species_plots_dirs.append(new_dir)
                if not os.path.exists(new_dir):
                    os.mkdir(os.path.join(self.plots_dir, "{}".format(sp.name)))
        else:
            self.species_plots_dirs = [self.plots_dir]

        for i, sp in enumerate(self.species):
            fig, ax = plt.subplots(1, 1)
            xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(self.dataframe["Time"],
                                                                   np.array([1]), 'Time', 'none', self.units)
            ax.plot(self.dataframe["Time"] * xmul,
                    self.dataframe["{} 4-2 moment ratio".format(sp.name)], label=r"4/2 ratio")
            ax.plot(self.dataframe["Time"] * xmul,
                    self.dataframe["{} 6-2 moment ratio".format(sp.name)], label=r"6/2 ratio")
            ax.axhline(1.0, ls='--', c='k', label='Equilibrium')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            #
            ax.set_xscale('log')
            if self.phase == 'equilibration':
                ax.set_yscale('log')
            ax.set_xlabel(r'$t$' + xlbl)
            #
            ax.set_title("Moments ratios of {}".format(sp.name) + '  Phase: ' + self.phase.capitalize())
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
    At : ndarray
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
    nkt : ndarray, cmplx
        Density fluctuations of all species. Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    ka_list :
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``ka_counts``.
        Shape=(`no_ka_values`, 5)

    ka_counts : numpy.ndarray
        Number of times each :math:`k` magnitude appears.

    species_np : numpy.ndarray
        Array with number of particles of each species.

    no_dumps : int
        Number of dumps.

    Returns
    -------

    Sk_all : ndarray
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
        Shape = ( ``no_species``, ``no_k_list``, ``no_dumps``)

    ka_list : list
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``ka_counts``.
        Shape=(`no_ka_values`, 5)

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
    Skw = np.empty((no_skw, len(ka_counts), no_dumps))

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
    Js : ndarray
        Electric current of each species. Shape = (``no_species``, ``no_dim``, ``no_dumps``)

    Jtot : ndarray
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
    moments: ndarray
        Velocity moments of each species per direction at each time step.

    no_dumps: int
        Number of saved timesteps.

    species_np: numpy.ndarray
        Number of particles of each species.

    Returns
    -------
    ratios: ndarray
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
    vel: ndarray
        Particles' velocity at each time step.

    nbins: int
        Number of bins to be used for the distribution.

    species_np: numpy.ndarray
        Number of particles of each species.

    Returns
    -------
    moments: ndarray
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
    pos_data : ndarray
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


def calc_nkt(fldr, no_dumps, dump_step, species_np, k_list):
    """
    Calculate density fluctuations :math:`n(k,t)` of all species.

    .. math::
        n_{A} ( k, t ) = \sum_i^{N_A} \exp [ -i \mathbf k \cdot \mathbf r_{i}(t) ]

    where :math:`N_A` is the number of particles of species :math:`A`.

    Parameters
    ----------
    fldr : str
        Name of folder containing particles data.

    no_dumps : int
        Number of saved timesteps.

    dump_step : int
        Timestep interval saving.

    species_np : numpy.ndarray
        Number of particles of each species.

    k_list : list
        List of :math: `k` vectors.

    Return
    ------
    nkt : ndarray, complex
        Density fluctuations.  Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)
    """
    # Read particles' position for all times
    print("Calculating n(k,t).")
    nkt = np.zeros((len(species_np), no_dumps, len(k_list)), dtype=np.complex128)
    for it in range(no_dumps):
        dump = int(it * dump_step)
        data = load_from_restart(fldr, dump)
        pos = data["pos"]
        sp_start = 0
        sp_end = 0
        for i, sp in enumerate(species_np):
            sp_end += sp
            nkt[i, it, :] = calc_nk(pos[sp_start:sp_end, :], k_list)
            sp_start = sp_end

    return nkt


@njit
def calc_pressure_tensor(pos, vel, acc, species_mass, species_np, box_volume):
    """
    Calculate the pressure tensor.

    Parameters
    ----------
    pos : ndarray
        Particles' positions.

    vel : ndarray
        Particles' velocities.

    acc : ndarray
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

    pressure_tensor : ndarray
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
def calc_vacf(vel, sp_num, sp_mass, time_averaging, it_skip):
    """
    Calculate the velocity autocorrelation function of each species and in each direction.

    Parameters
    ----------
    time_averaging: bool
        Flag for time averaging.

    it_skip: int
        Timestep interval for time averaging.

    vel : ndarray
        Particles' velocities.

    sp_num: numpy.ndarray
        Number of particles of each species.

    Returns
    -------
    vacf_x: ndarray
        x-velocity autocorrelation function

    vacf_y: ndarray
        y-velocity autocorrelation function

    vacf_z: ndarray
        z-velocity autocorrelation function

    vacf_tot: ndarray
        total velocity autocorrelation function
    """

    no_dim = vel.shape[0]
    no_dumps = vel.shape[2]
    no_species = len(sp_num)
    no_vacf = int(no_species * (no_species + 1) / 2)
    com_vel = np.zeros((no_species, 3, no_dumps))

    tot_mass_dens = np.sum(sp_num * sp_mass)

    sp_start = 0
    for i in range(no_species):
        sp_end = sp_start + sp_num[i]
        com_vel[i, :, :] = sp_mass[i] * np.sum(vel[:, sp_start: sp_end, :], axis=1) / tot_mass_dens
        sp_start = sp_end

    tot_com_vel = np.sum(com_vel, axis=0)

    jc_acf = np.zeros((no_vacf, no_dim + 1, no_dumps))

    if time_averaging:
        indx = 0
        for i in range(no_species):
            sp1_flux = sp_mass[i] * float(sp_num[i]) * (com_vel[i] - tot_com_vel)
            for j in range(i, no_species):
                sp2_flux = sp_mass[j] * float(sp_num[j]) * (com_vel[j] - tot_com_vel)
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
        indx = 0
        for i in range(no_species):
            sp1_flux = sp_mass[i] * float(sp_num[i]) * (com_vel[i] - tot_com_vel)
            for j in range(i, no_species):
                sp2_flux = sp_mass[j] * float(sp_num[j]) * (com_vel[j] - tot_com_vel)
                for d in range(no_dim):
                    jc_acf[indx, d, :] = correlationfunction_1D(sp1_flux[d, :], sp2_flux[d, :])

                jc_acf[indx, d + 1, :] = correlationfunction(sp1_flux, sp2_flux)
                indx += 1

    return jc_acf


@njit
def calc_vacf_single(vel, sp_num, time_averaging, it_skip):
    """
    Calculate the velocity autocorrelation function of each species and in each direction.

    Parameters
    ----------
    time_averaging: bool
        Flag for time averaging.

    it_skip: int
        Timestep interval for time averaging.

    vel : ndarray
        Particles' velocities.

    sp_num: numpy.ndarray
        Number of particles of each species.

    Returns
    -------
    vacf: numpy.ndarray
        Velocity autocorrelation functions.

    """
    no_dim = vel.shape[0]
    no_dumps = vel.shape[2]

    vacf = np.zeros((1, no_dim + 1, no_dumps))

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
        # Calculate species mass density flux
        for i in range(3):
            vacf_temp = np.zeros(no_dumps)
            for ptcl in range(sp_num[0]):
                vacf += autocorrelationfunction_1D(vel[i, ptcl, :])
            vacf[0, i, :] = vacf_temp / sp_num[0]

        vacf_temp = np.zeros(no_dumps)
        for ptcl in range(sp_num[0]):
            vacf_temp += autocorrelationfunction(vel[:, ptcl, :])

        vacf[0, -1, :] = vacf_temp / sp_num[0]

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


def calc_vkt(fldr, no_dumps, dump_step, species_np, k_list):
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

    no_dumps : int
        Number of saved timesteps.

    dump_step : int
        Timestep interval saving.

    species_np : numpy.ndarray
        Number of particles of each species.

    k_list : list
        List of :math: `k` vectors.

    Returns
    -------
    vkt : ndarray, complex
        Longitudinal velocity fluctuations.
        Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    vkt_perp_i : ndarray, complex
        Transverse velocity fluctuations along the :math:`x` axis.
        Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    vkt_perp_j : ndarray, complex
        Transverse velocity fluctuations along the :math:`y` axis.
        Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    vkt_perp_k : ndarray, complex
        Transverse velocity fluctuations along the :math:`z` axis.
        Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    """
    # Read particles' position for all times
    print("Calculating longitudinal and transverse microscopic velocity fluctuations v(k,t).")
    vkt_par = np.zeros((len(species_np), no_dumps, len(k_list)), dtype=np.complex128)
    vkt_perp_i = np.zeros((len(species_np), no_dumps, len(k_list)), dtype=np.complex128)
    vkt_perp_j = np.zeros((len(species_np), no_dumps, len(k_list)), dtype=np.complex128)
    vkt_perp_k = np.zeros((len(species_np), no_dumps, len(k_list)), dtype=np.complex128)
    for it in range(no_dumps):
        dump = int(it * dump_step)
        data = load_from_restart(fldr, dump)
        pos = data["pos"]
        vel = data["vel"]
        sp_start = 0
        sp_end = 0
        for i, sp in enumerate(species_np):
            sp_end += sp
            vkt_par[i, it, :], vkt_perp_i[i, it, :], vkt_perp_j[i, it, :], vkt_perp_k[i, it, :] = calc_vk(
                pos[sp_start:sp_end, :], vel[sp_start:sp_end], k_list)
            sp_start = sp_end

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
    At : ndarray
        Observable to correlate. Shape=(``no_dim``, ``no_steps``).

    Bt : ndarray
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


def kspace_setup(no_ka, box_lengths):
    """
    Calculate all allowed :math:`k` vectors.

    Parameters
    ----------
    no_ka : numpy.ndarray
        Number of harmonics in each direction.

    box_lengths : numpy.ndarray
        Length of each box's side.

    Returns
    -------
    k_arr : list
        List of all possible :math:`k` vectors with their corresponding magnitudes and indexes.

    k_counts : numpy.ndarray
        Number of occurrences of each :math:`k` magnitude.

    k_unique : numpy.ndarray
        Magnitude of each allowed :math:`k` vector.
    """
    # Obtain all possible permutations of the wave number arrays
    k_arr = [np.array([i / box_lengths[0], j / box_lengths[1], k / box_lengths[2]]) for i in range(no_ka[0])
             for j in range(no_ka[1])
             for k in range(no_ka[2])]

    # Compute wave number magnitude - don't use |k| (skipping first entry in k_arr)
    k_mag = np.sqrt(np.sum(np.array(k_arr) ** 2, axis=1)[..., None])

    # Add magnitude to wave number array
    k_arr = np.concatenate((k_arr, k_mag), 1)

    # Sort from lowest to highest magnitude
    ind = np.argsort(k_arr[:, -1])
    k_arr = k_arr[ind]

    # Count how many times a |k| value appears
    k_unique, k_counts = np.unique(k_arr[1:, -1], return_counts=True)

    # Generate a 1D array containing index to be used in S array
    k_index = np.repeat(range(len(k_counts)), k_counts)[..., None]

    # Add index to k_array
    k_arr = np.concatenate((k_arr[1:, :], k_index), 1)
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

    if units == 'cgs' and 'Length' == xlbl:
        xmax *= 1e2

    if units == 'cgs' and 'Length' == ylbl:
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
