"""
Module for calculating physical quantities from Sarkas checkpoints.
"""
import os
import yaml
from tqdm import tqdm
import numpy as np
from numba import njit, prange
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

plt.style.use(
    os.path.join(os.path.join(os.getcwd(), 'src'), 'PUBstyle'))

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


class CurrentCorrelationFunctions:
    """
    Current Correlation Functions: :math:`L(k,\omega) \quad T(k,\omega)`.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius.

        wp : float
            Total plasma frequency.

        dump_step : int
            Dump step frequency.

        dataframe_l : Pandas dataframe
            Dataframe of the longitudinal velocity correlation functions.

        dataframe_t : Pandas dataframe
            Dataframe of the transverse velocity correlation functions.

        l_filename_csv: str
            Name of file for the longitudinal velocities fluctuation correlation function.

        t_filename_csv: str
            Name of file for the transverse velocities fluctuation correlation function.

        fldr : str
            Jod directory.

        no_dumps : int
            Number of dumps.

        no_species : int
            Number of species.

        species_np: array
            Array of integers with the number of particles for each species.

        dt : float
            Timestep's value normalized by the total plasma frequency.

        species_names : list
            Names of particle species.

        species_wp : array
            Plasma frequency of each species.

        tot_no_ptcls : int
            Total number of particles.

        ptcls_fldr : str
            Directory of Sarkas dumps.

        k_fldr : str
            Directory of :math:`k`-space fluctuations.

        vkt_file : str
            Name of file containing velocity fluctuations functions of each species.

        k_file : str
            Name of file containing ``k_list``, ``k_counts``, ``ka_values``.

        k_list : list
            List of all possible :math:`k` vectors with their corresponding magnitudes and indexes.

        k_counts : array
            Number of occurrences of each :math:`k` magnitude.

        ka_values : array
            Magnitude of each allowed :math:`ka` vector.

        no_ka_values: int
            Length of ``ka_values`` array.

        box_lengths : array
            Length of each box side.
        """

    def __init__(self, params):

        self.fldr = params.Control.checkpoint_dir
        self.ptcls_fldr = params.Control.dump_dir
        self.k_fldr = os.path.join(self.fldr, "k_space_data")
        self.k_file = os.path.join(self.k_fldr, "k_arrays.npz")
        self.vkt_file = os.path.join(self.k_fldr, "vkt.npz")
        self.fname_app = params.Control.fname_app
        self.l_filename_csv = os.path.join(self.fldr,
                                           "LongitudinalVelocityCorrelationFunction_" + self.fname_app + '.csv')
        self.t_filename_csv = os.path.join(self.fldr,
                                           "TransverseVelocityCorrelationFunction_" + self.fname_app + '.csv')

        self.box_lengths = np.array([params.Lx, params.Ly, params.Lz])
        self.dump_step = params.Control.dump_step
        self.no_dumps = len(os.listdir(params.Control.dump_dir))
        self.no_species = len(params.species)
        self.tot_no_ptcls = params.total_num_ptcls

        self.species_np = np.zeros(self.no_species, dtype=int)
        self.species_names = []
        self.species_wp = np.zeros(self.no_species)
        for i in range(self.no_species):
            self.species_wp[i] = params.species[i].wp
            self.species_np[i] = int(params.species[i].num)
            self.species_names.append(params.species[i].name)

        self.dt = params.Control.dt
        self.a_ws = params.aws
        self.wp = params.wp

        # Create the lists of k vectors
        if len(params.PostProcessing.dsf_no_ka_values) == 0:
            self.no_ka = np.array([params.PostProcessing.dsf_no_ka_values,
                                   params.PostProcessing.dsf_no_ka_values,
                                   params.PostProcessing.dsf_no_ka_values], dtype=int)
        else:
            self.no_ka = params.PostProcessing.dsf_no_ka_values  # number of ka values

    def parse(self):
        """
        Read the Radial distribution function from the saved csv file.
        """
        try:
            self.dataframe_l = pd.read_csv(self.l_filename_csv, index_col=False)
            self.dataframe_t = pd.read_csv(self.t_filename_csv, index_col=False)
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]

        except FileNotFoundError:
            print("\nFiles not found!")
            print("\nComputing CCF now")
            self.compute()
        return

    def compute(self):
        """
        Calculate the velocity fluctuations correlation functions.
        """

        data = {"Frequencies": 2.0 * np.pi * np.fft.fftfreq(self.no_dumps, self.dt * self.dump_step)}
        data2 = {"Frequencies": 2.0 * np.pi * np.fft.fftfreq(self.no_dumps, self.dt * self.dump_step)}
        self.dataframe_l = pd.DataFrame(data)
        self.dataframe_t = pd.DataFrame(data2)
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
            self.k_list, self.k_counts, k_unique = kspace_setup(self.no_ka, self.box_lengths)
            self.ka_values = 2.0 * np.pi * k_unique * self.a_ws
            self.no_ka_values = len(self.ka_values)

            if not (os.path.exists(self.k_fldr)):
                os.mkdir(self.k_fldr)

            np.savez(self.k_file,
                     k_list=self.k_list,
                     k_counts=self.k_counts,
                     ka_values=self.ka_values)

            vkt, vkt_i, vkt_j, vkt_k = calc_vkt(self.ptcls_fldr, self.no_dumps, self.dump_step, self.species_np,
                                                self.k_list)
            np.savez(self.vkt_file,
                     longitudinal=vkt,
                     transverse_i=vkt_i,
                     transverse_j=vkt_j,
                     transverse_k=vkt_k)

        # Calculate Lkw
        Lkw = calc_Skw(vkt, self.k_list, self.k_counts, self.species_np, self.no_dumps, self.dt, self.dump_step)
        Tkw_i = calc_Skw(vkt_i, self.k_list, self.k_counts, self.species_np, self.no_dumps, self.dt, self.dump_step)
        Tkw_j = calc_Skw(vkt_j, self.k_list, self.k_counts, self.species_np, self.no_dumps, self.dt, self.dump_step)
        Tkw_k = calc_Skw(vkt_k, self.k_list, self.k_counts, self.species_np, self.no_dumps, self.dt, self.dump_step)
        Tkw = (Tkw_i + Tkw_j + Tkw_k) / 3.0
        print("Saving L(k,w) and T(k,w)")
        sp_indx = 0
        for sp_i in range(self.no_species):
            for sp_j in range(sp_i, self.no_species):
                for ik in range(len(self.k_counts)):
                    if ik == 0:
                        column = "{}-{} CCF ka_min".format(self.species_names[sp_i], self.species_names[sp_j])
                    else:
                        column = "{}-{} CCF {} ka_min".format(self.species_names[sp_i],
                                                              self.species_names[sp_j], ik + 1)

                    self.dataframe_l[column] = Lkw[sp_indx, ik, :]
                    self.dataframe_t[column] = Tkw[sp_indx, ik, :]
                sp_indx += 1

        self.dataframe_l.to_csv(self.l_filename_csv, index=False, encoding='utf-8')
        self.dataframe_t.to_csv(self.t_filename_csv, index=False, encoding='utf-8')

        return

    def plot(self, longitudinal=True, show=False, dispersion=False):
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
                self.dataframe = pd.read_csv(self.l_filename_csv, index_col=False)
            else:
                self.dataframe = pd.read_csv(self.t_filename_csv, index_col=False)
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]
            self.no_ka_values = len(self.ka_values)
        except FileNotFoundError:
            print("Computing L(k,w), T(k,w)")
            self.compute()

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        if self.no_species > 1:
            for sp_i in range(self.no_species):
                for sp_j in range(sp_i, self.no_species):
                    column = "{}-{} CCF ka_min".format(self.species_names[sp_i], self.species_names[sp_j])
                    ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.species_wp[0],
                            np.fft.fftshift(self.dataframe[column]),
                            label=r'$S_{' + self.species_names[sp_i] + self.species_names[sp_j] + '}(k,\omega)$')
        else:
            column = "{}-{} CCF ka_min".format(self.species_names[0], self.species_names[0])
            ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.species_wp[0],
                    np.fft.fftshift(self.dataframe[column]),
                    label=r'$ka = {:1.4f}$'.format(self.ka_values[0]))
            for i in range(1, 5):
                column = "{}-{} CCF {} ka_min".format(self.species_names[0], self.species_names[0], i + 1)
                ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.wp,
                        np.fft.fftshift(self.dataframe[column]),
                        label=r'$ka = {:1.4f}$'.format(self.ka_values[i]))

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=3)
        ax.set_yscale('log')
        ax.set_xlim(0, 3)
        if longitudinal:
            ax.set_ylabel(r'$L(k,\omega)$')
            fig_name = os.path.join(self.fldr, 'Lkw_' + self.fname_app + '.png')
        else:
            ax.set_ylabel(r'$T(k,\omega)$')
            fig_name = os.path.join(self.fldr, 'Tkw_' + self.fname_app + '.png')

        ax.set_xlabel(r'$\omega/\omega_p$')
        fig.tight_layout(fig_name)
        fig.savefig()
        if show:
            fig.show()

        if dispersion:
            w_array = np.array(self.dataframe["Frequencies"]) / self.wp
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
                fig.savefig(os.path.join(self.fldr, 'Lkw_Dispersion_' + self.fname_app + '.png'))
            else:
                fig.savefig(os.path.join(self.fldr, 'Tkw_Dispersion_' + self.fname_app + '.png'))
            if show:
                fig.show()


class DynamicStructureFactor:
    """
    Dynamic Structure factor.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius.

        wp : float
            Total plasma frequency.

        dump_step : int
            Dump step frequency.

        dataframe : Pandas dataframe
            Dataframe of the dynamic structure functions.

        filename_csv: str
            Filename in which to store the Dynamic structure functions.

        fldr : str
            Jod directory.

        no_dumps : int
            Number of dumps.

        no_species : int
            Number of species.

        species_np: array
            Array of integers with the number of particles for each species.

        dt : float
            Timestep's value normalized by the total plasma frequency.

        species_names : list
            Names of particle species.

        species_wp : array
            Plasma frequency of each species.

        tot_no_ptcls : int
            Total number of particles.

        ptcls_fldr : str
            Directory of Sarkas dumps.

        k_fldr : str
            Directory of :math:`k`-space fluctuations.

        nkt_file : str
            Name of file containing density fluctuations functions of each species.

        k_file : str
            Name of file containing ``k_list``, ``k_counts``, ``ka_values``.

        k_list : list
            List of all possible :math:`k` vectors with their corresponding magnitudes and indexes.

        k_counts : array
            Number of occurrences of each :math:`k` magnitude.

        ka_values : array
            Magnitude of each allowed :math:`ka` vector.

        no_ka_values: int
            Length of ``ka_values`` array.

        box_lengths : array
            Length of each box side.
        """

    def __init__(self, params):

        self.fldr = params.Control.checkpoint_dir
        self.ptcls_fldr = params.Control.dump_dir
        self.k_fldr = os.path.join(self.fldr, "k_space_data")
        self.k_file = os.path.join(self.k_fldr, "k_arrays.npz")
        self.nkt_file = os.path.join(self.k_fldr, "nkt.npy")
        self.fname_app = params.Control.fname_app
        self.filename_csv = os.path.join(self.fldr, "DynamicStructureFactor_" + self.fname_app + '.csv')

        self.box_lengths = np.array([params.Lx, params.Ly, params.Lz])
        self.dump_step = params.Control.dump_step
        self.no_dumps = len(os.listdir(params.Control.dump_dir))
        self.no_species = len(params.species)
        self.tot_no_ptcls = params.total_num_ptcls

        self.species_np = np.zeros(self.no_species, dtype=int)
        self.species_names = []
        self.species_wp = np.zeros(self.no_species)
        for i in range(self.no_species):
            self.species_wp[i] = params.species[i].wp
            self.species_np[i] = int(params.species[i].num)
            self.species_names.append(params.species[i].name)

        self.Nsteps = params.Control.Nsteps
        self.dt = params.Control.dt
        self.no_Skw = int(self.no_species * (self.no_species + 1) / 2)
        self.a_ws = params.aws
        self.wp = params.wp

        # Create the lists of k vectors
        if len(params.PostProcessing.dsf_no_ka_values) == 0:
            self.no_ka = np.array([params.PostProcessing.dsf_no_ka_values,
                                   params.PostProcessing.dsf_no_ka_values,
                                   params.PostProcessing.dsf_no_ka_values], dtype=int)
        else:
            self.no_ka = params.PostProcessing.dsf_no_ka_values  # number of ka values

    def parse(self):
        """
        Read the Radial distribution function from the saved csv file.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]

        except FileNotFoundError:
            print("\nFile {} not found!".format(self.filename_csv))
            print("\nComputing DSF now")
            self.compute()
        return

    def compute(self):
        """
        Compute :math:`S_{ij}(k,\omega)' and the array of :math:`\omega/\omega_p` values.
        ``self.Skw``. Shape = (``no_ws``, ``no_Sij``)
        """

        data = {"Frequencies": 2.0 * np.pi * np.fft.fftfreq(self.no_dumps, self.dt * self.dump_step)}
        self.dataframe = pd.DataFrame(data)

        # Parse nkt otherwise calculate it
        try:
            nkt = np.load(self.nkt_file)
            k_data = np.load(self.k_file)
            self.k_list = k_data["k_list"]
            self.k_counts = k_data["k_counts"]
            self.ka_values = k_data["ka_values"]
            self.no_ka_values = len(self.ka_values)
            print("Loaded")
            print(nkt.shape)
        except FileNotFoundError:
            self.k_list, self.k_counts, k_unique = kspace_setup(self.no_ka, self.box_lengths)
            self.ka_values = 2.0 * np.pi * k_unique * self.a_ws
            self.no_ka_values = len(self.ka_values)

            if not (os.path.exists(self.k_fldr)):
                os.mkdir(self.k_fldr)

            np.savez(self.k_file,
                     k_list=self.k_list,
                     k_counts=self.k_counts,
                     ka_values=self.ka_values)

            nkt = calc_nkt(self.ptcls_fldr, self.no_dumps, self.dump_step, self.species_np, self.k_list)
            np.save(self.nkt_file, nkt)

        # Calculate Skw
        Skw = calc_Skw(nkt, self.k_list, self.k_counts, self.species_np, self.no_dumps, self.dt, self.dump_step)
        print("Saving S(k,w)")
        sp_indx = 0
        for sp_i in range(self.no_species):
            for sp_j in range(sp_i, self.no_species):
                for ik in range(len(self.k_counts)):
                    if ik == 0:
                        column = "{}-{} DSF ka_min".format(self.species_names[sp_i], self.species_names[sp_j])
                    else:
                        column = "{}-{} DSF {} ka_min".format(self.species_names[sp_i],
                                                              self.species_names[sp_j], ik + 1)
                    self.dataframe[column] = Skw[sp_indx, ik, :]
                sp_indx += 1

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

        return

    def plot(self, show=False, dispersion=False):
        """
        Plot :math: `S(k,\omega)` and save the figure.
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

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        if self.no_species > 1:
            for sp_i in range(self.no_species):
                for sp_j in range(sp_i, self.no_species):
                    column = "{}-{} DSF ka_min".format(self.species_names[sp_i], self.species_names[sp_j])
                    ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.species_wp[0],
                            np.fft.fftshift(self.dataframe[column]),
                            label=r'$S_{' + self.species_names[sp_i] + self.species_names[sp_j] + '}(k,\omega)$')
        else:
            column = "{}-{} DSF ka_min".format(self.species_names[0], self.species_names[0])
            ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.species_wp[0],
                    np.fft.fftshift(self.dataframe[column]),
                    label=r'$ka = {:1.4f}$'.format(self.ka_values[0]))
            for i in range(1, 5):
                column = "{}-{} DSF {} ka_min".format(self.species_names[0], self.species_names[0], i + 1)
                ax.plot(np.fft.fftshift(self.dataframe["Frequencies"]) / self.wp,
                        np.fft.fftshift(self.dataframe[column]),
                        label=r'$ka = {:1.4f}$'.format(self.ka_values[i]))

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', ncol=3)
        ax.set_yscale('log')
        ax.set_xlim(0, 3)
        ax.set_ylabel(r'$S(k,\omega)$')
        ax.set_xlabel(r'$\omega/\omega_p$')
        fig.tight_layout()
        fig.savefig(os.path.join(self.fldr, 'Skw_' + self.fname_app + '.png'))
        if show:
            fig.show()

        if dispersion:
            w_array = np.array(self.dataframe["Frequencies"]) / self.wp
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
            plt.title("$S(k, \omega)$")
            fig.tight_layout()
            fig.savefig(os.path.join(self.fldr, 'Skw_Dispersion_' + self.fname_app + '.png'))
            if show:
                fig.show()


class ElectricCurrent:
    """
    Electric Current Auto-correlation function.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius.

        wp : float
            Total plasma frequency.

        dump_step : int
            Dump step frequency.

        dt : float
            Timestep magnitude.

        filename: str
            Name of output files.

        fldr : str
            Folder containing dumps.

        no_dumps : int
            Number of dumps.

        no_species : int
            Number of species.

        species_np: array
            Array of integers with the number of particles for each species.

        species_charge: array
            Array of with the charge of each species.

        species_names : list
            Names of particle species.

        tot_no_ptcls : int
            Total number of particles.
    """

    def __init__(self, params):
        self.fldr = params.Control.checkpoint_dir
        self.fname_app = params.Control.fname_app
        self.units = params.Control.units
        self.dump_dir = params.Control.dump_dir
        self.filename_csv = os.path.join(self.fldr, "ElectricCurrent_" + self.fname_app + '.csv')
        self.dump_step = params.Control.dump_step
        self.no_dumps = len(os.listdir(params.Control.dump_dir))
        self.no_species = len(params.species)
        self.species_np = np.zeros(self.no_species, dtype=int)
        self.species_names = []
        self.dt = params.Control.dt  # No of dump to skip
        self.species_charge = np.zeros(self.no_species)
        for i in range(self.no_species):
            self.species_np[i] = int(params.species[i].num)
            self.species_charge[i] = params.species[i].charge
            self.species_names.append(params.species[i].name)

        self.tot_no_ptcls = params.total_num_ptcls
        self.wp = params.wp
        self.a_ws = params.aws
        self.dt = params.Control.dt

    def parse(self):
        """
        Parse Electric functions from csv file if found otherwise compute them.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            data = {"Time": self.time}
            self.dataframe = pd.DataFrame(data)
            self.compute()

    def compute(self):
        """
        Compute the electric current and the corresponding auto-correlation functions.
        """

        # Parse the particles from the dump files
        vel = np.zeros((self.no_dumps, 3, self.tot_no_ptcls))
        #
        print("Parsing particles' velocities.")
        time = np.zeros(self.no_dumps)
        for it in tqdm(range(self.no_dumps)):
            dump = int(it * self.dump_step)
            time[it] = dump * self.dt
            datap = load_from_restart(self.dump_dir, dump)
            vel[it, 0, :] = datap["vel"][:, 0]
            vel[it, 1, :] = datap["vel"][:, 1]
            vel[it, 2, :] = datap["vel"][:, 2]
        #
        print("Calculating Electric current quantities.")
        species_current, total_current = calc_elec_current(vel, self.species_charge, self.species_np)
        data_dic = {"Time": time}
        self.dataframe = pd.DataFrame(data_dic)

        self.dataframe["Total Current X"] = total_current[0, :]
        self.dataframe["Total Current Y"] = total_current[1, :]
        self.dataframe["Total Current Z"] = total_current[2, :]

        cur_acf_xx = autocorrelationfunction_1D(total_current[0, :])
        cur_acf_yy = autocorrelationfunction_1D(total_current[1, :])
        cur_acf_zz = autocorrelationfunction_1D(total_current[2, :])

        tot_cur_acf = autocorrelationfunction(total_current)
        # Normalize and save
        self.dataframe["X Current ACF"] = cur_acf_xx / cur_acf_xx[0]
        self.dataframe["Y Current ACF"] = cur_acf_yy / cur_acf_yy[0]
        self.dataframe["Z Current ACF"] = cur_acf_zz / cur_acf_zz[0]
        self.dataframe["Total Current ACF"] = tot_cur_acf / tot_cur_acf[0]
        for sp in range(self.no_species):
            tot_acf = autocorrelationfunction(species_current[sp, :, :])
            acf_xx = autocorrelationfunction_1D(species_current[sp, 0, :])
            acf_yy = autocorrelationfunction_1D(species_current[sp, 1, :])
            acf_zz = autocorrelationfunction_1D(species_current[sp, 2, :])

            self.dataframe["{} Total Current".format(self.species_names[sp])] = np.sqrt(
                species_current[sp, 0, :] ** 2 + species_current[sp, 1, :] ** 2 + species_current[sp, 2, :] ** 2)
            self.dataframe["{} X Current".format(self.species_names[sp])] = species_current[sp, 0, :]
            self.dataframe["{} Y Current".format(self.species_names[sp])] = species_current[sp, 1, :]
            self.dataframe["{} Z Current".format(self.species_names[sp])] = species_current[sp, 2, :]

            self.dataframe["{} Total Current ACF".format(self.species_names[sp])] = tot_acf / tot_acf[0]
            self.dataframe["{} X Current ACF".format(self.species_names[sp])] = acf_xx / acf_xx[0]
            self.dataframe["{} Y Current ACF".format(self.species_names[sp])] = acf_yy / acf_yy[0]
            self.dataframe["{} Z Current ACF".format(self.species_names[sp])] = acf_zz / acf_zz[0]

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')
        return

    def plot(self, show=False):
        """
        Plot the electric current autocorrelation function and save the figure.
        """

        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            self.compute()

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(self.dataframe["Time"],
                                                               self.dataframe["Total Current ACF"], "Time", "none",
                                                               self.units)
        ax.plot(xmul * self.dataframe["Time"], self.dataframe["Total Current ACF"], '--o', label=r'$J_{tot} (t)$')

        ax.legend(loc='upper right')
        ax.set_ylabel(r'$J(t)$')
        ax.set_xlabel('Time' + xlbl)
        ax.set_xscale('log')
        fig.tight_layout()
        fig.savefig(os.path.join(self.fldr, 'TotalCurrentACF_' + self.fname_app + '.png'))
        if show:
            fig.show()


class RadialDistributionFunction:
    """
    Radial Distribution Function.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius.

        box_lengths : array
            Length of each side of the box.

        box_volume : float
            Volume of simulation's box.

        dataframe : Pandas dataframe
            It contains the radial distribution functions.

        dump_step : int
            Dump step frequency.

        filename_csv: str
            Name of csv file containing the radial distribution functions.

        fname_app: str
            Appendix of file names.

        fldr : str
            Folder containing dumps.

        no_bins : int
            Number of bins.

        no_dumps : int
            Number of dumps.

        no_grs : int
            Number of :math:`g_{ij}(r)` pairs.

        no_species : int
            Number of species.

        no_steps : int
            Total number of steps for which the RDF has been calculated.

        species_np: array
            Array of integers with the number of particles for each species.

        species_names : list
            Names of particle species.

        tot_no_ptcls : int
            Total number of particles.

        dr_rdf : float
            Size of each bin.
    """

    def __init__(self, params):
        self.no_bins = params.PostProcessing.rdf_nbins  # number of ka values
        self.fldr = params.Control.checkpoint_dir
        self.fname_app = params.Control.fname_app
        self.filename_csv = os.path.join(self.fldr, "RadialDistributionFunction_" + params.Control.fname_app + ".csv")
        self.dump_step = params.Control.dump_step
        self.no_dumps = len(os.listdir(params.Control.dump_dir))
        self.no_species = len(params.species)
        self.no_grs = int(params.num_species * (params.num_species + 1) / 2)
        self.tot_no_ptcls = params.total_num_ptcls

        self.no_steps = params.Control.Nsteps
        self.a_ws = params.aws
        self.dr_rdf = params.Potential.rc / self.no_bins / self.a_ws
        self.box_volume = params.box_volume / self.a_ws ** 3
        self.box_lengths = np.array([params.Lx / params.aws, params.Ly / params.aws, params.Lz / params.aws])
        self.species_np = np.zeros(self.no_species)  # Number of particles of each species
        self.species_names = []

        for i in range(self.no_species):
            self.species_np[i] = params.species[i].num
            self.species_names.append(params.species[i].name)

    def save(self, rdf_hist):
        """
        Parameters
        ----------
        rdf_hist : array
            Histogram of the radial distribution function.

        """
        # Initialize all the workhorse arrays
        ra_values = np.zeros(self.no_bins)
        bin_vol = np.zeros(self.no_bins)
        pair_density = np.zeros((self.no_species, self.no_species))
        gr = np.zeros((self.no_bins, self.no_grs))

        # No. of pairs per volume
        for i in range(self.no_species):
            pair_density[i, i] = self.species_np[i] * (self.species_np[i] - 1) / (2.0 * self.box_volume)
            for j in range(i + 1, self.no_species):
                pair_density[i, j] = self.species_np[i] * self.species_np[j] / self.box_volume
        # Calculate each bin's volume
        sphere_shell_const = 4.0 * np.pi / 3.0
        bin_vol[0] = sphere_shell_const * self.dr_rdf ** 3
        for ir in range(1, self.no_bins):
            r1 = ir * self.dr_rdf
            r2 = (ir + 1) * self.dr_rdf
            bin_vol[ir] = sphere_shell_const * (r2 ** 3 - r1 ** 3)
            ra_values[ir] = (ir + 0.5) * self.dr_rdf

        data = {"ra values": ra_values}
        self.dataframe = pd.DataFrame(data)

        gr_ij = 0
        for i in range(self.no_species):
            for j in range(i, self.no_species):
                if j == i:
                    pair_density[i, j] *= 2.0
                for ibin in range(self.no_bins):
                    gr[ibin, gr_ij] = (rdf_hist[ibin, i, j] + rdf_hist[ibin, j, i]) / (bin_vol[ibin]
                                                                                       * pair_density[i, j]
                                                                                       * self.no_steps)

                self.dataframe['{}-{} RDF'.format(self.species_names[i], self.species_names[j])] = gr[:, gr_ij]
                gr_ij += 1

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

        return

    def parse(self):
        """
        Read the Radial distribution function from the saved csv file.
        """
        self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        return

    def plot(self, show=False):
        """
        Plot :math: `g_{ij}(r)` and save the figure.

        Parameters
        ----------
        show : bool
            Flag for prompting the plot to screen. Default=False
        """
        self.dataframe = pd.read_csv(self.filename_csv, index_col=False)

        indx = 0
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for i in range(self.no_species):
            for j in range(i, self.no_species):
                subscript = self.species_names[i] + self.species_names[j]
                ax.plot(self.dataframe["ra values"],
                        self.dataframe["{}-{} RDF".format(self.species_names[i], self.species_names[j])],
                        label=r'$g_{' + subscript + '} (r)$')
                indx += 1
        ax.grid(True, alpha=0.3)
        if self.no_species > 2:
            ax.legend(loc='best', ncol=(self.no_species - 1))
        else:
            ax.legend(loc='best')

        ax.set_ylabel(r'$g(r)$')
        ax.set_xlabel(r'$r/a$')
        # ax.set_ylim(0, 5)
        fig.tight_layout()
        fig.savefig(os.path.join(self.fldr, 'RDF_' + self.fname_app + '.png'))
        if show:
            fig.show()
        return


class StaticStructureFactor:
    """ Static Structure Factors :math:`S_{ij}(k)`.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius.

        box_lengths : array
            Array with box length in each direction.

        dataframe : dict
            Pandas dataframe. It contains all the :math:`S_{ij}(k)` and :math:`ka`.

        dump_step : int
            Dump step frequency.

        filename_csv: str
            Name of output files.

        fname_app: str
            Appendix of filenames.

        fldr : str
            Folder containing dumps.

        no_dumps : int
            Number of dumps.

        no_species : int
            Number of species.

        no_Sk : int
            Number of :math: `S_{ij}(k)` pairs.

        species_np: array
            Array of integers with the number of particles for each species.

        species_names : list
            Names of particle species.

        tot_no_ptcls : int
            Total number of particles.

        ptcls_fldr : str
            Directory of Sarkas dumps.

        k_fldr : str
            Directory of :math:`k`-space fluctuations.

        nkt_file : str
            Name of file containing :math:`n(k,t)` of each species.

        k_file : str
            Name of file containing ``k_list``, ``k_counts``, ``ka_values``.

        k_list : list
            List of all possible :math:`k` vectors with their corresponding magnitudes and indexes.

        k_counts : array
            Number of occurrences of each :math:`k` magnitude.

        ka_values : array
            Magnitude of each allowed :math:`ka` vector.

        no_ka_values: int
            Length of ``ka_values`` array.
        """

    def __init__(self, params):

        self.fldr = params.Control.checkpoint_dir
        self.fname_app = params.Control.fname_app
        self.ptcls_fldr = params.Control.dump_dir
        self.k_fldr = os.path.join(self.fldr, "k_space_data")
        self.k_file = os.path.join(self.k_fldr, "k_arrays.npz")
        self.nkt_file = os.path.join(self.k_fldr, "nkt.npy")

        self.filename_csv = os.path.join(self.fldr, "StaticStructureFunction_" + self.fname_app + ".csv")
        self.dump_step = params.Control.dump_step
        self.no_dumps = len(os.listdir(params.Control.dump_dir))
        self.no_species = len(params.species)
        self.tot_no_ptcls = params.total_num_ptcls

        if len(params.PostProcessing.ssf_no_ka_values) == 0:
            self.no_ka = np.array([params.PostProcessing.ssf_no_ka_values,
                                   params.PostProcessing.ssf_no_ka_values,
                                   params.PostProcessing.ssf_no_ka_values], dtype=int)
        else:
            self.no_ka = params.PostProcessing.ssf_no_ka_values  # number of ka values

        self.no_Sk = int(self.no_species * (self.no_species + 1) / 2)
        self.a_ws = params.aws
        self.box_lengths = np.array([params.Lx, params.Ly, params.Lz])
        self.species_np = np.zeros(self.no_species, dtype=int)
        self.species_names = []

        for i in range(self.no_species):
            self.species_np[i] = params.species[i].num
            self.species_names.append(params.species[i].name)

    def parse(self):
        """
        Read the Radial distribution function from the saved csv file.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            print("\nError: {} not found!".format(self.filename_csv))
        return

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
            self.k_list, self.k_counts, k_unique = kspace_setup(self.no_ka, self.box_lengths)
            self.ka_values = 2.0 * np.pi * k_unique * self.a_ws
            self.no_ka_values = len(self.ka_values)

            if not (os.path.exists(self.k_fldr)):
                os.mkdir(self.k_fldr)

            np.savez(self.k_file,
                     k_list=self.k_list,
                     k_counts=self.k_counts,
                     ka_values=self.ka_values)

            nkt = calc_nkt(self.ptcls_fldr, self.no_dumps, self.dump_step, self.species_np, self.k_list)
            np.save(self.nkt_file, nkt)

        data = {"ka values": self.ka_values}
        self.dataframe = pd.DataFrame(data)

        print("Calculating S(k) ...")
        Sk_all = calc_Sk(nkt, self.k_list, self.k_counts, self.species_np, self.no_dumps)
        Sk = np.mean(Sk_all, axis=-1)
        Sk_err = np.std(Sk_all, axis=-1)

        sp_indx = 0
        for sp_i in range(self.no_species):
            for sp_j in range(sp_i, self.no_species):
                column = "{}-{} SSF".format(self.species_names[sp_i], self.species_names[sp_j])
                err_column = "{}-{} SSF Errorbar".format(self.species_names[sp_i], self.species_names[sp_j])
                self.dataframe[column] = Sk[sp_indx, :]
                self.dataframe[err_column] = Sk_err[sp_indx, :]

                sp_indx += 1

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

        return

    def plot(self, errorbars=False, show=False):
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

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for i in range(self.no_species):
            for j in range(i, self.no_species):
                subscript = self.species_names[i] + self.species_names[j]
                if errorbars:
                    ax.errorbar(self.dataframe["ka values"],
                                self.dataframe["{}-{} SSF".format(self.species_names[i], self.species_names[j])],
                                yerr=self.dataframe[
                                    "{}-{} SSF Errorbar".format(self.species_names[i], self.species_names[j])],
                                ls='--', marker='o', label=r'$S_{ ' + subscript + '} (k)$')
                else:
                    ax.plot(self.dataframe["ka values"],
                            self.dataframe["{}-{} SSF".format(self.species_names[i], self.species_names[j])],
                            label=r'$S_{ ' + subscript + '} (k)$')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_ylabel(r'$S(k)$')
        ax.set_xlabel(r'$ka$')
        fig.tight_layout()
        fig.savefig(os.path.join(self.fldr, 'StaticStructureFactor' + self.fname_app + '.png'))
        if show:
            fig.show()


class Thermodynamics:
    """
    Thermodynamic functions.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius.

        box_volume: float
            Box Volume

        dataframe : pandas dataframe
            It contains all the thermodynamics functions.
            options: "Total Energy", "Potential Energy", "Kinetic Energy", "Temperature", "time", "Pressure",
                    "Pressure Tensor ACF", "Pressure Tensor", "Gamma", "{species name} Temperature",
                    "{species name} Kinetic Energy".

        dump_step : int
            Dump step frequency.

        filename_csv : str
            Name of csv output file.

        fldr : str
            Folder containing dumps.

        eV2K : float
            Conversion factor from eV to Kelvin.

        no_dim : int
            Number of non-zero dimensions.

        no_dumps : int
            Number of dumps.

        no_species : int
            Number of species.

        species_np: array
            Array of integers with the number of particles for each species.

        species_names : list
            Names of particle species.

        species_masses : list
            Names of particle species.

        tot_no_ptcls : int
            Total number of particles.

        wp : float
            Plasma frequency.

        kB : float
            Boltzmann constant.

    """

    def __init__(self, params):
        self.fldr = params.Control.checkpoint_dir
        self.fname_app = params.Control.fname_app
        self.dump_step = params.Control.dump_step
        self.no_dumps = len(os.listdir(params.Control.dump_dir))
        self.no_dim = params.dimensions
        self.units = params.Control.units
        self.dt = params.Control.dt
        self.potential = params.Potential.type
        self.thermostat = params.Thermostat.type
        self.thermostat_tau = params.Thermostat.tau
        self.F_err = params.P3M.F_err
        self.Nsteps = params.Control.Nsteps
        if params.load_method == "restart":
            self.restart_sim = True
        else:
            self.restart_sim = False
        self.box_lengths = params.Lv
        self.box_volume = params.box_volume
        self.tot_no_ptcls = params.total_num_ptcls

        self.no_species = len(params.species)
        self.species_np = np.zeros(self.no_species)
        self.species_names = []
        self.species_masses = np.zeros(self.no_species)
        self.species_dens = np.zeros(self.no_species)
        for i in range(self.no_species):
            self.species_np[i] = params.species[i].num
            self.species_names.append(params.species[i].name)
            self.species_masses[i] = params.species[i].mass
            self.species_dens[i] = params.species[i].num_density
        # Output file with Energy and Temperature
        self.filename_csv = os.path.join(self.fldr, "Thermodynamics_" + self.fname_app + '.csv')
        # Constants
        self.wp = params.wp
        self.kB = params.kB
        self.eV2K = params.eV2K
        self.a_ws = params.aws
        self.T = params.T_desired
        self.Gamma_eff = params.Potential.Gamma_eff

    def compute_pressure_quantities(self):
        """
        Calculate Pressure, Pressure Tensor, Pressure Tensor Auto Correlation Function.
        """
        pos = np.zeros((self.no_dim, self.tot_no_ptcls))
        vel = np.zeros((self.no_dim, self.tot_no_ptcls))
        acc = np.zeros((self.no_dim, self.tot_no_ptcls))

        pressure = np.zeros(self.no_dumps)
        pressure_tensor_temp = np.zeros((3, 3, self.no_dumps))

        # Collect particles' positions, velocities and accelerations
        for it in range(int(self.no_dumps)):
            dump = int(it * self.dump_step)

            data = load_from_restart(self.fldr, dump)
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
                                                                                self.species_np, self.box_volume)

        self.dataframe["Pressure"] = pressure

        if self.no_dim == 3:
            dim_lbl = ['X', 'Y', 'Z']

        # Calculate the acf of the pressure tensor
        for i in range(self.no_dim):
            for j in range(self.no_dim):
                self.dataframe["Pressure Tensor {}{}".format(dim_lbl[i], dim_lbl[j])] = pressure_tensor_temp[i, j, :]
                pressure_tensor_acf_temp = autocorrelationfunction_1D(pressure_tensor_temp[i, j, :])
                self.dataframe["Pressure Tensor ACF {}{}".format(dim_lbl[i], dim_lbl[j])] = pressure_tensor_acf_temp / \
                                                                                            pressure_tensor_acf_temp[0]

        # Save the pressure acf to file
        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

        return

    def compute_pressure_from_rdf(self, r, gr, potential, potential_matrix):
        """
        Calculate the Pressure using the radial distribution function

        Parameters
        ----------
        r : array
            Particles' distances.

        gr : array
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

    def plot(self, quantity="Total Energy", show=False):
        """
        Plot `quantity` vs time and save the figure with appropriate name.

        Parameters
        ----------
        show
        quantity : str
            Quantity to plot. Default = Total Energy.

        delta : bool
            Flag for plotting relative difference of `quantity`. Default = True.

        """
        self.dataframe = pd.read_csv(self.filename_csv, index_col=False)

        if quantity[:8] == "Pressure":
            if not "Pressure" in self.dataframe.columns:
                print("Calculating Pressure quantities ...")
                self.compute_pressure_quantities()

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ylbl = {}
        ylbl["Total Energy"] = r"$E_{tot}(t)$"
        ylbl["Kinetic Energy"] = r"$K_{tot}(t)$"
        ylbl["Potential Energy"] = r"$U_{tot}(t)$"
        ylbl["Temperature"] = r"$T(t)$"
        ylbl[
            "Pressure Tensor ACF"] = r'$P_{\alpha\beta} = \langle P_{\alpha\beta}(0)P_{\alpha\beta}(t)\rangle$'
        ylbl["Pressure Tensor"] = r"$P_{\alpha\beta}(t)$"
        ylbl["Gamma"] = r"$\Gamma(t)$"
        ylbl["Pressure"] = r"$P(t)$"
        dim_lbl = ['X', 'Y', 'Z']

        if quantity == "Pressure Tensor ACF":
            for i in range(self.no_dim):
                for j in range(self.no_dim):
                    ax.plot(self.dataframe["Time"] * self.wp,
                            self.dataframe["Pressure Tensor ACF {}{}".format(dim_lbl[i], dim_lbl[j])],
                            label=r'$P_{' + dim_lbl[i] + dim_lbl[j] + '} (t)$')
            ax.set_xscale('log')
            ax.legend(loc='best', ncol=3)
            ax.set_ylim(-1, 1.5)

        elif quantity == "Pressure Tensor":
            for i in range(self.no_dim):
                for j in range(self.no_dim):
                    ax.plot(self.dataframe["Time"] * self.wp,
                            self.dataframe["Pressure Tensor {}{}".format(dim_lbl[i], dim_lbl[j])],
                            label=r'$P_{' + dim_lbl[i] + dim_lbl[j] + '} (t)$')
            ax.set_xscale('log')
            ax.legend(loc='best', ncol=3)

        else:
            ax.plot(self.dataframe["Time"] * self.wp, self.dataframe[quantity])

        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylbl[quantity])
        ax.set_xlabel(r'$\omega_p t$')
        fig.tight_layout()
        fig.savefig(os.path.join(self.fldr, quantity + '_' + self.fname_app + '.png'))
        if show:
            fig.show()

    def parse(self):
        """
        Parse Thermodynamics functions from saved csv file.
        """
        self.dataframe = pd.read_csv(self.filename_csv, index_col=False)

    def statistics(self, quantity="Total Energy", max_no_divisions=100, show=True):

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
        fig.savefig(os.path.join(self.fldr, quantity + 'StatisticalEfficiency_' + self.fname_app + '.png'))

        if show:
            fig.show()

        return

    def boxplot(self, show=False):
        self.parse()

        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(4, 8)
        # quantity = "Temperature"
        self.no_dumps = len(self.dataframe["Time"])
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
        T_main_plot.axhline(ymul * self.T, ls='--', c='r', alpha=0.7, label='Desired T')

        Delta_T = (self.dataframe["Temperature"] - self.T) * 100 / self.T
        Delta_T_cum_avg = Delta_T.cumsum() / [i for i in range(1, self.no_dumps + 1)]
        T_delta_plot.plot(self.dataframe["Time"], Delta_T, alpha=0.5)
        T_delta_plot.plot(self.dataframe["Time"], Delta_T_cum_avg, alpha=0.8)

        T_delta_plot.get_xaxis().set_ticks([])
        T_delta_plot.set_ylabel(r'Deviation [%]')
        T_delta_plot.tick_params(labelsize=12)
        T_main_plot.tick_params(labelsize=14)
        T_main_plot.legend(loc='best')
        T_main_plot.set_ylabel("Temperature" + ylbl)
        T_main_plot.set_xlabel("Time" + xlbl)
        T_hist_plot.hist(self.dataframe['Temperature'], bins=nbins, density=True, orientation='horizontal',
                         alpha=0.75)
        T_hist_plot.get_xaxis().set_ticks([])
        T_hist_plot.get_yaxis().set_ticks([])
        T_hist_plot.set_xlim(T_hist_plot.get_xlim()[::-1])

        # Energy plots
        xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(self.dataframe["Time"],
                                                               self.dataframe["Total Energy"], "Time",
                                                               "Total Energy", self.units)
        E_cumavg = self.dataframe["Total Energy"].cumsum() / [i for i in range(1, self.no_dumps + 1)]

        E_main_plot.plot(xmul * self.dataframe["Time"], ymul * self.dataframe["Total Energy"], alpha=0.7)
        E_main_plot.plot(xmul * self.dataframe["Time"], ymul * E_cumavg, label='Cum Avg')
        E_main_plot.axhline(ymul * self.dataframe["Total Energy"].mean(), ls='--', c='r', alpha=0.7, label='Avg')

        Delta_E = (self.dataframe["Total Energy"] - self.dataframe["Total Energy"][0]) * 100 / \
                  self.dataframe["Total Energy"][0]
        Delta_E_cum_avg = Delta_E.cumsum() / [i for i in range(1, self.no_dumps + 1)]
        E_delta_plot.plot(self.dataframe["Time"], Delta_E, alpha=0.5)
        E_delta_plot.plot(self.dataframe["Time"], Delta_E_cum_avg, alpha=0.8)

        E_delta_plot.get_xaxis().set_ticks([])
        E_delta_plot.set_ylabel(r'Deviation [%]')
        E_delta_plot.tick_params(labelsize=12)
        E_main_plot.tick_params(labelsize=14)
        E_main_plot.legend(loc='best')
        E_main_plot.set_ylabel("Total Energy" + ylbl)
        E_main_plot.set_xlabel("Time" + xlbl)
        E_hist_plot.hist(xmul * self.dataframe['Total Energy'], bins=nbins, density=True,
                         orientation='horizontal', alpha=0.75)
        E_hist_plot.get_xaxis().set_ticks([])
        E_hist_plot.get_yaxis().set_ticks([])

        xmul, ymul, xprefix, yprefix, xlbl, ylbl = plot_labels(np.array([self.dt]),
                                                               self.dataframe["Temperature"], "Time",
                                                               "Temperature", self.units)
        Info_plot.axis([0, 10, 0, 10])
        Info_plot.grid(False)
        fsz = 14
        Info_plot.text(0., 10, "Job ID: {}".format(self.fname_app))
        Info_plot.text(0., 9.5, "No. of species = {}".format(len(self.species_np)), fontsize=fsz)
        y_coord = 9.0
        for isp, sp in enumerate(self.species_names):
            Info_plot.text(0., y_coord, "Species {} : {}".format(isp + 1, sp), fontsize=fsz)
            Info_plot.text(0.0, y_coord - 0.5, "  No. of particles = {} ".format(self.species_np[isp]), fontsize=fsz)
            Info_plot.text(0.0, y_coord - 1., "  Temperature = {:1.2f} {}".format(ymul * self.dataframe['{} Temperature'.format(sp)].iloc[-1],
                                                                                   ylbl), fontsize=fsz)
            y_coord -= 1.5

        y_coord -= 0.25
        Info_plot.text(0., y_coord, "Total $N$ = {}".format(self.species_np.sum()), fontsize=fsz)
        Info_plot.text(0., y_coord - 0.5, "Thermostat: {}".format(self.thermostat), fontsize=fsz)
        Info_plot.text(0., y_coord - 1., "Berendsen rate = {:1.2f}".format(1.0 / self.thermostat_tau), fontsize=fsz)
        Info_plot.text(0., y_coord - 1.5, "Potential: {}".format(self.potential), fontsize=fsz)
        Info_plot.text(0., y_coord - 2., "Tot Force Error = {:1.4e}".format(self.F_err), fontsize=fsz)

        Info_plot.text(0., y_coord - 2.5, "Timestep = {:1.4f} {}".format(xmul * self.dt, xlbl), fontsize=fsz)
        Info_plot.text(0., y_coord - 3.5, "{:1.2f} % Production Completed".format(
            100 * self.dump_step * self.no_dumps / self.Nsteps), fontsize=fsz)

        Info_plot.axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(self.fldr, 'EnsembleCheckPlot_' + self.fname_app + '.png'))
        if show:
            fig.show()


class TransportCoefficients:
    """
    Transport Coefficients class

    Attributes
    ----------
    params : class
        Simulation parameters.
    """

    def __init__(self, params):
        self.params = params
        self.transport_coefficients = {}
        return

    def compute(self, quantity="Electrical Conductivity", show=True):
        """
        Calculate the desired transport coefficient

        Parameters
        ----------
        show
        quantity: str
            Desired transport coefficient to calculate.

        tau: float
            Upper limit of time integration.

        Returns
        -------

        transport_coeff : float
            Desired transport coefficient value scaled by appropriate units

        """
        if quantity == "Electrical Conductivity":
            J = ElectricCurrent(self.params)
            J.plot(show=True)
            sigma = np.zeros(J.no_dumps)
            integrand = np.array(J.dataframe["Total Current ACF"])
            time = np.array(J.dataframe["Time"])
            for it in range(1, J.no_dumps):
                sigma[it] = np.trapz(integrand[:it], x=time[:it]) / 3.0
            self.transport_coefficients["Electrical Conductivity"] = sigma
            # Plot the transport coefficient at different integration times
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            ax.plot(time, sigma, label=r'$\sigma (t)$')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            ax.set_ylabel(r'$\sigma(t)$')
            ax.set_xlabel(r'$\omega_p t$')
            fig.tight_layout()
            fig.savefig(os.path.join(self.params.Control.checkpoint_dir,
                                     'ConductivityPlot_' + self.params.Control.fname_app + '.png'))
            if show:
                fig.show()

        elif quantity == "Diffusion":
            Z = VelocityAutocorrelationFunctions(self.params)
            Z.plot(show=True)
            no_int = int(self.params.Control.Nsteps / self.params.Control.dump_step) + 1
            D = np.zeros((self.params.num_species, no_int))
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            for i, sp in enumerate(self.params.species):
                integrand = np.array(Z.dataframe["{} Total Velocity ACF".format(sp.name)])
                time = np.array(Z.dataframe["Time"])
                const = 1.0 / 3.0 / Z.tot_mass_density
                for it in range(1, no_int):
                    D[i, it] = const * np.trapz(integrand[:it], x=time[:it])

                # Sk = StaticStructureFactor(self.params)
                # try:
                #     Sk.dataframe = pd.read_csv(Sk.filename_csv, index_col=False)
                # except FileNotFoundError:
                #     Sk.compute()
                # Take the determinant of the matrix
                # Take the limit k --> 0 .

                self.transport_coefficients["{} Diffusion".format(sp.name)] = D[i, :]
                # Find the minimum slope. This would be the ideal value
                # indx = np.gradient(D[i, :]).argmin()
                # lgnd_label = r'$D_{' + sp.name + '} =' + '{:1.4f}$'.format(D[i, indx]) \
                #              + " @ $t = {:2.2f}$".format(time[half_t + indx]*self.params.wp)
                ax.plot(time * self.params.wp, D[i, :], label=r'$D_{' + sp.name + '}(t)$')
                # ax2.semilogy(time*self.params.wp, -np.gradient(np.gradient(D[i, :])), ls='--', lw=LW - 1,
                #          label=r'$\nabla D_{' + sp.name + '}(t)$')
            # Complete figure
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            ax.set_ylabel(r'$D_{\alpha}(t)/(a^2\omega_{\alpha})$')
            ax.set_xlabel(r'$\omega_p t$')

            # ax2.legend(loc='best')
            # ax2.tick_params(labelsize=FSZ)
            # ax2.set_ylabel(r'$\nabla D_{\alpha}(t)/(a^2\omega_{\alpha})$')

            fig.tight_layout()
            fig.savefig(os.path.join(self.params.Control.checkpoint_dir,
                                     'DiffusionPlot_' + self.params.Control.fname_app + '.png'))
            if show:
                fig.show()

        elif quantity == "Interdiffusion":
            Z = VelocityAutocorrelationFunctions(self.params)
            Z.plot(show=True)
            no_int = Z.no_dumps
            no_dij = int(Z.no_species * (Z.no_species - 1) / 2)
            D_ij = np.zeros((no_dij, no_int))

            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            indx = 0
            for i, sp1 in enumerate(self.params.species):
                for j in range(i + 1, self.params.num_species):
                    integrand = np.array(Z.dataframe["{}-{} Total Current ACF".format(sp1.name,
                                                                                      self.params.species[j].name)])
                    time = np.array(Z.dataframe["Time"])
                    const = 1.0 / (3.0 * self.params.wp * self.params.aws ** 2)
                    const /= (sp1.concentration * self.params.species[j].concentration)
                    for it in range(1, no_int):
                        D_ij[indx, it] = const * np.trapz(integrand[:it], x=time[:it])

                    self.transport_coefficients["{}-{} Inter Diffusion".format(sp1.name,
                                                                               self.params.species[j].name)] = D_ij[i,
                                                                                                               :]

        return


class VelocityAutocorrelationFunctions:
    """
        Velocity Auto-correlation function.

        Attributes
        ----------
            a_ws : float
                Wigner-Seitz radius.

            wp : float
                Total plasma frequency.

            dump_step : int
                Dump step frequency.

            dt : float
                Timestep magnitude.

            filename: str
                Name of output files.

            fldr : str
                Folder containing dumps.

            no_dumps : int
                Number of dumps.

            no_species : int
                Number of species.

            species_np: array
                Array of integers with the number of particles for each species.

            species_charge: array
                Array of with the charge of each species.

            species_names : list
                Names of particle species.

            tot_no_ptcls : int
                Total number of particles.
        """

    def __init__(self, params):
        self.fldr = params.Control.checkpoint_dir
        self.dump_dir = params.Control.dump_dir
        self.fname_app = params.Control.fname_app
        self.filename_csv = os.path.join(self.fldr, "VelocityACF_" + self.fname_app + '.csv')
        self.dump_step = params.Control.dump_step
        self.no_dumps = len(os.listdir(params.Control.dump_dir))
        self.no_species = len(params.species)
        self.species_names = []
        self.dt = params.Control.dt  # No of dump to skip
        self.species_np = np.zeros(self.no_species, dtype=int)
        self.species_masses = np.zeros(self.no_species)
        self.species_dens = np.zeros(self.no_species)
        for i in range(self.no_species):
            self.species_np[i] = int(params.species[i].num)
            self.species_dens[i] = params.species[i].num_density
            self.species_masses[i] = params.species[i].mass
            self.species_names.append(params.species[i].name)
        self.tot_mass_density = self.species_masses.transpose() @ self.species_dens
        self.tot_no_ptcls = params.total_num_ptcls
        self.wp = params.wp
        self.a_ws = params.aws
        self.dt = params.Control.dt

    def parse(self):
        """
        Parse vacf from csv file if found otherwise compute them.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            self.compute()

    def compute(self, time_averaging=False, it_skip=100):
        """
        Compute the velocity auto-correlation functions.
        """

        # Parse the particles from the dump files
        vel = np.zeros((3, self.tot_no_ptcls, self.no_dumps))
        #
        print("Parsing particles' velocities.")
        time = np.zeros(self.no_dumps)
        for it in tqdm(range(self.no_dumps)):
            dump = int(it * self.dump_step)
            time[it] = dump * self.dt
            datap = load_from_restart(self.dump_dir, dump)
            vel[0, :, it] = datap["vel"][:, 0]
            vel[1, :, it] = datap["vel"][:, 1]
            vel[2, :, it] = datap["vel"][:, 2]
        #

        data_dic = {"Time": time}
        self.dataframe = pd.DataFrame(data_dic)
        if time_averaging:
            print("Calculating vacf with time averaging on...")
        else:
            print("Calculating vacf with time averaging off...")
        vacf = calc_vacf(vel, self.species_np, self.species_masses, time_averaging, it_skip)

        # Save to csv
        v_ij = 0
        for sp in range(self.no_species):
            self.dataframe["{} X Velocity ACF".format(self.species_names[sp])] = vacf[v_ij, 0, :]
            self.dataframe["{} Y Velocity ACF".format(self.species_names[sp])] = vacf[v_ij, 1, :]
            self.dataframe["{} Z Velocity ACF".format(self.species_names[sp])] = vacf[v_ij, 2, :]
            self.dataframe["{} Total Velocity ACF".format(self.species_names[sp])] = vacf[v_ij, 3, :]
            for sp2 in range(sp + 1, self.no_species):
                v_ij += 1
                self.dataframe["{}-{} X Current ACF".format(self.species_names[sp],
                                                            self.species_names[sp2])] = vacf[v_ij, 0, :]
                self.dataframe["{}-{} Y Current ACF".format(self.species_names[sp],
                                                            self.species_names[sp2])] = vacf[v_ij, 1, :]
                self.dataframe["{}-{} Z Current ACF".format(self.species_names[sp],
                                                            self.species_names[sp2])] = vacf[v_ij, 2, :]
                self.dataframe["{}-{} Total Current ACF".format(self.species_names[sp],
                                                                self.species_names[sp2])] = vacf[v_ij, 3, :]

        self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')
        return

    def plot(self, intercurrent=False, show=False):
        """
        Plot the velocity autocorrelation function and save the figure.
        """
        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            self.compute()

        if intercurrent:
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            for i, sp_name in enumerate(self.species_names):
                for j in range(i + 1, self.no_species):
                    J = self.dataframe["{}-{} Total Current ACF".format(sp_name, self.species_names[j])]
                    ax.plot(self.dataframe["Time"] * self.wp,
                            J / J[0], label=r'$J_{' + sp_name + self.species_names[j] + '} (t)$')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_ylabel(r'$J(t)$', )
            ax.set_xlabel(r'$\omega_p t$')
            ax.set_xscale('log')
            ax.set_ylim(-0.2, 1.2)
            fig.tight_layout()
            fig.savefig(os.path.join(self.fldr, 'InterCurrentACF_' + self.fname_app + '.png'))
            if show:
                fig.show()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            for i, sp_name in enumerate(self.species_names):
                Z = self.dataframe["{} Total Velocity ACF".format(sp_name)]
                ax.plot(self.dataframe["Time"] * self.wp,
                        Z / Z[0], label=r'$Z_{' + sp_name + '} (t)$')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_ylabel(r'$Z(t)$', )
            ax.set_xlabel(r'$\omega_p t$')
            ax.set_xscale('log')
            ax.set_ylim(-0.2, 1.2)
            fig.tight_layout()
            fig.savefig(os.path.join(self.fldr, 'TotalVelocityACF_' + self.fname_app + '.png'))
            if show:
                fig.show()


class XYZWriter:
    """
    Write the XYZ file for OVITO visualization.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius. Used for rescaling.

        dump_skip : int
            Dump step interval.

        dump_dir : str
            Directory containing Sarkas dumps.

        dump_step : int
            Dump step frequency.

        filename: str
            Name of output files.

        fldr : str
            Folder containing dumps.

        no_dumps : int
            Number of dumps.

        tot_no_ptcls : int
            Total number of particles.

        wp : float
            Plasma frequency used for rescaling.
    """

    def __init__(self, params):
        self.fldr = params.Control.checkpoint_dir
        self.dump_dir = params.Control.dump_dir
        self.filename = os.path.join(self.fldr, "pva_" + params.Control.fname_app + '.xyz')
        self.dump_step = params.Control.dump_step
        self.no_dumps = len(os.listdir(params.Control.dump_dir))
        self.dump_skip = 1
        self.tot_no_ptcls = params.total_num_ptcls
        self.a_ws = params.aws
        self.wp = params.wp
        self.verbose = params.Control.verbose

    def save(self, dump_skip=1):
        """
        Save the XYZ file by reading Sarkas dumps.

        Parameters
        ----------
        dump_skip : int
            Interval of dumps to skip. Default = 1

        """

        self.dump_skip = dump_skip
        f_xyz = open(self.filename, "w+")

        # Rescale constants. This is needed since OVITO has a small number limit.
        pscale = 1.0 / self.a_ws
        vscale = 1.0 / (self.a_ws * self.wp)
        ascale = 1.0 / (self.a_ws * self.wp ** 2)

        for it in tqdm(range(int(self.no_dumps / self.dump_skip)), disable=not self.verbose):
            dump = int(it * self.dump_step * self.dump_skip)

            data = load_from_restart(self.dump_dir, dump)

            f_xyz.writelines("{0:d}\n".format(self.tot_no_ptcls))
            f_xyz.writelines("name x y z vx vy vz ax ay az\n")
            np.savetxt(f_xyz,
                       np.c_[data["species_name"], data["pos"] * pscale, data["vel"] * vscale, data["acc"] * ascale],
                       fmt="%s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")

        f_xyz.close()


def read_pickle(input_file):
    """
    Read Pickle File containing params.

    Parameters
    ----------
    input_file: str
        Input YAML file of the simulation.
    Returns
    -------
    data : dict
        Params dictionary.
    """
    with open(input_file, 'r') as stream:
        dics = yaml.load(stream, Loader=yaml.FullLoader)
        for lkey in dics:
            if lkey == "Control":
                for keyword in dics[lkey]:
                    for key, value in keyword.items():
                        # Directory where to store Checkpoint files
                        if key == "output_dir":
                            checkpoint_dir = os.path.join("Simulations", value)

    pickle_file = os.path.join(checkpoint_dir, "S_parameters.pickle")

    data = np.load(pickle_file, allow_pickle=True)

    return data


def count_dumps(dump_dir):
    files = os.listdir(dump_dir)


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

    file_name = os.path.join(fldr, "S_checkpoint_" + str(it) + ".npz")
    data = np.load(file_name, allow_pickle=True)
    return data


def kspace_setup(no_ka, box_lengths):
    """
    Calculate all allowed :math:`k` vectors.

    Parameters
    ----------
    no_ka : array
        Number of harmonics in each direction.

    box_lengths : array
        Length of each box's side.

    Returns
    -------
    k_arr : list
        List of all possible :math:`k` vectors with their corresponding magnitudes and indexes.

    k_counts : array
        Number of occurrences of each :math:`k` magnitude.

    k_unique : array
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


@njit
def calc_Sk(nkt, ka_list, ka_counts, species_np, no_dumps):
    """
    Calculate :math:`S_{ij}(k)` at each saved timestep.

    Parameters
    ----------
    nkt : ndarray, complex
        Density fluctuations of all species. Shape = ( ``no_species``, ``no_dumps``, ``no_ka_values``)

    ka_list :
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``ka_counts``.
        Shape=(`no_ka_values`, 5)

    ka_counts : array
        Number of times each :math:`k` magnitude appears.

    species_np : array
        Array with number of particles of each species.

    no_dumps : int
        Number of dumps.

    Returns
    -------

    Sk_all : ndarray
        Array containing :math:`S_{ij}(k)`. Shape=(``no_Sk``,``no_ka_values``, ``no_dumps``)

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


@njit
def calc_elec_current(vel, sp_charge, sp_num):
    """
    Calculate the total electric current and electric current of each species.

    Parameters
    ----------
    vel: array
        Particles' velocities.

    sp_charge: array
        Charge of each species.

    sp_num: array
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


@njit(parallel=True)
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

    sp_num: array
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
                jc_acf[indx, d + 1, :] = temp / norm_counter
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


@njit(parallel=True)
def calc_vacf_single(vel, sp_num, sp_mass, time_averaging, it_skip=100):
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

    sp_num: array
        Number of particles of each species.

    Returns
    -------
    vacf: ndarray
        Velocity autocorrelation functions. Shape =
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
        for i in prange(3):
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
def autocorrelationfunction(At):
    """
    Calculate the autocorrelation function of the array input.

    .. math::
        A(\tau) =  \sum_j^D \sum_i^T A_j(t_i)A_j(t_i + \tau)

    where :math:`D` (= ``no_dim``) is the number of dimensions and :math:`T` (= ``no_steps``) is the total length
    of the simulation.

    Parameters
    ----------
    At : ndarray
        Observable to autocorrelate. Shape=(``no_dim``, ``no_steps``).

    Returns
    -------
    ACF : array
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
        A(\tau) =  \sum_i^T A(t_i)A(t_i + \tau)

    where :math:`T` (= ``no_steps``) is the total length of the simulation.

    Parameters
    ----------
    At : array
        Observable to autocorrelate. Shape=(``no_steps``).

    Returns
    -------
    ACF : array
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
def correlationfunction(At, Bt):
    """
    Calculate the correlation function :math:`\mathbf{A}(t)` and :math:`\mathbf{B}(t)`

    .. math::
        C_{AB}(\tau) =  \sum_j^D \sum_i^T A_j(t_i)B_j(t_i + \tau)

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
    CF : array
        Correlation function :math:`C_{AB}(\tau)`
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
        C_{AB}(\tau) =  \sum_i^T A(t_i)B(t_i + \tau)

    where :math:`T` (= ``no_steps``) is the total length of the simulation.

    Parameters
    ----------
    At : array
        Observable to correlate. Shape=(``no_steps``).

    Bt : array
        Observable to correlate. Shape=(``no_steps``).

    Returns
    -------
    CF : array
        Correlation function :math:`C_{AB}(\tau)`
    """
    no_steps = At.shape[0]
    CF = np.zeros(no_steps)
    Norm_counter = np.zeros(no_steps)

    for it in range(no_steps):
        CF[: no_steps - it] += At[it] * Bt[it:no_steps]
        Norm_counter[: no_steps - it] += 1.0

    return CF / Norm_counter


@njit
def timeaveraging(At):
    no_steps = At.shape[0]
    avg = np.zeros(no_steps)
    norm_counter = np.zeros(no_steps)

    for it in range(no_steps):
        for jt in range(it, no_steps - 1):
            avg[it: no_steps - jt] += At[jt:no_steps] / At[jt]
            norm_counter[it:no_steps - jt] += 1.0

    return avg / norm_counter


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

    species_mass : array
        Mass of each species.

    species_np : array
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
    no_dim = pos.shape[0]
    pressure_tensor = np.zeros((no_dim, no_dim))
    sp_start = 0
    # Rescale vel and acc of each particle by their individual mass
    for sp in range(len(species_np)):
        sp_end = sp_start + species_np[sp]
        vel[:, sp_start: sp_end] *= np.sqrt(species_mass[sp])
        acc[:, sp_start: sp_end] *= species_mass[sp]  # force
        sp_start = sp_end

    pressure = 0.0
    for i in range(no_dim):
        for j in range(no_dim):
            pressure_tensor[i, j] = np.sum(vel[i, :] * vel[j, :] + pos[i, :] * acc[j, :]) / box_volume
        pressure += pressure_tensor[i, i] / 3.0

    return pressure, pressure_tensor


def calc_nkt(fldr, no_dumps, dump_step, species_np, k_list):
    """
    Calculate density fluctuations :math:`n(k,t)` of all species.

    .. math::
        n_{A}(\mathbf{k},t) = \sum_i^N_A \exp \left [ - i \mathbf{k} \cdot \mathbf{r}_{Ai}(t) \right]

    where :math:`N_A` is the number of particles of species :math:`A`.

    Parameters
    ----------
    fldr : str
        Name of folder containing particles data.

    no_dumps : int
        Number of saved timesteps.

    dump_step : int
        Timestep interval saving.

    species_np : array
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
    for it in tqdm(range(no_dumps)):
        dump = int(it * dump_step)
        data = load_from_restart(fldr, dump)
        pos = data["pos"]
        sp_start = 0
        for i, sp in enumerate(species_np):
            sp_end = sp_start + sp
            nkt[i, it, :] = calc_nk(pos[sp_start:sp_end, :], k_list)
            sp_start = sp_end

    return nkt


def calc_vkt(fldr, no_dumps, dump_step, species_np, k_list):
    """
    Calculate the longitudinal and transverse velocities fluctuations of all species.

    Longitudinal
    .. math::
        \lambda_A(\mathbf{k}, t) = \sum_i^N_{A} \mathbf{k} \cdot \mathbf{v}_{A,i}(t) \exp \left[ - i \mathbf{k} \cdot \mathbf{r}_{A,i}(t) \right]

    Transverse
    .. math::
        \tau_A(\mathbf{k}, t) = \sum_i^N_{A} \mathbf{k} \times \mathbf{v}_{A,i}(t) \exp \left[ - i \mathbf{k} \cdot \mathbf{r}_{A,i}(t) \right]

    where :math:`N_A` is the number of particles of species :math:`A`.

    Parameters
    ----------
    fldr : str
        Name of folder containing particles data.

    no_dumps : int
        Number of saved timesteps.

    dump_step : int
        Timestep interval saving.

    species_np : array
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
    for it in tqdm(range(no_dumps)):
        dump = int(it * dump_step)
        data = load_from_restart(fldr, dump)
        pos = data["pos"]
        vel = data["vel"]
        sp_start = 0
        for i, sp in enumerate(species_np):
            sp_end = sp_start + sp
            vkt_par[i, it, :], vkt_perp_i[i, it, :], vkt_perp_j[i, it, :], vkt_perp_k[i, it, :] = calc_vk(
                pos[sp_start:sp_end, :], vel[sp_start:sp_end], k_list)
            sp_start = sp_end

    return vkt_par, vkt_perp_i, vkt_perp_j, vkt_perp_k


@njit
def calc_vk(pos_data, vel_data, k_list):
    """
    Calculate the instantaneous longitudinal and transverse velocity fluctuations.

    Parameters
    ----------
    pos_data : ndarray
        Particles' position. Shape = ( ``no_dumps``, , ``tot_no_ptcls`)

    vel_data : ndarray
        Particles' velocities. Shape = ( ``no_dumps``, 3, ``tot_no_ptcls``)

    k_list : list
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``ka_counts``.
        Shape=(``no_ka_values``, 5)

    Returns
    -------
    vkt : array
        Array containing longitudinal velocity fluctuations.

    vkt_i : array
        Array containing transverse velocity fluctuations in the :math:`x` direction.

    vkt_j : array
        Array containing transverse velocity fluctuations in the :math:`y` direction.

    vkt_k : array
        Array containing transverse velocity fluctuations in the :math:`z` direction.

    """

    # Longitudinal
    vk = np.zeros(len(k_list), dtype=np.complex128)

    # Transverse
    vk_i = np.zeros(len(k_list), dtype=np.complex128)
    vk_j = np.zeros(len(k_list), dtype=np.complex128)
    vk_k = np.zeros(len(k_list), dtype=np.complex128)

    for ik, k_vec in enumerate(k_list):
        kr_i = 2.0 * np.pi * (k_vec[0] * pos_data[:, 0] + k_vec[1] * pos_data[:, 1] + k_vec[2] * pos_data[:, 2])
        k_dot_v = 2.0 * np.pi * (k_vec[0] * vel_data[:, 0] + k_vec[1] * vel_data[:, 1] + k_vec[2] * vel_data[:, 2])
        vk[ik] = np.sum(k_dot_v * np.exp(-1j * kr_i))

        k_cross_v_i = 2.0 * np.pi * (k_vec[1] * vel_data[:, 2] - k_vec[2] * vel_data[:, 1])
        k_cross_v_j = -2.0 * np.pi * (k_vec[0] * vel_data[:, 2] - k_vec[2] * vel_data[:, 0])
        k_cross_v_k = 2.0 * np.pi * (k_vec[0] * vel_data[:, 1] - k_vec[1] * vel_data[:, 0])
        vk_i[ik] = np.sum(k_cross_v_i * np.exp(-1j * kr_i))
        vk_j[ik] = np.sum(k_cross_v_j * np.exp(-1j * kr_i))
        vk_k[ik] = np.sum(k_cross_v_k * np.exp(-1j * kr_i))

    return vk, vk_i, vk_j, vk_k


@njit
def calc_nk(pos_data, k_list):
    """
    Calculate instantaneous density fluctuations :math:`n(k)`.

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
    nk : array
        Array containing :math:`n(k)`.
    """

    nk = np.zeros(len(k_list), dtype=np.complex128)

    for ik, k_vec in enumerate(k_list):
        kr_i = 2.0 * np.pi * (k_vec[0] * pos_data[:, 0] + k_vec[1] * pos_data[:, 1] + k_vec[2] * pos_data[:, 2])
        nk[ik] = np.sum(np.exp(-1j * kr_i))

    return nk


def calc_Skw(nkt, ka_list, ka_counts, species_np, no_dumps, dt, dump_step):
    """
    Calculate the Fourier transform of the correlation function of ``nkt``.

    Parameters
    ----------
    nkt : nkarray, complex
        Particles' density or velocity fluctuations.
        Shape = ( ``no_species``, ``no_k_list``, ``no_dumps``)

    ka_list : list
        List of :math:`k` indices in each direction with corresponding magnitude and index of ``ka_counts``.
        Shape=(`no_ka_values`, 5)

    ka_counts : array
        Number of times each :math:`k` magnitude appears.

    species_np : array
        Array with one element giving number of particles.

    no_dumps : int
        Number of dumps.

    Returns
    -------
    Skw : ndarray
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
def calc_statistical_efficiency(observable, run_avg, run_std, max_no_divisions, no_dumps):
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


def plot_labels(xdata, ydata, xlbl, ylbl, units):
    """
    Create plot labels with correct units and prefixes.

    Parameters
    ----------
    xdata: array
        X values.

    ydata: array
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
        Y label units
    """
    xmax = xdata.max()
    ymax = ydata.max()

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

    # Find the correct Units
    units_dict = UNITS[1] if units == 'cgs' else UNITS[0]
    for key in units_dict:
        if key in ylbl:
            ylabel = ' [' + yprefix + units_dict[key] + ']'

    for key in units_dict:
        if key in xlbl:
            xlabel = ' [' + xprefix + units_dict[key] + ']'

    return xmul, ymul, xprefix, yprefix, xlabel, ylabel
