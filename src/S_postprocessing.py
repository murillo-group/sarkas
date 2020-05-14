"""
Module for calculating physical quantities from Sarkas dumps
"""
import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt
import time as tme

from matplotlib import rc

rc('text', usetex=True)

lw = 2
fsz = 14
msz = 8


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

        filename_output: str
            Name of the energy output file.

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
        self.fldr = params.Control.checkpoint_dir + '/'
        self.fname_app = params.Control.fname_app
        self.dump_step = params.Control.dump_step
        self.no_dumps = int(params.Control.Nsteps / params.Control.dump_step)
        self.no_dim = params.d
        self.box_volume = params.box_volume
        self.tot_no_ptcls = params.total_num_ptcls

        self.no_species = len(params.species)
        self.species_np = np.zeros(self.no_species)
        self.species_names = []
        self.species_masses = np.zeros(self.no_species)
        for i in range(self.no_species):
            self.species_np[i] = params.species[i].num
            self.species_names.append(params.species[i].name)
            self.species_masses[i] = params.species[i].mass

        # Output file with Energy and Temperature
        self.filename_output = self.fldr + "Energy_" + self.fname_app
        self.filename_csv = self.fldr + "Thermodynamics_" + self.fname_app + '.csv'

        data = {}
        output_data = np.loadtxt(self.filename_output + '.out')
        self.time = output_data[:, 0]
        data["time"] = output_data[:, 0]
        data["Temperature"] = output_data[:, 1]
        data["Total Energy"] = output_data[:, 2]
        data["Kinetic Energy"] = output_data[:, 3]
        data["Potential Energy"] = output_data[:, 4]
        data["Gamma"] = params.Potential.Gamma_eff * params.T_desired / output_data[:, 1]

        # if self.no_species > 1:
        #     indx = 5 + self.no_species
        #     for i in range(self.no_species):
        #         data["{} Temperature".format(self.species_names[i])] = output_data[:, 5 + i]
        #         data["{} Kinetic Energy".format(self.species_names[i])] = output_data[:, indx + i]

        # Constants
        self.wp = params.wp
        self.kB = params.kB
        self.eV2K = params.eV2K
        self.a_ws = params.aws

        self.dataframe = pd.DataFrame(data)
        try:
            open(self.filename_csv, 'r')
        except FileNotFoundError:
            self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

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

    def plot(self, quantity="Total Energy", delta=True):
        """
        Plot `quantity` vs time and save the figure with appropriate name.

        Parameters
        ----------
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
        lbl = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        ylbl = {}
        ylbl["Total Energy"] = r"$E_{\textrm{tot}}(t)$"
        ylbl["Kinetic Energy"] = r"$K_{\textrm{tot}}(t)$"
        ylbl["Potential Energy"] = r"$U_{\textrm{tot}}(t)$"
        ylbl["Temperature"] = r"$T(t)$"
        ylbl[
            "Pressure Tensor ACF"] = r'$\mathcal P_{\alpha\beta} = \langle P_{\alpha\beta}(0)P_{\alpha\beta}(t)\rangle$'
        ylbl["Pressure Tensor"] = r"$P_{\alpha\beta} (t)$"
        ylbl["Gamma"] = r"$\Gamma (t)$"
        ylbl["Pressure"] = r"$P(t)$"
        if quantity[:-3] == "Pressure Tensor ACF":
            for i in range(self.no_dim * self.no_dim):
                ax.plot(self.time * self.wp, self.dataframe[quantity][:, i], lw=lw,
                        label=r'$\mathcal P_{' + lbl[i] + '} (t)$')
            ax.set_xscale('log')
            ax.legend(loc='upper right', ncol=3, fontsize=fsz)
            ax.set_ylim(-1, 1.5)

        elif quantity[:-3] == "Pressure Tensor":
            for i in range(self.no_dim * self.no_dim):
                ax.plot(self.time * self.wp, self.dataframe[quantity][:, i], lw=lw, label=r'$P_{' + lbl[i] + '} (t)$')
            ax.set_xscale('log')
            ax.legend(loc='upper right', ncol=3, fontsize=fsz)

        else:
            if delta:
                delta = (self.dataframe[quantity] - self.dataframe[quantity][0])
                ax.plot(self.time * self.wp, delta, lw=lw)
                ylbl[quantity] = r"$\Delta$" + ylbl[quantity]
                # ax.set_title(r"Relative $\Delta$ " + quantity, fontsize=fsz)
            else:
                ax.plot(self.time * self.wp, self.dataframe[quantity], lw=lw)
                # ax.set_title(quantity, fontsize=fsz)

            # ax.axhline(mean, ls='--', lw=lw, color='k')

        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fsz)
        ax.set_ylabel(ylbl[quantity], fontsize=fsz)
        ax.set_xlabel(r'$\omega_p t$', fontsize=fsz)
        fig.tight_layout()
        fig.savefig(self.fldr + quantity + '_' + self.fname_app + '.png')


class ElectricCurrent:
    """
    Electric Current Auto-correlation function.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius.

        wp : float
            Total plasma frequency.

        dump_skip : int
            Interval between dumps.

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

        sp_names : list
            Names of particle species.

        tot_no_ptcls : int
            Total number of particles.

        dump_skip : int
            Interval between dumps.
    """

    def __init__(self, params):
        self.fldr = params.Control.checkpoint_dir + '/'
        self.fname_app = params.Control.fname_app
        self.filename_csv = self.fldr + "ElectricCurrent_" + self.fname_app + '.csv'
        self.dump_step = params.Control.dump_step
        self.no_dumps = int(params.Control.Nsteps / params.Control.dump_step)
        self.no_species = len(params.species)
        self.species_np = np.zeros(self.no_species, dtype=int)
        self.species_names = []
        self.dt = params.Control.dt  # No of dump to skip
        self.species_charge = np.zeros(self.no_species)
        for i in range(self.no_species):
            self.species_np[i] = int(params.species[i].num)
            self.species_charge[i] = params.species[i].charge
            self.species_names.append(params.species[i].name)

        self.time = np.arange(self.no_dumps) * self.dt * self.dump_step
        self.tot_no_ptcls = params.total_num_ptcls
        self.wp = params.wp
        self.a_ws = params.aws
        self.dt = params.Control.dt

        try:
            self.dataframe = pd.read_csv(self.filename_csv, index_col=False)
        except FileNotFoundError:
            data = {"Time": self.time}
            self.dataframe = pd.DataFrame(data)
            self.compute()
            self.dataframe.to_csv(self.filename_csv, index=False, encoding='utf-8')

    def compute(self):
        """
        Compute the electric current and the corresponding auto-correlation functions.
        """
        # Dev Note: The first index is the value of ka,
        # The second index indicates S_ij
        # The third index indicates S_ij(t)

        # Parse the particles from the dump files
        vel = np.zeros((self.no_dumps, 3, self.tot_no_ptcls))

        # vscale = self.a_ws * self.wp
        for it in range(self.no_dumps):
            dump = int(it * self.dump_step)
            datap = load_from_restart(self.fldr, dump)
            vel[it, 0, :] = datap["vel"][:, 0]
            vel[it, 1, :] = datap["vel"][:, 1]
            vel[it, 2, :] = datap["vel"][:, 2]

        species_current, total_current = calc_elec_current(vel, self.species_charge, self.species_np)

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

        return

    def plot(self):
        """
        Plot the electric current autocorrelation function and save the figure.
        """

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.plot(self.dataframe["Time"] * self.wp, self.dataframe["Total Current ACF"], lw=lw,
                label=r'$J_{\textrm{tot}} (t)$')

        if self.no_species > 1:
            for i in range(self.no_species):
                ax.plot(self.dataframe["Time"] * self.wp,
                        self.dataframe["{} Total Current ACF".format(self.species_names[i])],
                        lw=lw, label=r'$J_{' + self.species_names[i] + '} (t)$')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=fsz)
        ax.tick_params(labelsize=fsz)
        ax.set_ylabel(r'$J(t)$', fontsize=fsz)
        ax.set_xlabel(r'$\omega_p t$', fontsize=fsz)
        ax.set_xscale('log')
        fig.tight_layout()
        fig.savefig(self.fldr + 'TotalCurrentACF_' + self.fname_app + '.png')


class XYZFile:
    """
    Write the XYZ file for OVITO visualization.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius. Used for rescaling.

        dump_skip : int
            Dump step interval.

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
        self.fldr = params.Control.checkpoint_dir + '/'
        self.filename = self.fldr + "pva_" + params.Control.fname_app + '.xyz'
        self.dump_step = params.Control.dump_step
        self.no_dumps = int(params.Control.Nsteps / params.Control.dump_step)
        self.dump_skip = 1
        self.tot_no_ptcls = params.total_num_ptcls

        self.a_ws = params.aws
        self.wp = params.wp

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
        vscale = self.wp * self.a_ws
        ascale = self.wp ** 2 * self.a_ws
        for it in range(int(self.no_dumps / self.dump_skip)):
            dump = int(it * self.dump_step * self.dump_skip)

            data = load_from_restart(self.fldr, dump)

            f_xyz.writelines("{0:d}\n".format(self.tot_no_ptcls))
            f_xyz.writelines("name x y z vx vy vz ax ay az\n")
            np.savetxt(f_xyz, np.c_[data["species_name"], data["pos"] / self.a_ws, data["vel"] / vscale,
                                    data["acc"] / ascale],
                       fmt="%s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")

        f_xyz.close()


class StaticStructureFactor:
    """ Static Structure Factors :math:`S_{ij}(k)`.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius.

        box_lengths : array
            Array with box length in each direction.

        dataframe : dict
            Pandas dataframe. It contains all the :math:`S_{ij}(k)` and :math:`ka_values`.

        dump_step : int
            Dump step frequency.

        filename_csv: str
            Name of output files.

        fname_app: str
            Appendix of filenames.

        fldr : str
            Folder containing dumps.

        ka_min : float
            Smallest possible (non-dimensional) wavenumber :math:`ka = 2\pi/L`.

        no_dumps : int
            Number of dumps.

        no_ka : int
            Number of integer multiples of minimum :math:`ka` value

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
        """

    def __init__(self, params):
        self.no_ka = params.PostProcessing.no_ka_values  # number of ka values
        self.fldr = params.Control.checkpoint_dir + '/'
        self.fname_app = params.Control.fname_app
        self.filename_csv = self.fldr + "StaticStructureFunction_" + self.fname_app + ".csv"
        self.dump_step = params.Control.dump_step
        self.no_dumps = int(params.Control.Nsteps / params.Control.dump_step)
        self.no_species = len(params.species)
        self.tot_no_ptcls = params.total_num_ptcls

        self.no_Sk = int(self.no_species * (self.no_species + 1) / 2)
        self.a_ws = params.aws
        self.box_lengths = np.array([params.Lx, params.Ly, params.Lz])
        self.ka_min = 2.0 * np.pi * self.a_ws / params.Lx
        self.species_np = np.zeros(self.no_species)
        self.species_names = []
        for i in range(self.no_species):
            self.species_np[i] = params.species[i].num
            self.species_names.append(params.species[i].name)

    def compute(self, principal_axis=True):
        """
        Calculate all :math:`S_{ij}(k)`, save them into a Pandas dataframe, and write them to a csv.

        Parameters
        ----------
        principal_axis : bool
            Flag to calculate :math:`S_{ij}(k)` only along the principal cartesian direction :math:`X, Y, Z`.

        """
        # Dev Note: The first index is the value of ka,
        # The second index indicates S_ij
        # The third index indicates S_ij(t)

        # Parse the particles from the dump files

        if self.no_species == 1:
            if principal_axis:
                calculate = calc_Sk_single_pa
                ka_unique = np.arange(1, self.no_ka + 1)
            else:
                calculate = calc_Sk_single
                ka_list, ka_counts, ka_unique = kspace_setup(self.no_ka)
                # print(ka_list)
                # print(ka_list.shape)
                # print(ka_unique)
        else:
            if principal_axis:
                calculate = calc_Sk_multi_pa
                ka_unique = np.arange(1, self.no_ka + 1)
            else:
                calculate = calc_Sk_multi
                ka_list, ka_counts, ka_unique = kspace_setup(self.no_ka)
                # print(ka_list)
                # print(ka_list.shape)
                # print(ka_unique)

        ka_values = ka_unique * self.ka_min
        data = {"ka values": ka_values}
        self.dataframe = pd.DataFrame(data)

        # Grab particles positions
        pos = np.zeros((self.no_dumps, 3, self.tot_no_ptcls))
        print("Parsing Particles' Positions ...")
        for it in range(self.no_dumps):
            dump = int(it * self.dump_step)
            data = load_from_restart(self.fldr, dump)
            pos[it, 0, :] = data["pos"][:, 0] / self.box_lengths[0]
            pos[it, 1, :] = data["pos"][:, 1] / self.box_lengths[1]
            pos[it, 2, :] = data["pos"][:, 2] / self.box_lengths[2]

        start = tme.time()
        if principal_axis:
            print("Calculating S(k) along principal axis only ...")
            Sk_all = calculate(pos, ka_values * self.a_ws, self.species_np, self.no_dumps)
        else:
            print("Calculating S(k) ...")
            Sk_all = calculate(pos, ka_list, ka_counts, self.species_np, self.no_dumps)
        end = tme.time()
        print('Elapsed time = ', (end - start))
        Sk = np.mean(Sk_all, axis=-1)
        Sk_err = np.std(Sk_all, axis=-1)

        if self.no_species > 1:

            for i in range(self.no_species):
                for j in range(i, self.no_species):
                    self.dataframe['{}-{} SSF'.format(self.species_names[i], self.species_names[j])] = Sk[:, i, j]
                    self.dataframe[
                        '{}-{} SSF Errorbar'.format(self.species_names[i], self.species_names[j])] = Sk_err[:, i, j]
        else:
            for i in range(self.no_species):
                self.dataframe['{}-{} SSF'.format(self.species_names[i], self.species_names[i])] = Sk[:]
                self.dataframe['{}-{} SSF Errorbar'.format(self.species_names[i], self.species_names[i])] = Sk_err[:]

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
                                lw=lw, ls='--', marker='o', ms=msz, label=r'$S_{ ' + subscript + '} (k)$')
                else:
                    ax.plot(self.dataframe["ka values"],
                            self.dataframe["{}-{} SSF".format(self.species_names[i], self.species_names[j])],
                            lw=lw, label=r'$S_{ ' + subscript + '} (k)$')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=fsz)
        ax.tick_params(labelsize=fsz)
        ax.set_ylabel(r'$S(k)$', fontsize=fsz)
        ax.set_xlabel(r'$ka$', fontsize=fsz)
        fig.tight_layout()
        fig.savefig(self.fldr + 'StaticStructureFactor' + self.fname_app + '.png')
        if show:
            fig.show()


# class DynamicStructureFactor:
#     """ Dynamic Structure factor.
#
#     Attributes
#     ----------
#         a_ws : float
#             Wigner-Seitz radius.
#
#         dump_step : int
#             Dump step frequency.
#
#         filename: str
#             Name of output files.
#
#         fldr : str
#             Folder containing dumps.
#
#         ka_min : float
#             Smallest possible (non-dimensional) wavenumber :math:`ka = 2\pi/L`.
#
#         ka_max : float
#             Largest possible (non-dimensional) wavenumber = ``no_ka * ka_min``
#
#         no_dumps : int
#             Number of dumps.
#
#         no_ka : int
#             Number of integer multiples of minimum :math:`ka` value
#
#         no_species : int
#             Number of species.
#
#         species_np: array
#             Array of integers with the number of particles for each species.
#
#         dt : float
#             Timestep's value normalized by the total plasma frequency.
#
#         sp_names : list
#             Names of particle species.
#
#         tot_no_ptcls : int
#             Total number of particles.
#
#         """
#
#     def __init__(self, params):
#         self.no_ka = params.PostProcessing.no_ka_values  # number of ka values
#         self.ka_values = np.zeros(self.no_ka)
#         self.fldr = params.Control.checkpoint_dir + '/'
#         self.fname_app = params.Control.fname_app
#         self.filename_csv = self.fldr + "DynamicStructureFactor_" + self.fname_app + '.csv'
#         self.box_lengths = np.array([params.Lx, params.Ly, params.Lz])'
#         self.dump_step = params.Control.dump_step
#         self.no_dumps = int(params.Control.Nsteps / params.Control.dump_step)
#         self.no_species = len(params.species)
#         self.tot_no_ptcls = params.total_num_ptcls
#
#
#         self.ka_min = 2.0 * np.pi * self.a_ws / params.Lx
#         self.species_np = np.zeros(self.no_species)
#         self.species_names = []
#         for i in range(self.no_species):
#             self.species_np[i] = params.species[i].num
#             self.species_names.append(params.species[i].name)
#
#         self.Nsteps = params.Control.Nsteps
#         self.dt = params.Control.dt * params.wp
#         self.no_Skw = int(self.no_species * (self.no_species + 1) / 2)
#         self.a_ws = params.aws
#         self.wp = params.wp
#
#     def compute(self, principal_axis=True):
#         """
#         Compute :math: `S_{ij}(k,\omega)' and the array of :math: `\omega/\omega_p` values.
#         ``self.Skw``. Shape = (``no_ws``, ``no_Sij``)
#         """
#         if self.no_species == 1:
#             if principal_axis:
#                 calculate = calc_nkt_single_pa
#                 ka_unique = np.arange(1, self.no_ka + 1)
#             else:
#                 calculate = calc_nkt_single()
#                 ka_list, ka_counts, ka_unique = kspace_setup(self.no_ka)
#                 # print(ka_list)
#                 # print(ka_list.shape)
#                 # print(ka_unique)
#         else:
#             if principal_axis:
#                 calculate = calc_nkt_multi_pa
#                 ka_unique = np.arange(1, self.no_ka + 1)
#             else:
#                 calculate = calc_nkt_multi
#                 ka_list, ka_counts, ka_unique = kspace_setup(self.no_ka)
#                 # print(ka_list)
#                 # print(ka_list.shape)
#                 # print(ka_unique)
#
#         ka_values = ka_unique * self.ka_min
#
#         tot_time = self.dt * self.dump_step * self.no_dumps
#         w_min = 2.0 * np.pi / tot_time
#         data = {"Frequencies": np.fft.fftfreq(self.no_dumps, w_min)}
#
#         # Parse the particles from the dump files
#         pos = np.zeros((self.no_dumps, 3, self.tot_no_ptcls))
#
#         for it in range(self.no_dumps):
#             dump = int(it * self.dump_step)
#             data = load_from_restart(self.fldr, dump)
#             pos[it, 0, :] = data["pos"][:, 0] / self.box_lengths[0]
#             pos[it, 1, :] = data["pos"][:, 1] / self.box_lengths[1]
#             pos[it, 2, :] = data["pos"][:, 2] / self.box_lengths[2]
#
#         start = tme.time()
#         if principal_axis:
#             print("Calculating n(k,t) along principal axis only ...")
#             nkt_all = calculate(pos, ka_values * self.a_ws, self.species_np, self.no_dumps)
#             for i in range( self.no_species):
#                 nkx_w_i = (np.fft.fft(nkt_all[i][ik, 0, :])) * self.dt * self.dump_step
#                 nky_w_i = (np.fft.fft(nkt_all[i][ik, 1, :])) * self.dt * self.dump_step
#                 nkz_w_i = (np.fft.fft(nkt_all[i][ik, 2, :])) * self.dt * self.dump_step
#
#                 for j in range(i, self.no_species):
#                     self.dataframe = pd.DataFrame(data)
#
#
#                     for ik in range(self.no_ka):
#
#
#                         Skx_w = np.abs(nkx_w) ** 2 / tot_time
#                         Sky_w = np.abs(nky_w) ** 2 / tot_time
#                         Skz_w = np.abs(nkz_w) ** 2 / tot_time
#                         column = "{}-{} DSF ka = {} ka_min".format(self.species_names[0], self.species_names[0], ik)
#                         self.dataframe[column] = (Skx_w + Sky_w + Skz_w) / 3.0
#         else:
#             print("Calculating n(k,t) ...")
#             nkt_all = calculate(pos, ka_list, ka_counts, self.species_np, self.no_dumps)
#
#             for ik in range(self.no_ka):
#                 nkw_i = (np.fft.fft(nkt_all[ik, :])) * self.dt * self.dump_step
#                 column = "{}-{} DSF ka = {} ka_min".format(self.species_names[0], self.species_names[0], ik)
#
#                 self.dataframe[column] = np.real(nkw_i * np.conj(nkw_i) ) / tot_time
#
#         end = tme.time()
#         print('Elapsed time = ', (end - start))
#
#
#
#
#
#         datafile = open(self.filename + '.out', "w")
#         np.savetxt(datafile, np.c_[self.w_array, self.Skw])
#         datafile.close()
#
#         return
#
#     def plot(self):
#         """
#         Plot :math: `S(k,\omega)` and save the figure.
#         """
#         try:
#             data = np.loadtxt(self.filename + '.out')
#             for ik in range(self.no_ka):
#                 self.ka_values[ik] = (ik + 1) * self.ka_min
#             self.w_array = data[:, 0]
#             self.Skw = data[:, 1:]
#         except OSError:
#             self.compute()
#
#         fig, ax = plt.subplots(1, 1, figsize=(10, 7))
#         for ik in range(1):
#             ax.plot(np.fft.fftshift(self.w_array), np.fft.fftshift(self.Skw[:, ik]), lw=lw,
#                     label=r'$ka = {:1.4f}$'.format(self.ka_values[ik]))
#
#         ax.grid(True, alpha=0.3)
#         ax.legend(loc='best', ncol=3, fontsize=fsz)
#         ax.tick_params(labelsize=fsz)
#         ax.set_yscale('log')
#         ax.set_xlim(-10, 10)
#         ax.set_ylabel(r'$S(k,\omega)$', fontsize=fsz)
#         ax.set_xlabel(r'$\omega/\omega_p$', fontsize=fsz)
#         fig.tight_layout()
#         fig.savefig(self.filename + '.png')


class RadialDistributionFunction:
    """
    Pair Distribution Function.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius.

        box_lengths : array
            Length of each side of the box.

        dump_step : int
            Dump step frequency.

        filename: str
            Name of output files.

        fldr : str
            Folder containing dumps.

        no_bins : int
            Number of bins.

        no_dumps : int
            Number of dumps.

        no_grs : int
            Number of :math: `g_{ij}(r)` pairs.

        no_species : int
            Number of species.

        ra_values : array
            Array of particles' pairs distances normalized by Wigner-Seitz radius.

        species_np: array
            Array of integers with the number of particles for each species.

        sp_names : list
            Names of particle species.

        tot_no_ptcls : int
            Total number of particles.

        gr : ndarray
            Radial Pair distribution functions. Shape=(no_bins, no_grs)
    """

    def __init__(self, params):
        self.no_bins = params.PostProcessing.rdf_nbins  # number of ka values
        self.fldr = params.Control.checkpoint_dir + '/'
        self.filename = self.fldr + "pdf_" + params.Control.fname_app
        self.dump_step = params.Control.dump_step
        self.no_dumps = int(params.Control.Nsteps / params.Control.dump_step)
        self.no_species = len(params.species)
        self.no_grs = int(params.num_species * (params.num_species + 1) / 2)
        self.tot_no_ptcls = params.total_num_ptcls

        self.no_steps = params.Control.Nsteps
        self.a_ws = params.aws
        self.dr_rdf = params.Potential.rc / self.no_bins / self.a_ws
        self.box_volume = params.box_volume / self.a_ws ** 3
        self.box_lengths = np.array([params.Lx / params.aws, params.Ly / params.aws, params.Lz / params.aws])
        self.species_np = np.zeros(self.no_species)  # Number of particles of each species
        self.sp_names = []
        self.gr = np.zeros((self.no_bins, self.no_grs))
        self.ra_values = np.zeros(self.no_bins)

        for i in range(self.no_species):
            self.species_np[i] = params.species[i].num
            self.sp_names.append(params.species[i].name)

    def save(self, rdf_hist):
        """
        Parameters
        ----------
        rdf_hist : array
            Histogram of the radial distribution function.

        """
        # Initialize all the workhorse arrays
        bin_vol = np.zeros(self.no_bins)
        pair_density = np.zeros((self.no_species, self.no_species))

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
            self.ra_values[ir] = (ir + 0.5) * self.dr_rdf

        gr_ij = 0
        for i in range(self.no_species):
            for j in range(i, self.no_species):
                if j == i:
                    pair_density[i, j] *= 2.0
                for bin in range(self.no_bins):
                    self.gr[bin, gr_ij] = (rdf_hist[bin, i, j] + rdf_hist[bin, j, i]) / (bin_vol[bin]
                                                                                         * pair_density[i, j]
                                                                                         * self.no_steps)
                gr_ij += 1

        # Save the rdf data to file.
        datafile = open(self.filename + '.out', "w")
        np.savetxt(datafile, np.c_[self.ra_values, self.gr])
        datafile.close()

        return

    def plot(self):
        """
        Plot :math: `g_{ij}(r)` and save the figure.
        """
        data = np.loadtxt(self.filename + '.out')
        self.ra_values = data[:, 0]
        self.gr = data[:, 1:]
        indx = 0
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for i in range(self.no_species):
            for j in range(i, self.no_species):
                subscript = self.sp_names[i] + self.sp_names[j]
                ax.plot(self.ra_values, self.gr[:, indx], lw=lw, label=r'$g_{ ' + subscript + '} (r)$')
                indx += 1
        ax.grid(True, alpha=0.3)
        if self.no_species > 2:
            ax.legend(loc='best', ncol=(self.no_species - 1), fontsize=fsz)
        else:
            ax.legend(loc='best', fontsize=fsz)

        ax.tick_params(labelsize=fsz)
        ax.set_ylabel(r'$g(r)$', fontsize=fsz)
        ax.set_xlabel(r'$r/a$', fontsize=fsz)
        # ax.set_ylim(0, 5)
        fig.tight_layout()
        fig.savefig(self.filename + '.png')

        return


# class TransportCoefficients:
#
#     def __init__(self, params, quantity="Electrical Conductivity"):
#         self.fldr = params.Control.checkpoint_dir + '/'
#         self.fname_app = params.Control.fname_app
#
#         self.dump_step = params.Control.dump_step
#         self.no_dumps = int(params.Control.Nsteps / params.Control.dump_step)
#         self.no_species = len(params.species)
#         self.species_np = np.zeros(self.no_species, dtype=int)
#         self.species_names = []
#         self.species_charge = np.zeros(self.no_species)
#         for i in range(self.no_species):
#             self.species_np[i] = int(params.species[i].num)
#             self.species_charge[i] = params.species[i].charge
#             self.species_names.append(params.species[i].name)
#
#         self.tot_no_ptcls = params.total_num_ptcls
#         self.wp = params.wp
#         self.a_ws = params.aws
#         self.dt = params.Control.dt
#
#         self.data = {}
#
#     def compute(self, quantity):
#
#         if quantity == "Electrical Conductivity":
#             J = ElectricCurrent(params)
#             J.plot()
#             self.data = {"Electrical Conductivity", np.trapz(J.dataframe["Total Current ACF"], x=J.dataframe["Time"])}
#
#         if quantity == "Diffusion":
#             VACF = VelocityCurrent(params)
#             whatever = np.trapz( VACF.dataframe["Total Velocity ACF"], x = VACF.dataframe["Time"])
#
#         return whatever


def load_from_restart(fldr, it):
    """
    Load particles' data from checkpoints.

    Parameters
    ----------
    fldr : str
        Folder containing dumps.
    it : int
        Timestep.

    Returns
    -------
    data : dict
        Particles' data.
    """

    file_name = fldr + "/" + "S_checkpoint_" + str(it) + ".npz"
    data = np.load(file_name, allow_pickle=True)
    return data


def kspace_setup(no_ka):
    # Obtain all possible permutations of the wave number arrays
    k_arr = [np.array([i, j, k]) for i in range(no_ka)
             for j in range(no_ka)
             for k in range(no_ka)]

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
    k_nparr = np.array(k_arr)
    return k_nparr, k_counts, k_unique


@nb.njit
def calc_Sk_single(pos_data, ka_list, ka_counts, species_np, no_dumps):
    """
    Calculate :math:`S(k)`.

    Parameters
    ----------
    pos_data : ndarray
        Particles' position scaled by the box lengths. Shape = ( `no_dumps`, 3, `tot_no_ptcls')

    ka_list : list
        List of :math:`k` indices in each direction with corresponding magnitude and index of `ka_counts`.
        Shape=(`no_ka_values`, 5)

    ka_counts : array
        Number of times each :math:`k` magnitude appears.

    species_np : array
        Array with one element giving number of particles.

    no_dumps : int
        Number of dumps.

    Returns
    -------

    Sk : ndarray
        Array containing :math:`S(k)`. Shape=(`no_ka_values`, `no_dumps`)
    """
    num_ka_values = len(ka_counts)

    Sk = np.zeros((num_ka_values, no_dumps))

    # I don't know if this will cause problem with large numbers
    for it in range(no_dumps):
        for ik in range(ka_list.shape[0]):
            kr_i = (ka_list[ik][0] * pos_data[it, 0, :]
                    + ka_list[ik][1] * pos_data[it, 1, :]
                    + ka_list[ik][2] * pos_data[it, 2, :]) * 2.0 * np.pi

            nk_i = np.sum(np.exp(-1j * kr_i))
            indx = int(ka_list[ik][-1])

            Sk[indx, it] += np.abs(nk_i) ** 2 / (ka_counts[indx] * species_np[0])

    return Sk


@nb.njit
def calc_Sk_multi(pos_data, ka_list, ka_counts, species_np, no_dumps):
    """
    Calculate all :math:`S_{ij}(k)`.

    Parameters
    ----------
    pos_data : ndarray
        Particles' position scaled by the box lengths. Shape = ( `no_dumps`, 3, `tot_no_ptcls')

    ka_list :
        List of :math:`k` indices in each direction with corresponding magnitude and index of `ka_counts`.
        Shape=(`no_ka_values`, 5)

    ka_counts : array
        Number of times each :math:`k` magnitude appears.

    species_np : array
        Array with number of particles of each species.

    no_dumps : int
        Number of dumps.

    Returns
    -------

    Sk_mat : ndarray
        Array containing :math:`S_{ij}(k)`. Shape=(`no_ka_values`, `no_sp`, `no_sp`, `no_dumps`)

    """
    no_sp = len(species_np)
    num_ka_values = len(ka_counts)

    Sk_mat = np.zeros((num_ka_values, no_sp, no_sp, no_dumps))

    # I don't know if this will cause problem with large numbers
    for it in range(no_dumps):
        for ik in range(ka_list.shape[0]):
            indx = int(ka_list[ik][-1])
            sp1_start = 0
            # Calculate density of first species
            for i in range(no_sp):
                sp1_end = sp1_start + species_np[i]
                sp2_start = sp1_start
                kr_i = 2.0 * np.pi * (ka_list[ik][0] * pos_data[it, 0, sp1_start:sp1_end]
                                      + ka_list[ik][1] * pos_data[it, 1, sp1_start:sp1_end]
                                      + ka_list[ik][2] * pos_data[it, 2, sp1_start:sp1_end])

                nk_i = np.sum(np.exp(-1j * kr_i))
                # Calculate density of second species
                for j in range(i, no_sp):
                    sp2_end = sp2_start + species_np[j]
                    kr_j = 2.0 * np.pi * (ka_list[ik][0] * pos_data[it, 0, sp2_start:sp2_end]
                                          + ka_list[ik][1] * pos_data[it, 1, sp2_start:sp2_end]
                                          + ka_list[ik][2] * pos_data[it, 2, sp2_start:sp2_end])

                    nk_j = np.sum(np.exp(1j * kr_j))

                    Sk_mat[indx, i, j, it] += np.real(nk_i * nk_j) / (
                            ka_counts[indx] * np.sqrt(species_np[i] * species_np[j]))

                    sp2_start = sp2_end

                sp1_start = sp1_end

    # for i in range(no_sp):
    #     Sk_mat[:, i, i, :] /= 2

    return Sk_mat


@nb.njit
def calc_Sk_single_pa(pos_data, ka_values, species_np, no_dumps):
    """
    Calculate :math: `S(k)` along principal axis only."

    Parameters
    ----------
    pos_data : array
        x - coordinate of all the particles.

    ka_values : array
        Array of :math: `ka` values.

    species_np : array
        Array with one element giving number of particles.

    no_dumps : int
        Number of saved checkpoints.

    Returns
    -------
    Sk : darray
        Static structure factor :math: `S_{ij}(k)`.
    """

    num_ka_values = len(ka_values)

    # Number of independent S_ij(k)  = no_sp*(no_sp + 1)/2

    # Dev Notes. I could not find a smart indexing way. So I went for the most fundamental one
    # Create a matrix of S_{ij}(k) and then add the off_diagonal together in the final Sk

    Sk = np.zeros((num_ka_values, no_dumps))

    for it in range(no_dumps):
        for (ik, ka) in enumerate(ka_values):
            krx_i = ka * pos_data[it, 0, :]
            kry_i = ka * pos_data[it, 1, :]
            krz_i = ka * pos_data[it, 2, :]

            nkx_i = np.sum(np.exp(-1j * krx_i))
            nky_i = np.sum(np.exp(-1j * kry_i))
            nkz_i = np.sum(np.exp(-1j * krz_i))

            Sk[ik, it] += (np.abs(nkx_i) ** 2 + np.abs(nky_i) ** 2 + np.abs(nkz_i) ** 2) / (3.0 * species_np[0])

    return Sk


@nb.njit
def calc_Sk_multi_pa(pos_data, ka_values, species_np, no_dumps):
    """
    Calculate all :math: `S_{ij}(k)` along the principal axis only."

    Parameters
    ----------
    pos_data : array
        x - coordinate of all the particles.

    ka_values : array
        Array of :math: `ka` values.

    species_np : array
        Array with only one element = number of particles for each species.

    no_dumps : int
        Number of saved checkpoints.

    Returns
    -------
    Sk : ndarray
        Static structure factors :math: `S_{ij}(k)`.
    """

    no_sp = len(species_np)
    num_ka_values = len(ka_values)

    # Dev Notes. I could not find a smart indexing way. So I went for the most fundamental one
    # Create a matrix of S_{ij}(k) and then add the off_diagonal together in the final Sk

    Sk_mat = np.zeros((num_ka_values, no_sp, no_sp, no_dumps))

    for it in range(no_dumps):
        for (ik, ka) in enumerate(ka_values):
            sp1_start = 0
            for i in range(no_sp):
                sp1_end = sp1_start + species_np[i]

                krx_i = ka * pos_data[it, 0, sp1_start:sp1_end]
                kry_i = ka * pos_data[it, 1, sp1_start:sp1_end]
                krz_i = ka * pos_data[it, 2, sp1_start:sp1_end]

                nkx_i = np.sum(np.exp(-1j * krx_i))
                nky_i = np.sum(np.exp(-1j * kry_i))
                nkz_i = np.sum(np.exp(-1j * krz_i))

                sp2_start = sp1_start
                for j in range(i, no_sp):
                    sp2_end = sp2_start + species_np[j]

                    krx_j = ka * pos_data[it, 0, sp2_start:sp2_end]
                    kry_j = ka * pos_data[it, 1, sp2_start:sp2_end]
                    krz_j = ka * pos_data[it, 2, sp2_start:sp2_end]

                    nkx_j = np.sum(np.exp(-1j * krx_j))
                    nky_j = np.sum(np.exp(-1j * kry_j))
                    nkz_j = np.sum(np.exp(-1j * krz_j))

                    Sk_mat[ik, i, j, it] += (np.real(nkx_i * nkx_j) + np.real(nky_i * nky_j) + np.real(nkz_i * nkz_j)
                                             ) / (3.0 * np.sqrt(species_np[i] * species_np[j]))

                    sp2_start = sp2_end

                sp1_start = sp1_end

    return Sk_mat


@nb.njit
def calc_elec_current(vel, sp_charge, sp_num):
    """
    Calcualte the total electric current and the electric current of each species.

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
        Electric current of each species.

    Jtot : ndarray
        Total electric current.
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


@nb.njit
def autocorrelationfunction(At):
    """
    Calculate the autocorrelation function of the input.

    Parameters
    ----------
    At : array
        Observable to autocorrelate. Shape=(ndim, nsteps).

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


@nb.njit
def autocorrelationfunction_1D(At):
    """
    Calculate the autocorrelation function of the input.

    Parameters
    ----------
    At : array
        Observable to autocorrelate. Shape=(nsteps).

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


@nb.njit
def calc_pressure_tensor(pos, vel, acc, species_mass, species_np, box_volume):
    no_dim = pos.shape[0]
    pressure_tensor = np.zeros((no_dim, no_dim))
    sp_start = 0
    sp_end = 0
    # Rescale vel and acc of each particle by their individual mass
    for sp in range(len(species_np)):
        sp_end = sp_start + species_np[sp]
        vel[:, sp_start: sp_end] *= np.sqrt(species_mass[sp])
        acc[:, sp_start: sp_end] /= species_mass[sp]
        sp_start = sp_end

    pressure = 0.0
    for i in range(no_dim):
        for j in range(no_dim):
            pressure_tensor[i, j] = np.sum(vel[i, :] * vel[j, :] + pos[i, :] * acc[j, :]) / box_volume
        pressure += pressure_tensor[i, i] / 3.0

    return pressure, pressure_tensor


@nb.njit
def calc_nkt_single_pa(pos_data, ka_values, species_np, no_dumps):
    """
    Calculate :math: `n(k,t)' along the principal axis only."

    Parameters
    ----------
    pos_data : array
        x - coordinate of all the particles.

    ka_values : array
        Array of :math: `ka` values.

    species_np : array
        Number of particles.

    no_dumps : int
        Number of dumps from simulation.

    Returns
    -------
    nkt : ndarray
        Density :math: `n(k_x, k_y, k_z, t)/\sqrt(N)`.
        Shape=( `num_ka_values`, 3, `no_dumps`)
    """
    num_ka_values = len(ka_values)
    nkt = np.zeros((num_ka_values, 3, no_dumps), dtype=np.complex128)

    # Calculate nk(t)
    for it in range(no_dumps):
        for (ik, ka) in enumerate(ka_values):
            krx_i = ka * pos_data[it, 0, :]
            kry_i = ka * pos_data[it, 1, :]
            krz_i = ka * pos_data[it, 2, :]

            nkt[ik, 0, it] = np.sum(np.exp(-1j * krx_i)) / np.sqrt(species_np[0])
            nkt[ik, 1, it] = np.sum(np.exp(-1j * kry_i)) / np.sqrt(species_np[0])
            nkt[ik, 2, it] = np.sum(np.exp(-1j * krz_i)) / np.sqrt(species_np[0])

    return nkt


@nb.njit
def calc_nkt_multi_pa(pos_data, ka_min, num_ka_values, species_np, no_dumps):
    """
    Calculate all :math: `S_{ij}(k)` in case of multi-species simulation."

    Parameters
    ----------
    pos_data : array
        x - coordinate of all the particles.

    ka_min : float
        Smallest possible (non-dimensional) wavenumber :math:`ka = 2\pi/L`.

    num_ka_values : int
        Number of :math: `ka` values to compute.

    species_np : array
        Array of integers with the number of particles for each species.

    Returns
    -------
    ka_values : array
        Array of :math: `ka` values.

    Sk : ndarray
        Static structure factors :math: `S_{ij}(k)`.
    """

    no_sp = len(species_np)
    # Number of independent S_ij(k)  = no_sp*(no_sp + 1)/2
    no_Sk = int(no_sp * (no_sp + 1) / 2)

    Sk = np.zeros((num_ka_values, no_Sk, no_dumps))

    ka_values = np.zeros(num_ka_values)
    for ik in range(num_ka_values):
        ka_values[ik] = (ik + 1) * ka_min

    for it in range(no_dumps):
        for (ik, ka) in enumerate(ka_values):
            sp1_start = 0
            for i in range(no_sp):
                sp1_end = sp1_start + species_np[i]
                sp2_start = 0
                krx_i = ka * pos_data[it, 0, sp1_start:sp1_end]
                kry_i = ka * pos_data[it, 1, sp1_start:sp1_end]
                krz_i = ka * pos_data[it, 2, sp1_start:sp1_end]

                nkx_i = np.sum(np.exp(-1j * krx_i))
                nky_i = np.sum(np.exp(-1j * kry_i))
                nkz_i = np.sum(np.exp(-1j * krz_i))

                for j in range(no_sp):
                    sp2_end = sp2_start + species_np[j]
                    krx_j = ka * pos_data[it, 0, sp2_start:sp2_end]
                    kry_j = ka * pos_data[it, 1, sp2_start:sp2_end]
                    krz_j = ka * pos_data[it, 2, sp2_start:sp2_end]

                    nkx_j = np.sum(np.exp(1j * krx_j))
                    nky_j = np.sum(np.exp(1j * kry_j))
                    nkz_j = np.sum(np.exp(1j * krz_j))

                    indx = i * (no_sp - 1) + j
                    if i == j:
                        degeneracy = 1.0
                    else:
                        degeneracy = 2.0
                    Sk[ik, indx, it] += (np.real(nkx_i * nkx_j) +
                                         np.real(nky_i * nky_j) +
                                         np.real(nkz_i * nkz_j)
                                         ) / (3.0 * degeneracy * np.sqrt(species_np[i] * species_np[j]))

                    sp2_start = sp2_end

                sp1_start = sp1_end

    return ka_values, Sk


@nb.njit
def calc_nkt_single(pos_data, ka_list, ka_counts, species_np, no_dumps):
    """
    Calculate :math:`S(k)`.

    Parameters
    ----------
    pos_data : ndarray
        Particles' position scaled by the box lengths. Shape = ( `no_dumps`, 3, `tot_no_ptcls')

    ka_list : list
        List of :math:`k` indices in each direction with corresponding magnitude and index of `ka_counts`.
        Shape=(`no_ka_values`, 5)

    ka_counts : array
        Number of times each :math:`k` magnitude appears.

    species_np : array
        Array with one element giving number of particles.

    no_dumps : int
        Number of dumps.

    Returns
    -------

    Sk : ndarray
        Array containing :math:`S(k)`. Shape=(`no_ka_values`, `no_dumps`)
    """
    num_ka_values = len(ka_counts)

    nkt = np.zeros((num_ka_values, no_dumps), dtype=np.complex128)

    # I don't know if this will cause problem with large numbers
    for it in range(no_dumps):
        for ik in range(ka_list.shape[0]):
            kr_i = (ka_list[ik][0] * pos_data[it, 0, :]
                    + ka_list[ik][1] * pos_data[it, 1, :]
                    + ka_list[ik][2] * pos_data[it, 2, :]) * 2.0 * np.pi

            indx = int(ka_list[ik][-1])
            nkt[indx, it] += np.sum(np.exp(-1j * kr_i)) / (np.sqrt(ka_counts[indx] * species_np[0]) )

    return nkt


@nb.njit
def calc_nkt_multi(stuff, stuhh, sana):
    """

    Parameters
    ----------
    stuff
    stuhh
    sana

    Returns
    -------

    """
    pass
