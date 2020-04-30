"""
Module for calculating physical quantities from Sarkas dumps
"""
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from matplotlib import rc

rc('text', usetex=True)

from S_particles import Particles

lw = 2
fsz = 14
msz = 8


class StaticStructureFactor:
    """ Static Structure factor.

    Attributes
    ----------
        a_ws : float
            Wigner-Seitz radius.

        dump_step : int
            Dump step frequency.

        filename: str
            Name of output files.

        fldr : str
            Folder containing dumps.

        ka_min : float
            Smallest possible (non-dimensional) wavenumber :math:`ka = 2\pi/L`.

        ka_max : float
            Largest possible (non-dimensional) wavenumber = ``no_ka * ka_min``

        no_dumps : int
            Number of dumps.

        no_ka : int
            Number of integer multiples of minimum :math:`ka` value

        no_species : int
            Number of species.

        np_species: array
            Array of integers with the number of particles for each species.

        ptcls : class
            ``S_particles.py`` class with particles positions.

        sp_names : list
            Names of particle species.

        tot_no_ptcls : int
            Total number of particles.

        """

    def __init__(self, params):
        self.no_ka = 50  # number of ka values
        self.fldr = params.Control.checkpoint_dir + '/'
        self.filename = self.fldr + "Sk_" + params.Control.fname_app
        self.dump_step = params.Control.dump_step
        self.no_dumps = int(params.Control.Nsteps / params.Control.dump_step)
        self.no_species = len(params.species)
        self.tot_no_ptcls = params.total_num_ptcls

        self.no_Sk = int(self.no_species * (self.no_species + 1) / 2)
        self.a_ws = params.aws
        self.ka_min = 2.0 * np.pi * params.aws / params.Lx
        self.ka_max = self.no_ka * self.ka_min

        self.np_species = np.zeros(self.no_species)
        self.sp_names = []
        for i in range(self.no_species):
            self.np_species[i] = params.species[i].num
            self.sp_names.append(params.species[i].name)
        self.ptcls = Particles(params)

    def compute(self):
        """
        Compute :math: `S_{ij}(k)' .

        Returns
        -------
        ka_values : array
            Wavenumber values.

        Sk_mean : ndarray
            Time Average of each :math: `S_{ij}(k)'.

        Sk_std : ndarray
             Time Standard deviation of each :math: `S_{ij}(k)'.

        """
        # Dev Note: The first index is the value of ka,
        # The second index indicates S_ij
        # The third index indicates S_ij(t)

        # Parse the particles from the dump files
        pos = np.zeros((self.no_dumps, 3, self.tot_no_ptcls))
        if self.no_Sk == 1:
            calculate = calc_Sk_single
        else:
            calculate = calc_Sk_multi

        for it in range(self.no_dumps):
            dump = int(it * self.dump_step)
            self.ptcls.load_from_restart(dump)
            pos[it, 0, :] = self.ptcls.pos[:, 0] / self.a_ws
            pos[it, 1, :] = self.ptcls.pos[:, 1] / self.a_ws
            pos[it, 2, :] = self.ptcls.pos[:, 2] / self.a_ws

        ka_values, Sk_all = calculate(pos, self.ka_min, self.no_ka, self.np_species)
        Sk_mean = np.mean(Sk_all, axis=2)
        Sk_std = np.std(Sk_all, axis=2)

        datafile = open(self.filename + '.out', "w")
        np.savetxt(datafile, np.c_[ka_values, Sk_mean, Sk_std])
        datafile.close()

        return ka_values, Sk_mean, Sk_std

    def plot(self, errorbars=False):
        """
        Plot S(k) and save the figure.

        Parameters
        ----------
        errorbars : bool
            Plot errorbars. Default = False.

        """

        ka_values, Sk, Sk_err = self.compute()

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for i in range(self.no_species):
            for j in range(i, self.no_species):
                indx = i*(self.no_species - 1) + j
                subscript = self.sp_names[i] + self.sp_names[j]
                if errorbars:
                    ax.errorbar(ka_values, Sk[:, indx], yerr=Sk_err[:, indx], lw=lw, ls='--', marker='o', ms=msz,
                                label=r'$S_{ ' + subscript + '} (k)$')
                else:
                    ax.plot(ka_values, Sk[:, indx], lw=lw, label=r'$S_{ ' + subscript + '} (k)$')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=fsz)
        ax.tick_params(labelsize=fsz)
        ax.set_ylabel(r'$S(k)$', fontsize=fsz)
        ax.set_xlabel(r'$ka$', fontsize=fsz)
        fig.tight_layout()
        fig.savefig(self.filename + '.png')


@nb.njit
def calc_Sk_single(pos_data, ka_min, num_ka_values, Np):
    """
    Calculate all S_{ij}(k) in case of a single species simulation."

    Parameters
    ----------
    pos_data : array
        x - coordinate of all the particles.

    ka_min : float
        Smallest possible (non-dimensional) wavenumber :math:`ka = 2\pi/L`.

    num_ka_values : int
        Number of :math: `ka` values to compute.

    Np : array
        Array with only one element = number of particles for each species.

    Returns
    -------
    ka_values : array
        Array of :math: `ka` values.

    Sk : ndarray
        Static structure factors :math: `S_{ij}(k)`.
    """
    no_Sk = 1
    no_dumps = pos_data.shape[0]
    Sk = np.zeros((num_ka_values, no_Sk, no_dumps))
    ka_values = np.zeros(num_ka_values)
    # Ideally this would be three nested loops over kx, ky,kz
    # This is left for future development
    for ik in range(num_ka_values):
        ka_values[ik] = (ik + 1) * ka_min

    # I don't know if this will cause problem with large numbers
    for it in range(no_dumps):

        for (ik, ka) in enumerate(ka_values):
            krx_i = ka * pos_data[it, 0, :]
            kry_i = ka * pos_data[it, 1, :]
            krz_i = ka * pos_data[it, 2, :]

            nk_i = np.sum(np.exp(-1j * krx_i) + np.exp(-1j * kry_i) + np.exp(-1j * krz_i))
            # Dev Note: Numba is giving error if using np.conjugate
            Sk[ik, 0, it] += abs(nk_i) ** 2 / (3.0 * Np[0])

    return ka_values, Sk


@nb.njit
def calc_Sk_multi(pos_data, ka_min, num_ka_values, species_np):
    """
    Calculate all S_{ij}(k) in case of multi-species simulation."

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
    no_dumps = pos_data.shape[0]

    Sk = np.zeros((num_ka_values, no_Sk, no_dumps))

    ka_values = np.zeros(num_ka_values)
    for ik in range(num_ka_values):
        ka_values[ik] = (ik + 1) * ka_min

    for it in range( no_dumps):
        for (ik, ka) in enumerate( ka_values):
            sp1_start = 0
            for i in range(no_sp):
                sp1_end = sp1_start + species_np[i]
                sp2_start = 0
                krx_i = ka * pos_data[it, 0, sp1_start:sp1_end]
                kry_i = ka * pos_data[it, 1, sp1_start:sp1_end]
                krz_i = ka * pos_data[it, 2, sp1_start:sp1_end]

                nk_i = np.sum(np.exp(-1j * krx_i) + np.exp(-1j * kry_i) + np.exp(-1j * krz_i))

                for j in range(no_sp):
                    sp2_end = sp2_start + species_np[j]
                    krx_j = ka * pos_data[it, 0, sp2_start:sp2_end]
                    kry_j = ka * pos_data[it, 1, sp2_start:sp2_end]
                    krz_j = ka * pos_data[it, 2, sp2_start:sp2_end]

                    nk_j = np.sum(np.exp(1j * krx_j) + np.exp(1j * kry_j) + np.exp(1j * krz_j))

                    indx = i*(no_sp - 1) + j
                    if i == j:
                        degeneracy = 1.0
                    else:
                        degeneracy = 2.0
                    Sk[ik, indx,it] += abs(nk_i * nk_j) / (3.0 * degeneracy * np.sqrt(species_np[i] * species_np[j]))

                    sp2_start += sp2_end

                sp1_start += sp1_end

    return ka_values, Sk


class EnergyTemperature:
    """ Energies and Temperatures.

    Attributes
    ----------
        dump_step : int
            Dump step frequency.

        filename: str
            Name of output files.

        fldr : str
            Folder containing dumps.

        no_dumps : int
            Number of dumps.

        no_species : int
            Number of species.

        np_species: array
            Array of integers with the number of particles for each species.

        sp_names : list
            Names of particle species.

        """

    def __init__(self, params):
        self.fldr = params.Control.checkpoint_dir + '/'
        self.filename = self.fldr + "Energy_" + params.Control.fname_app
        self.dump_step = params.Control.dump_step
        self.no_dumps = int(params.Control.Nsteps / params.Control.dump_step)
        self.no_species = len(params.species)
        self.np_species = np.zeros(self.no_species)
        self.sp_names = []
        for i in range(self.no_species):
            self.np_species[i] = params.species[i].num
            self.sp_names.append(params.species[i].name)

        data = np.loadtxt(self.filename + '.out')

        self.time = data[:, 0]
        self.temperature = data[:, 1]
        self.energy = data[:, 2]
        self.kinetic = data[:, 3]
        self.potential = data[:, 4]
        self.sp_temps = data[:, 5:(5 + self.no_species)]
        indx = 5 + self.no_species
        self.sp_kinetic = data[:, indx:(indx + self.no_species)]

        self.Gamma = params.qe ** 2 / (params.fourpie0 * params.kB * params.aws * self.temperature)

    def plot(self, observable, Delta):
        """
        Plot either temperature or energy

        Parameters
        ----------
        observable : str
            "energy" or "temperature".

        Delta : bool
            True = Plot the relative error of ``observable`` vs time.

        """

        if observable == 'energy':
            Y = self.energy
            ylbl = r'$E$ [J]'
        elif observable == 'temperature':
            Y = self.temperature
            ylbl = r'$T$ [K]'
        elif observable == 'Gamma':
            Y = self.Gamma
            ylbl = r'$\Gamma$'

        if Delta:
            DeltaY = (Y[:] - Y[0]) / Y[0]
            Y = DeltaY

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.plot(self.time, Y, lw=lw, ls='--', marker='o', ms=msz)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=fsz)
        ax.set_ylabel(ylbl, fontsize=fsz)
        ax.set_xlabel(r'$t$', fontsize=fsz)
        fig.tight_layout()
        fig.savefig(self.filename + '.png')

        return


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

        no_species : int
            Number of species.

        np_species: array
            Array of integers with the number of particles for each species.

        sp_names : list
            Names of particle species.

        tot_no_ptcls : int
            Total number of particles.

        ptcls : class
            ``S_particles.py`` class with particles positions.
    """

    def __init__(self, params):
        self.no_bins = params.PostProcessing.rdf_nbins # number of ka values
        self.fldr = params.Control.checkpoint_dir + '/'
        self.filename = self.fldr + "pdf_" + params.Control.fname_app
        self.dump_step = params.Control.dump_step
        self.no_dumps = int(params.Control.Nsteps / params.Control.dump_step)
        self.no_species = len(params.species)
        self.no_grs = int(params.num_species*(params.num_species + 1)/2)
        self.tot_no_ptcls = params.total_num_ptcls

        self.no_steps = params.Control.Nsteps
        self.a_ws = params.aws
        self.dr_rdf = params.Potential.rc/self.no_bins/self.a_ws
        self.box_volume = params.box_volume/self.a_ws**3
        self.box_lengths = np.array([params.Lx / params.aws, params.Ly / params.aws, params.Lz / params.aws])
        self.species_np = np.zeros(self.no_species) # Number of particles of each species
        self.sp_names = []

        for i in range(self.no_species):
            self.species_np[i] = params.species[i].num
            self.sp_names.append(params.species[i].name)


    def save(self, rdf_hist):
        """
        Parameters
        ----------
        rdf_hist : array
            Histogram of the radial distribution function

        """

        rdf_norm = np.zeros((self.no_bins, self.no_grs))
        ra_values = np.zeros(self.no_bins)
        bin_vol = np.zeros(self.no_bins)
        pair_density = np.zeros((self.no_species,self.no_species))

        for i in range(self.no_species):
            pair_density[i, i] = self.species_np[i] * (self.species_np[i] - 1) / (2.0 * self.box_volume)
            for j in range(i + 1, self.no_species):
                pair_density[i, j] = self.species_np[i] * self.species_np[j] / (2.0 * self.box_volume)

        sphere_shell_const = 4.0 * np.pi / 3.0
        r0 = 0.5 * self.dr_rdf
        bin_vol[0] = sphere_shell_const * r0 ** 3
        for ir in range(1, self.no_bins):
            r1 = (ir - 0.5) * self.dr_rdf
            r2 = (ir + 0.5) * self.dr_rdf
            bin_vol[ir] = sphere_shell_const * (r2 ** 3 - r1 ** 3)
            ra_values[ir] = (ir - 0.5) * self.dr_rdf

        for i in range(self.no_species):
            for j in range(i, self.no_species):
                gr_ij = i * (self.no_species - 1) + j
                for bin in range(self.no_bins):
                    rdf_norm[bin, gr_ij] = rdf_hist[bin, gr_ij] / (bin_vol[bin] * pair_density[i, j]*self.no_steps)

        datafile = open(self.filename + '.out', "w")
        np.savetxt(datafile, np.c_[ra_values, rdf_norm])
        datafile.close()

        return


    def plot(self):
        """
        Plot g(r) and save the figure.

        Parameters
        ----------
        errorbars : bool
            Plot errorbars. Defaul = False.

        """
        gr = np.zeros( (self.no_bins, self.no_grs) )
        data = np.loadtxt( self.filename + '.out')
        ra_values = data[:,0]
        gr = data[:,1:]

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for i in range(self.no_species):
            for j in range(i, self.no_species):
                indx = i*(self.no_species -1 ) + j
                subscript = self.sp_names[i] + self.sp_names[j]
                ax.plot(ra_values, gr[:, indx], lw=lw, label=r'$g_{ ' + subscript + '} (r)$')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=fsz)
        ax.tick_params(labelsize=fsz)
        ax.set_ylabel(r'$g(r)$', fontsize=fsz)
        ax.set_xlabel(r'$r/a$', fontsize=fsz)
        fig.tight_layout()
        fig.savefig(self.filename + '.png')