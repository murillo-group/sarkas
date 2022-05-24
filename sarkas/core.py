"""
Module containing the three basic classes: Parameters, Particles, Species.
"""

from copy import copy as py_copy
from numpy import arange, array, ceil, count_nonzero, cross, empty, floor
from numpy import load as np_load
from numpy import loadtxt, meshgrid, ndarray, pi, sqrt, triu_indices, zeros
from numpy.random import Generator, PCG64
from os.path import join
from scipy.constants import physical_constants
from scipy.linalg import norm
from scipy.spatial.distance import pdist
from warnings import warn

from .utilities.exceptions import ParticlesError, ParticlesWarning


class Parameters:
    """
    Class containing all the constants and physical constants of the simulation.

    Parameters
    ----------
    dic : dict, optional
        Dictionary to be copied.

    Attributes
    ----------
    a_ws : float
        Wigner-Seitz radius. Calculated from the ``total_num_density`` .

    boundary_conditions : str
        Type of boundary conditions.

    box_volume : float
        Volume of simulation box.

    pbox_volume : float
        Volume of initial particle box.

    dimensions : int
        Number of non-zero dimensions. Default = 3.

    fourpie0: float
        Electrostatic constant :math:`4\\pi \\epsilon_0`.

    num_species : int
        Number of species.

    kB : float
        Boltzmann constant obtained from ``scipy.constants``.

    hbar : float
        Reduced Planck's constant.

    hbar2 : float
        Square of reduced Planck's constant.

    a0 : float
        Bohr Radius.

    c0 : float
        Speed of light.

    qe : float
        Elementary charge.

    me : float
        Electron mass.

    eps0 : float
        Vacuum electrical permittivity.

    eV2K : float
        Conversion factor from eV to Kelvin obtained from ``scipy.constants``.

    J2erg : float
        Conversion factor from Joules to erg. Needed for cgs units.

    QFactor : float
        Charge Factor defined as :math:`\mathcal Q = \sum_{i}^{N} q_{i}^2` .

    Lx : float
        Box length in the :math:`x` direction.

    Ly : float
        Box length in the :math:`y` direction.

    Lz : float
        Box length in the :math:`z` direction.

    e1 : float
        Unit vector in the :math:`x` direction.

    e2 : float
        Unit vector in the :math:`y` direction.

    e3 : float
        Unit vector in the :math:`z` direction.

    LPx : float
        Initial particle box length in the :math:`x` direction.

    LPy : float
        Initial particle box length in the :math:`y` direction.

    LPz : float
        Initial particle box length in the :math:`z` direction.

    ep1 : float
        Unit vector of the initial particle box in the :math:`x` direction.

    ep2 : float
        Unit vector of the initial particle box in the :math:`y` direction.

    ep3 : float
        Unit vector of the initial particle box in the :math:`z` direction.

    input_file : str
        YAML Input file with all the simulation's parameters.

    T_desired : float
        Target temperature for the equilibration phase.

    species_num : numpy.ndarray
        Number of particles of each species. Shape = (``num_species``)

    species_concentrations : numpy.ndarray
        Concentration of each species. Shape = (``num_species``)

    species_temperatures : numpy.ndarray
        Initial temperature of each species. Shape = (``num_species``)

    species_masses : numpy.ndarray
        Mass of each species. Shape = (``num_species``)

    species_charges : numpy.ndarray
        Charge of each species. Shape = (``num_species``)

    species_names : list
        Name of each species. Len = (``num_species``)

    species_plasma_frequencies : numpy.ndarray
        Plasma Frequency of each species. Shape = (``num_species``)

    species_num_dens : numpy.ndarray
        Number density of each species. Shape = (``num_species``)

    total_ion_temperature : float
        Total initial ion temperature calculated as `` = species_concentration @ species_temperatures``.

    total_net_charge : float
        Total charge in the system.

    total_num_density : float
        Total number density. Calculated from the sum of ``Species.num_density``.

    total_num_ptcls : int
        Total number of particles. Calculated from the sum of ``Species.num``.

    measure : bool
        Flag for production phase.

    verbose : bool
        Flag for screen output.

    simulations_dir : str
        Name of directory where to store simulations.

    job_dir : str
        Directory name of the current job/run

    production_dir : str
        Directory name where to store simulation's files of the production phase. Default = 'Production'.

    equilibration_dir : str
        Directory name where to store simulation's file of the equilibration phase. Default = 'Equilibration'.

    preprocessing_dir : str
        Directory name where to store preprocessing files. Default = "PreProcessing".

    postprocessing_dir : str
        Directory name where to store postprocessing files. Default = "PostProcessing".

    prod_dump_dir : str
        Directory name where to store production phase's simulation's checkpoints. Default = 'dumps'.

    eq_dump_dir : str
        Directory name where to store equilibration phase's simulation's checkpoints. Default = 'dumps'.

    job_id : str
        Appendix of all simulation's files.

    log_file : str
        Filename of the simulation's log.

    np_per_side : numpy.ndarray
        Number of particles per simulation's box side.
        The product of its components should be equal to ``total_num_ptcls``.

    pre_run : bool
        Flag for preprocessing phase.

    """

    def __init__(self, dic: dict = None) -> None:

        self.particles_input_file = None
        self.load_perturb = None
        self.load_rejection_radius = None
        self.load_halton_bases = None
        self.load_method = None
        self.rs = None
        self.eta_e = None
        self.wdm_parameter = None
        self.electron_cyclotron_frequency = None
        self.relativistic_parameter = None
        self.fermi_energy = None
        self.electron_coupling = None
        self.electron_degeneracy_parameter = None
        self.kF = None
        self.lambda_TF = None
        self.lambda_deB = None
        self.electron_temperature = None
        self.ae_ws = None
        self.ne = None
        self.potential_type = None
        self.units = None
        self.electron_magnetic_energy = None
        self.input_file = None

        # Sim box geometry
        self.Lx = 0.0
        self.Ly = 0.0
        self.Lz = 0.0
        self.LPx = 0.0
        self.LPy = 0.0
        self.LPz = 0.0
        self.e1 = None
        self.e2 = None
        self.e3 = None
        self.ep1 = None
        self.ep2 = None
        self.ep3 = None
        self.box_lengths = None
        self.pbox_lengths = None
        self.box_volume = 0.0
        self.pbox_volume = 0.0
        self.dimensions = 3

        # Physical Constants and conversion units
        self.J2erg = 1.0e7  # erg/J
        self.eps0 = physical_constants["vacuum electric permittivity"][0]
        self.fourpie0 = 4.0 * pi * self.eps0
        self.mp = physical_constants["proton mass"][0]
        self.me = physical_constants["electron mass"][0]
        self.qe = physical_constants["elementary charge"][0]
        self.hbar = physical_constants["reduced Planck constant"][0]
        self.hbar2 = self.hbar**2
        self.c0 = physical_constants["speed of light in vacuum"][0]
        self.eV2K = physical_constants["electron volt-kelvin relationship"][0]
        self.eV2J = physical_constants["electron volt-joule relationship"][0]
        self.a0 = physical_constants["Bohr radius"][0]
        self.kB = physical_constants["Boltzmann constant"][0]
        self.kB_eV = physical_constants["Boltzmann constant in eV/K"][0]
        self.a_ws = 0.0

        # Control and Timing
        self.boundary_conditions = "periodic"
        self.job_id = None
        self.job_dir = None
        self.log_file = None
        self.measure = False
        self.magnetized = False
        self.plot_style = None
        self.pre_run = False
        self.electrostatic_equilibration = False
        self.simulations_dir = "Simulations"
        self.production_dir = "Production"
        self.magnetization_dir = "Magnetization"
        self.equilibration_dir = "Equilibration"
        self.preprocessing_dir = "PreProcessing"
        self.postprocessing_dir = "PostProcessing"
        self.prod_dump_dir = "dumps"
        self.eq_dump_dir = "dumps"
        self.mag_dump_dir = "dumps"
        self.verbose = True
        self.restart_step = None
        self.np_per_side = None
        self.num_species = 1
        self.magnetic_field = None
        self.species_lj_sigmas = None
        self.species_names = None
        self.species_num = None
        self.species_num_dens = None
        self.species_concentrations = None
        self.species_temperatures = None
        self.species_temperatures_eV = None
        self.species_masses = None
        self.species_charges = None
        self.species_plasma_frequencies = None
        self.species_cyclotron_frequencies = None
        self.species_couplings = None

        self.coupling_constant = 0.0
        self.total_num_density = 0.0
        self.total_num_ptcls = 0
        self.total_plasma_frequency = 0.0
        self.total_debye_length = 0.0
        self.total_mass_density = 0.0
        self.total_ion_temperature = 0.0
        self.T_desired = 0.0
        self.total_net_charge = 0.0
        self.QFactor = 0.0

        self.average_charge = None
        self.average_mass = None
        self.hydrodynamic_frequency = None

        if dic:
            self.from_dict(dic)

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Parameters( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def from_dict(self, input_dict: dict) -> None:
        """
        Update attributes from input dictionary.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """
        self.__dict__.update(input_dict)

    def setup(self, species) -> None:
        """
        Setup simulations' parameters.

        Parameters
        ----------
        species : list
            List of :class:`sarkas.core.Species` objects.

        """
        self.check_units()
        self.calc_parameters(species)
        self.calc_coupling_constant(species)

    def check_units(self) -> None:
        """Adjust default physical constants for cgs unit system and check for LJ potential."""
        # Physical constants
        if self.units == "cgs":
            self.kB *= self.J2erg
            self.c0 *= 1e2  # cm/s
            self.mp *= 1e3
            # Coulomb to statCoulomb conversion factor. See https://en.wikipedia.org/wiki/Statcoulomb
            C2statC = 1.0e-01 * self.c0
            self.hbar = self.J2erg * self.hbar
            self.hbar2 = self.hbar**2
            self.qe *= C2statC
            self.me *= 1.0e3
            self.eps0 = 1.0
            self.fourpie0 = 1.0
            self.a0 *= 1e2

        if self.potential_type == "lj":
            self.fourpie0 = 1.0
            self.species_lj_sigmas = zeros(self.num_species)

    def calc_parameters(self, species: list):
        """
        Assign the parsed parameters.

        Parameters
        ----------
        species : list
            List of :class:`sarkas.core.Species` .

        """

        self.num_species = len(species)
        # Initialize the arrays containing species attributes. This is needed for postprocessing
        self.species_names = []
        self.species_num = zeros(self.num_species, dtype=int)
        self.species_num_dens = zeros(self.num_species)
        self.species_concentrations = zeros(self.num_species)
        self.species_temperatures = zeros(self.num_species)
        self.species_temperatures_eV = zeros(self.num_species)
        self.species_masses = zeros(self.num_species)
        self.species_charges = zeros(self.num_species)
        self.species_plasma_frequencies = zeros(self.num_species)
        self.species_cyclotron_frequencies = zeros(self.num_species)
        self.species_couplings = zeros(self.num_species)

        # Loop over species and assign missing attributes
        # Collect species properties in single arrays
        wp_tot_sq = 0.0
        lambda_D = 0.0

        if self.magnetized:
            self.magnetic_field = array(self.magnetic_field, dtype=float)
        # Initialization of attributes
        self.total_num_ptcls = 0
        self.total_num_density = 0.0
        # Calculate species attributes and add them to simulation parameters
        for i, sp in enumerate(species):
            self.total_num_ptcls += sp.num

            # Calculate the mass of the species from the atomic weight if given
            if sp.atomic_weight:
                # Choose between atomic mass constant or proton mass
                # u = const.physical_constants["atomic mass constant"][0]
                sp.mass = self.mp * sp.atomic_weight

            # Calculate the mass of the species from the mass density if given
            if sp.mass_density:
                Av = physical_constants["Avogadro constant"][0]
                sp.number_density = sp.mass_density * Av / sp.atomic_weight
                self.total_num_density += sp.number_density
            else:
                self.total_num_density += sp.number_density

            if not hasattr(sp, "number_density"):
                raise AttributeError(f"\nSpecies {sp.name} number density not defined")

            # Update arrays of species information
            self.species_names.append(sp.name)
            self.species_num[i] = sp.num
            self.species_masses[i] = sp.mass
            # Calculate the temperature in K if eV has been provided and vice versa
            if sp.temperature_eV:
                sp.temperature = self.eV2K * sp.temperature_eV
                self.species_temperatures_eV[i] = sp.temperature_eV
            else:
                # Convert to eV and save
                sp.temperature_eV = sp.temperature / self.eV2K
                self.species_temperatures_eV[i] = sp.temperature_eV

            self.species_temperatures[i] = sp.temperature

            # Calculate the species charge based on the inputs
            if sp.charge:
                self.species_charges[i] = sp.charge
                sp.Z = sp.charge / self.qe
            elif sp.Z:
                self.species_charges[i] = sp.Z * self.qe
                sp.charge = sp.Z * self.qe
            elif sp.epsilon:
                # Lennard-Jones potentials don't have charge but have the equivalent epsilon.
                sp.charge = sqrt(sp.epsilon)
                sp.Z = 1.0
                self.species_charges[i] = sqrt(sp.epsilon)
            else:
                sp.charge = 0.0
                sp.Z = 0.0
                self.species_charges[i] = 0.0

            # Calculate the (total) plasma frequency, QFactor, debye_length
            if not self.potential_type == "lj":
                # Q^2 factor see eq.(2.10) in Ballenegger et al. J Chem Phys 128 034109 (2008)
                sp.QFactor = sp.num * sp.charge**2
                self.QFactor += sp.QFactor / self.fourpie0

                sp.calc_plasma_frequency(self.fourpie0)
                wp_tot_sq += sp.plasma_frequency**2
                sp.calc_debye_length(self.kB, self.fourpie0)
                lambda_D += sp.debye_length**2
            else:
                sp.QFactor = 0.0
                self.QFactor += sp.QFactor / self.fourpie0
                constant = 4.0 * pi * sp.number_density * sp.sigma**2
                sp.calc_plasma_frequency(constant)
                wp_tot_sq += sp.plasma_frequency**2
                sp.calc_debye_length(self.kB, constant)
                lambda_D += sp.debye_length**2
                self.species_lj_sigmas[i] = sp.sigma

            # Calculate cyclotron frequency in case of a magnetized simulation
            if self.magnetized:

                if self.units == "cgs":
                    sp.calc_cyclotron_frequency(norm(self.magnetic_field) / self.c0)
                else:
                    sp.calc_cyclotron_frequency(norm(self.magnetic_field))

                sp.beta_c = sp.cyclotron_frequency / sp.plasma_frequency
                self.species_cyclotron_frequencies[i] = sp.cyclotron_frequency

            self.species_plasma_frequencies[i] = sp.plasma_frequency
            self.species_num_dens[i] = sp.number_density

        self.total_mass_density = self.species_masses.transpose() @ self.species_num_dens

        # Calculate species concentrations
        for i, sp in enumerate(species):
            sp.concentration = float(sp.num / self.total_num_ptcls)
            self.species_concentrations[i] = float(sp.num / self.total_num_ptcls)
            # Calculate species mass properties
            if sp.mass_density is None:
                sp.mass_density = sp.concentration * self.total_mass_density
                if sp.atomic_weight is None:
                    sp.atomic_weight = sp.mass / self.mp

        # Calculate total quantities
        self.total_net_charge = (self.species_charges.transpose()) @ self.species_num
        self.total_ion_temperature = (self.species_concentrations.transpose()) @ self.species_temperatures
        self.total_plasma_frequency = sqrt(wp_tot_sq)
        self.total_debye_length = sqrt(lambda_D)

        self.average_charge = (self.species_charges.transpose()) @ self.species_concentrations
        self.average_mass = (self.species_masses.transpose()) @ self.species_concentrations
        # Hydrodynamic Frequency aka Virtual Average Atom
        self.hydrodynamic_frequency = sqrt(
            4.0 * pi * self.average_charge**2 * self.total_num_density / (self.fourpie0 * self.average_mass)
        )

        # Simulation Box Parameters
        # Wigner-Seitz radius calculated from the total number density
        self.a_ws = (3.0 / (4.0 * pi * self.total_num_density)) ** (1.0 / 3.0)

        # Calculate initial particle's and simulation's box parameters
        if self.np_per_side:
            if int(self.np_per_side.prod()) != self.total_num_ptcls:
                raise ParticlesError("Number of particles per dimension does not match total number of particles.")

            self.LPx = self.a_ws * self.np_per_side[0] * (4.0 * pi / 3.0) ** (1.0 / 3.0)
            self.LPy = self.a_ws * self.np_per_side[1] * (4.0 * pi / 3.0) ** (1.0 / 3.0)
            self.LPz = self.a_ws * self.np_per_side[2] * (4.0 * pi / 3.0) ** (1.0 / 3.0)
        else:
            self.LPx = self.a_ws * (4.0 * pi * self.total_num_ptcls / 3.0) ** (1.0 / 3.0)
            self.LPy = self.a_ws * (4.0 * pi * self.total_num_ptcls / 3.0) ** (1.0 / 3.0)
            self.LPz = self.a_ws * (4.0 * pi * self.total_num_ptcls / 3.0) ** (1.0 / 3.0)

        if self.Lx == 0:
            self.Lx = self.LPx
        if self.Ly == 0:
            self.Ly = self.LPy
        if self.Lz == 0:
            self.Lz = self.LPz

        self.pbox_lengths = array([self.LPx, self.LPy, self.LPz])  # initial particle box length vector
        self.box_lengths = array([self.Lx, self.Ly, self.Lz])  # box length vector

        # Dev Note: The following are useful for future geometries
        self.e1 = array([self.Lx, 0.0, 0.0])
        self.e2 = array([0.0, self.Ly, 0.0])
        self.e3 = array([0.0, 0.0, self.Lz])

        self.ep1 = array([self.LPx, 0.0, 0.0])
        self.ep2 = array([0.0, self.LPy, 0.0])
        self.ep3 = array([0.0, 0.0, self.LPz])

        self.box_volume = abs(cross(self.e1, self.e2).dot(self.e3))
        self.pbox_volume = abs(cross(self.ep1, self.ep2).dot(self.ep3))

        self.dimensions = count_nonzero(self.box_lengths)  # no. of dimensions
        # Transform the list of species names into a array
        self.species_names = array(self.species_names)
        # Redundancy!!!
        self.T_desired = self.total_ion_temperature

    def calc_coupling_constant(self, species):
        """
        Calculate the coupling constant of each species and the total coupling constant. For more information see
        the theory pages.

        Parameters
        ----------
        species: list
            List of ``sarkas.core.Species`` objects.

        """
        z_avg = (self.species_charges.transpose()) @ self.species_concentrations

        for i, sp in enumerate(species):
            const = self.fourpie0 * self.kB
            sp.calc_coupling(self.a_ws, z_avg, const)
            self.species_couplings[i] = sp.coupling
            self.coupling_constant += sp.concentration * sp.coupling

    def pretty_print(self, electron_properties: bool = True):
        """
        Print simulation parameters in a user-friendly way.

        Parameters
        ----------
        electron_properties: bool
            Flag for printing electron properties. Default = True

        """
        print("\nSIMULATION AND INITIAL PARTICLE BOX:")
        print(f"Units: {self.units}")
        print(f"Wigner-Seitz radius = {self.a_ws:.6e} ", end="")
        print("[cm]" if self.units == "cgs" else "[m]")
        print(f"No. of non-zero box dimensions = {int(self.dimensions)}")
        box_a = self.box_lengths / self.a_ws
        print(f"Box side along x axis = {box_a[0]:.6e} a_ws = {self.box_lengths[0]:.6e} ", end="")
        print("[cm]" if self.units == "cgs" else "[m]")
        print(f"Box side along y axis = {box_a[1]:.6e} a_ws = {self.box_lengths[1]:.6e} ", end="")
        print("[cm]" if self.units == "cgs" else "[m]")
        print(f"Box side along z axis = {box_a[2]:.6e} a_ws = {self.box_lengths[2]:.6e} ", end="")
        print("[cm]" if self.units == "cgs" else "[m]")
        print(f"Box Volume = {self.box_volume:.6e} ", end="")
        print("[cm^3]" if self.units == "cgs" else "[m^3]")

        pbox_a = self.pbox_lengths / self.a_ws
        print(f"Initial particle box side along x axis = {pbox_a[0]:.6e} a_ws = {self.pbox_lengths[0]:.6e} ", end="")
        print("[cm]" if self.units == "cgs" else "[m]")
        print(f"Initial particle box side along y axis = {pbox_a[1]:.6e} a_ws = {self.pbox_lengths[1]:.6e} ", end="")
        print("[cm]" if self.units == "cgs" else "[m]")
        print(f"Initial particle box side along z axis = {pbox_a[2]:.6e} a_ws = {self.pbox_lengths[2]:.6e} ", end="")
        print("[cm]" if self.units == "cgs" else "[m]")
        print(f"Initial particle box Volume = {self.pbox_volume:.6e} ", end="")
        print("[cm^3]" if self.units == "cgs" else "[m^3]")

        print("Boundary conditions: {}".format(self.boundary_conditions))

        if electron_properties:
            print("\nELECTRON PROPERTIES:")
            print(f"Number density: n_e = {self.ne:.6e} ", end="")
            print("[N/cc]" if self.units == "cgs" else "[N/m^3]")

            print(f"Wigner-Seitz radius: a_e = {self.ae_ws:.6e} ", end="")
            print("[cm]" if self.units == "cgs" else "[m]")

            print(
                f"Temperature: T_e = {self.electron_temperature:.6e} [K] = {self.electron_temperature / self.eV2K:.6e} [eV]"
            )

            print(f"de Broglie wavelength: lambda_deB = {self.lambda_deB:.6e} ", end="")
            print("[cm]" if self.units == "cgs" else "[m]")

            print(f"Thomas-Fermi length: lambda_TF = {self.lambda_TF:.6e} ", end="")
            print("[cm]" if self.units == "cgs" else "[m]")

            print(f"Fermi wave number: k_F = {self.kF:.6e} ", end="")
            print("[1/cm]" if self.units == "cgs" else "[1/m]")

            print(f"Fermi Energy: E_F = {self.fermi_energy / self.kB / self.eV2K:.6e} [eV]")

            print(f"Relativistic parameter: x_F = {self.relativistic_parameter:.6e}".format(), end="")
            kf_xf = self.me * self.c0**2 * (sqrt(1.0 + self.relativistic_parameter**2) - 1.0)
            print(f" --> E_F = {(kf_xf / self.kB / self.eV2K):.6e} [eV]")

            print(f"Degeneracy parameter: Theta = {self.electron_degeneracy_parameter:.6e} ")
            print(f"Coupling: r_s = {self.rs:.6f},  Gamma_e = {self.electron_coupling:.6f}")
            print(f"Warm Dense Matter Parameter: W = {self.wdm_parameter:.4e}")
            mu_EF = self.eta_e * self.kB * self.electron_temperature / self.fermi_energy
            print(f"Chemical potential: mu = {self.eta_e:.4e} k_B T_e = {mu_EF:.4e} E_F")

            if self.magnetized:
                print(f"Electron cyclotron frequency: w_c = {self.electron_cyclotron_frequency:.6e}")
                print(f"Lowest Landau energy level: h w_c/2 = {0.5 * self.electron_magnetic_energy:.6e}")
                b_ef = self.electron_magnetic_energy / self.fermi_energy
                b_t = self.electron_magnetic_energy / (self.kB * self.electron_temperature)
                print(
                    f"Electron magnetic energy gap: h w_c = {self.electron_magnetic_energy:.6e} "
                    f"= {b_ef:.4e} E_F = {b_t:.4e} k_B T_e"
                )

        if self.magnetized:
            print("\nMAGNETIC FIELD:")
            print("Magnetic Field = [{:.4e}, {:.4e}, {:.4e}] ".format(*self.magnetic_field), end="")
            print("[Tesla]" if self.units == "mks" else "[Gauss]")
            print(f"Magnetic Field Magnitude = {norm(self.magnetic_field):.4e} ", end="")
            print("[Tesla]" if self.units == "mks" else "[Gauss]")
            print(
                "Magnetic Field Unit Vector = [{:.4e}, {:.4e}, {:.4e}]".format(
                    *self.magnetic_field / norm(self.magnetic_field)
                )
            )


class Particles:
    """
    Class handling particles' properties.

    Attributes
    ----------
    kB : float
        Boltzmann constant.

    fourpie0: float
        Electrostatic constant :math:`4\\pi \\epsilon_0`.

    pos : numpy.ndarray
        Particles' positions.

    vel : numpy.ndarray
        Particles' velocities.

    acc : numpy.ndarray
        Particles' accelerations.

    box_lengths : numpy.ndarray
        Box sides' lengths.

    pbox_lengths : numpy.ndarray
        Initial particle box sides' lengths.

    masses : numpy.ndarray
        Mass of each particle. Shape = (attr:`sarkas.core.Parameters.total_num_ptcls`).

    charges : numpy.ndarray
        Charge of each particle. Shape = (attr:`sarkas.core.Parameters.total_num_ptcls`).

    id : numpy.ndarray,
        Species identifier. Shape = (attr:`sarkas.core.Parameters.total_num_ptcls`).

    names : numpy.ndarray
        Species' names. (attr:`sarkas.core.Parameters.total_num_ptcls`).

    rdf_nbins : int
        Number of bins for radial pair distribution.

    no_grs : int
        Number of independent :math:`g_{ij}(r)`.

    rdf_hist : numpy.ndarray
        Histogram array for the radial pair distribution function.

    prod_dump_dir : str
        Directory name where to store production phase's simulation's checkpoints. Default = 'dumps'.

    eq_dump_dir : str
        Directory name where to store equilibration phase's simulation's checkpoints. Default = 'dumps'.

    total_num_ptcls : int
        Total number of simulation's particles.

    num_species : int
        Number of species.

    species_num : numpy.ndarray
        Number of particles of each species. Shape = (attr:`sarkas.core.Particles.num_species`).

    dimensions : int
        Number of non-zero dimensions. Default = 3.

    potential_energy : float
        Instantaneous value of the potential energy.

    rnd_gen : numpy.random.Generator
        Random number generator.

    """

    def __init__(self):
        self.mag_dump_dir = None
        self.rdf_nbins = None
        self.potential_energy = 0.0
        self.kB = None
        self.fourpie0 = None
        self.prod_dump_dir = None
        self.eq_dump_dir = None
        self.box_lengths = None
        self.pbox_lengths = None
        self.total_num_ptcls = None
        self.num_species = 1
        self.species_num = None
        self.dimensions = None
        self.rnd_gen = None

        self.pos = None
        self.vel = None
        self.acc = None

        self.virial = None
        self.pbc_cntr = None

        self.names = None
        self.id = None

        self.species_init_vel = None
        self.species_thermal_velocity = None

        self.masses = None
        self.charges = None
        self.cyclotron_frequencies = None

        self.no_grs = None
        self.rdf_hist = None

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Particles( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def __copy__(self):
        """Make a shallow copy of the object using copy."""
        return py_copy(self)

    def setup(self, params, species):
        """
        Initialize class' attributes

        Parameters
        ----------
        params: :class:`sarkas.core.Parameters`
            Simulation's parameters.

        species : list
            List of :meth:`sarkas.core.Species` objects.

        """

        self.kB = params.kB
        self.fourpie0 = params.fourpie0
        self.prod_dump_dir = params.prod_dump_dir
        self.eq_dump_dir = params.eq_dump_dir
        self.box_lengths = params.box_lengths.copy()
        self.pbox_lengths = params.pbox_lengths.copy()
        self.total_num_ptcls = params.total_num_ptcls
        self.num_species = params.num_species
        self.species_num = params.species_num.copy()
        self.dimensions = params.dimensions

        if hasattr(params, "rand_seed"):
            self.rnd_gen = Generator(PCG64(params.rand_seed))
        # else:
        #     self.rnd_gen = Generator(PCG64(123456789))

        self.pos = zeros((self.total_num_ptcls, params.dimensions))
        self.vel = zeros((self.total_num_ptcls, params.dimensions))
        self.acc = zeros((self.total_num_ptcls, params.dimensions))

        self.pbc_cntr = zeros((self.total_num_ptcls, params.dimensions))
        self.virial = zeros((params.dimensions, params.dimensions, self.total_num_ptcls))

        self.names = empty(self.total_num_ptcls, dtype=params.species_names.dtype)
        self.id = zeros(self.total_num_ptcls, dtype=int)

        self.species_init_vel = zeros((params.num_species, 3))
        self.species_thermal_velocity = zeros((params.num_species, 3))

        self.masses = zeros(self.total_num_ptcls)  # mass of each particle
        self.charges = zeros(self.total_num_ptcls)  # charge of each particle
        self.cyclotron_frequencies = zeros(self.total_num_ptcls)
        # No. of independent rdf
        self.no_grs = int(self.num_species * (self.num_species + 1) / 2)
        if hasattr(params, "rdf_nbins"):
            self.rdf_nbins = params.rdf_nbins
        else:
            # nbins = 5% of the number of particles.
            self.rdf_nbins = int(0.05 * params.total_num_ptcls)
            params.rdf_nbins = self.rdf_nbins

        self.rdf_hist = zeros((self.rdf_nbins, self.num_species, self.num_species))

        self.update_attributes(species)

        self.load(params)

    def load(self, params):
        """
        Initialize particles' positions and velocities.
        Positions are initialized based on the load method while velocities are chosen
        from a Maxwell-Boltzmann distribution.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            Simulation's parameters.

        """
        # Particles Position Initialization
        if params.load_method in [
            "equilibration_restart",
            "eq_restart",
            "magnetization_restart",
            "mag_restart",
            "production_restart",
            "prod_restart",
        ]:
            # checks
            if params.restart_step is None:
                raise AttributeError("Restart step not defined." "Please define Parameters.restart_step.")

            if type(params.restart_step) is not int:
                params.restart_step = int(params.restart_step)

            if params.load_method[:2] == "eq":
                self.load_from_restart("equilibration", params.restart_step)
            elif params.load_method[:2] == "pr":
                self.load_from_restart("production", params.restart_step)
            elif params.load_method[:2] == "ma":
                self.load_from_restart("magnetization", params.restart_step)

        elif params.load_method == "file":
            # check
            if not hasattr(params, "particles_input_file"):
                raise AttributeError("Input file not defined." "Please define Parameters.particles_input_file.")
            self.load_from_file(params.particles_input_file)

        # position distribution.
        elif params.load_method == "lattice":
            self.lattice(params.load_perturb)

        elif params.load_method == "random_reject":
            # check
            if not hasattr(params, "load_rejection_radius"):
                raise AttributeError("Rejection radius not defined. " "Please define Parameters.load_rejection_radius.")
            self.random_reject(params.load_rejection_radius)

        elif params.load_method == "halton_reject":
            # check
            if not hasattr(params, "load_rejection_radius"):
                raise AttributeError("Rejection radius not defined. " "Please define Parameters.load_rejection_radius.")
            self.halton_reject(params.load_halton_bases, params.load_rejection_radius)

        elif params.load_method in ["uniform", "random_no_reject"]:
            self.pos = self.uniform_no_reject(
                0.5 * params.box_lengths - 0.5 * params.pbox_lengths, 0.5 * params.box_lengths + 0.5 * params.pbox_lengths
            )

        else:
            raise AttributeError("Incorrect particle placement scheme specified.")

    def gaussian(self, mean, sigma, num_ptcls):
        """
        Initialize particles' velocities according to a normalized Maxwell-Boltzmann (Normal) distribution.
        It calls ``numpy.random.Generator.normal``

        Parameters
        ----------
        num_ptcls : int
            Number of particles to initialize.

        mean : float
            Center of the normal distribution.

        sigma : float
            Scale of the normal distribution.

        Returns
        -------
         : numpy.ndarray
            Particles property distributed according to a Normal probability density function.

        """
        return self.rnd_gen.normal(mean, sigma, (num_ptcls, 3))

    def random_unit_vectors(self, num_ptcls, dimensions):
        """
        Initialize random unit vectors for particles' velocities (e.g. for monochromatic energies but random velocities)
        It calls ``numpy.random.Generator.normal``

        Parameters
        ----------
        num_ptcls : int
            Number of particles to initialize.

        dimensions : int
            Number of non-zero dimensions.

        Returns
        -------
        uvec : numpy.ndarray
            Random unit vectors of specified dimensions for all particles

        """

        uvec = self.rnd_gen.normal(size=(num_ptcls, dimensions))
        # Broadcasting
        uvec /= norm(uvec).reshape(num_ptcls, 1)

        return uvec

    def update_attributes(self, species):
        """
        Assign particles attributes.

        Parameters
        ----------
        species : list
            List of ``sarkas.core.Species`` objects.

        """
        species_end = 0
        for ic, sp in enumerate(species):
            species_start = species_end
            species_end += sp.num

            self.names[species_start:species_end] = sp.name
            self.masses[species_start:species_end] = sp.mass

            if hasattr(sp, "charge"):
                self.charges[species_start:species_end] = sp.charge
            else:
                self.charges[species_start:species_end] = 1.0

            if hasattr(sp, "cyclotron_frequency"):
                self.cyclotron_frequencies[species_start:species_end] = sp.cyclotron_frequency

            self.id[species_start:species_end] = ic

            if hasattr(sp, "init_vel"):
                self.species_init_vel[ic, :] = sp.init_vel

            if sp.initial_velocity_distribution == "boltzmann":
                if isinstance(sp.temperature, (int, float)):
                    sp_temperature = array([1.0 for _ in range(self.dimensions)]) * sp.temperature

                self.species_thermal_velocity[ic] = sqrt(self.kB * sp_temperature / sp.mass)
                self.vel[species_start:species_end, :] = self.gaussian(
                    sp.initial_velocity, self.species_thermal_velocity[ic], sp.num
                )

            elif sp.initial_velocity_distribution == "monochromatic":
                vrms = sqrt(self.dimensions * self.kB * sp.temperature / sp.mass)
                self.vel[species_start:species_end, :] = vrms * self.random_unit_vectors(sp.num, self.dimensions)

    def load_from_restart(self, phase, it):
        """
        Load particles' data from a checkpoint of a previous run

        Parameters
        ----------
        it : int
            Timestep.

        phase: str
            Restart phase.

        """
        if phase == "equilibration":
            file_name = join(self.eq_dump_dir, "checkpoint_" + str(it) + ".npz")
            data = np_load(file_name, allow_pickle=True)
            self.id = data["id"]
            self.names = data["names"]
            self.pos = data["pos"]
            self.vel = data["vel"]
            self.acc = data["acc"]

        elif phase == "production":
            file_name = join(self.prod_dump_dir, "checkpoint_" + str(it) + ".npz")
            data = np_load(file_name, allow_pickle=True)
            self.id = data["id"]
            self.names = data["names"]
            self.pos = data["pos"]
            self.vel = data["vel"]
            self.acc = data["acc"]
            self.pbc_cntr = data["cntr"]
            self.rdf_hist = data["rdf_hist"]

        elif phase == "magnetization":
            file_name = join(self.mag_dump_dir, "checkpoint_" + str(it) + ".npz")
            data = np_load(file_name, allow_pickle=True)
            self.id = data["id"]
            self.names = data["names"]
            self.pos = data["pos"]
            self.vel = data["vel"]
            self.acc = data["acc"]
            self.pbc_cntr = data["cntr"]
            self.rdf_hist = data["rdf_hist"]

    def load_from_file(self, f_name):
        """
        Load particles' data from a specific file.

        Parameters
        ----------
        f_name : str
            Filename
        """
        pv_data = loadtxt(f_name)
        if not (pv_data.shape[0] == self.total_num_ptcls):
            msg = (
                f"Number of particles is not same between input file and initial p & v data file. \n "
                f"Input file: N = {self.total_num_ptcls}, load data: N = {pv_data.shape[0]}"
            )
            raise ParticlesError(msg)

        self.pos[:, 0] = pv_data[:, 0]
        self.pos[:, 1] = pv_data[:, 1]
        self.pos[:, 2] = pv_data[:, 2]

        self.vel[:, 0] = pv_data[:, 3]
        self.vel[:, 1] = pv_data[:, 4]
        self.vel[:, 2] = pv_data[:, 5]

    def uniform_no_reject(self, mins, maxs):
        """
        Randomly distribute particles along each direction.

        Parameters
        ----------
        mins : float
            Minimum value of the range of a uniform distribution.

        maxs : float
            Maximum value of the range of a uniform distribution.

        Returns
        -------
         : numpy.ndarray
            Particles' property, e.g. pos, vel. Shape = (``total_num_ptcls``, 3).

        """

        return self.rnd_gen.uniform(mins, maxs, (self.total_num_ptcls, 3))

    def lattice(self, perturb):
        """
        Place particles in a simple cubic lattice with a slight perturbation ranging
        from 0 to 0.5 times the lattice spacing.

        Parameters
        ----------
        perturb : float
            Value of perturbation, p, such that 0 <= p <= 1.

        """

        # Check if perturbation is below maximum allowed. If not, default to maximum perturbation.
        if perturb > 1:
            warn("\nWARNING: Random perturbation must not exceed 1. Setting perturb = 1.", category=ParticlesWarning)

        print(f"Initializing particles with maximum random perturbation of {perturb * 0.5} times the lattice spacing.")

        # Determining number of particles per side of simple cubic lattice
        part_per_side = self.total_num_ptcls ** (1.0 / 3.0)  # Number of particles per side of cubic lattice

        # Check if total number of particles is a perfect cube, if not, place more than the requested amount
        if round(part_per_side) ** 3 != self.total_num_ptcls:
            part_per_side = ceil(self.total_num_ptcls ** (1.0 / 3.0))
            warn(
                f"\nWARNING: Total number of particles requested is not a perfect cube. "
                f"Initializing with {int(part_per_side ** 3)} particles.",
                category=ParticlesWarning,
            )

        dx_lattice = self.pbox_lengths[0] / (self.total_num_ptcls ** (1.0 / 3.0))  # Lattice spacing
        dy_lattice = self.pbox_lengths[1] / (self.total_num_ptcls ** (1.0 / 3.0))  # Lattice spacing
        dz_lattice = self.pbox_lengths[2] / (self.total_num_ptcls ** (1.0 / 3.0))  # Lattice spacing

        # Create x, y, and z position arrays
        x = arange(0, self.pbox_lengths[0], dx_lattice) + 0.5 * dx_lattice
        y = arange(0, self.pbox_lengths[1], dy_lattice) + 0.5 * dy_lattice
        z = arange(0, self.pbox_lengths[2], dz_lattice) + 0.5 * dz_lattice

        # Create a lattice with appropriate x, y, and z values based on arange
        X, Y, Z = meshgrid(x, y, z)

        # Perturb lattice
        X += self.rnd_gen.uniform(-0.5, 0.5, X.shape) * perturb * dx_lattice
        Y += self.rnd_gen.uniform(-0.5, 0.5, Y.shape) * perturb * dy_lattice
        Z += self.rnd_gen.uniform(-0.5, 0.5, Z.shape) * perturb * dz_lattice

        # Flatten the meshgrid values for plotting and computation
        self.pos[:, 0] = X.ravel() + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
        self.pos[:, 1] = Y.ravel() + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
        self.pos[:, 2] = Z.ravel() + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

    def random_reject(self, r_reject):
        """
        Place particles by sampling a uniform distribution from 0 to LP (the initial particle box length)
        and uses a rejection radius to avoid placing particles to close to each other.

        Parameters
        ----------
        r_reject : float
            Value of rejection radius.
        """

        # Initialize Arrays
        x = zeros(self.total_num_ptcls)
        y = zeros(self.total_num_ptcls)
        z = zeros(self.total_num_ptcls)

        # Set first x, y, and z positions
        x_new = self.rnd_gen.uniform(0, self.pbox_lengths[0])
        y_new = self.rnd_gen.uniform(0, self.pbox_lengths[1])
        z_new = self.rnd_gen.uniform(0, self.pbox_lengths[2])

        # Append to arrays
        x[0] = x_new
        y[0] = y_new
        z[0] = z_new

        # Particle counter
        i = 1

        cntr_reject = 0
        cntr_total = 0
        # Loop to place particles
        while i < self.total_num_ptcls:

            # Set x, y, and z positions
            x_new = self.rnd_gen.uniform(0.0, self.pbox_lengths[0])
            y_new = self.rnd_gen.uniform(0.0, self.pbox_lengths[1])
            z_new = self.rnd_gen.uniform(0.0, self.pbox_lengths[2])

            # Check if particle was place too close relative to all other current particles
            for j in range(len(x)):

                # Flag for if particle is outside of cutoff radius (True -> not inside rejection radius)
                flag = 1

                # Compute distance b/t particles for initial placement
                x_diff = x_new - x[j]
                y_diff = y_new - y[j]
                z_diff = z_new - z[j]

                # periodic condition applied for minimum image
                if x_diff < -self.pbox_lengths[0] / 2:
                    x_diff += self.pbox_lengths[0]
                if x_diff > self.pbox_lengths[0] / 2:
                    x_diff -= self.pbox_lengths[0]

                if y_diff < -self.pbox_lengths[1] / 2:
                    y_diff += self.pbox_lengths[1]
                if y_diff > self.pbox_lengths[1] / 2:
                    y_diff -= self.pbox_lengths[1]

                if z_diff < -self.pbox_lengths[2] / 2:
                    z_diff += self.pbox_lengths[2]
                if z_diff > self.pbox_lengths[2] / 2:
                    z_diff -= self.pbox_lengths[2]

                # Compute distance
                r = sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                # Check if new particle is below rejection radius. If not, break out and try again
                if r <= r_reject:
                    flag = 0  # new position not added (False -> no longer outside reject r)
                    cntr_reject += 1
                    cntr_total += 1
                    break

            # If flag true add new position
            if flag == 1:
                x[i] = x_new
                y[i] = y_new
                z[i] = z_new

                # Increment particle number
                i += 1
                cntr_total += 1

        self.pos[:, 0] = x + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
        self.pos[:, 1] = y + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
        self.pos[:, 2] = z + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

    def halton_reject(self, bases, r_reject):
        """
        Place particles according to a Halton sequence from 0 to LP (the initial particle box length)
        and uses a rejection radius to avoid placing particles to close to each other.

        Parameters
        ----------
        bases : numpy.ndarray
            Array of 3 ints each of which is a base for the Halton sequence.
            Defualt: bases = array([2,3,5])

        r_reject : float
            Value of rejection radius.

        """

        # Get bases
        b1, b2, b3 = bases

        # Allocate space and store first value from Halton
        x = zeros(self.total_num_ptcls)
        y = zeros(self.total_num_ptcls)
        z = zeros(self.total_num_ptcls)

        # Initialize particle counter and Halton counter
        i = 1
        k = 1

        # Loop over all particles
        while i < self.total_num_ptcls:

            # Increment particle counter
            n = k
            m = k
            p = k

            # Determine x coordinate
            f1 = 1
            r1 = 0
            while n > 0:
                f1 /= b1
                r1 += f1 * (n % int(b1))
                n = floor(n / b1)
            x_new = self.pbox_lengths[0] * r1  # new x value

            # Determine y coordinate
            f2 = 1
            r2 = 0
            while m > 0:
                f2 /= b2
                r2 += f2 * (m % int(b2))
                m = floor(m / b2)
            y_new = self.pbox_lengths[1] * r2  # new y value

            # Determine z coordinate
            f3 = 1
            r3 = 0
            while p > 0:
                f3 /= b3
                r3 += f3 * (p % int(b3))
                p = floor(p / b3)
            z_new = self.pbox_lengths[2] * r3  # new z value

            # Check if particle was place too close relative to all other current particles
            for j in range(len(x)):

                # Flag for if particle is outside of cutoff radius (1 -> not inside rejection radius)
                flag = 1

                # Compute distance b/t particles for initial placement
                x_diff = x_new - x[j]
                y_diff = y_new - y[j]
                z_diff = z_new - z[j]

                # Periodic condition applied for minimum image
                if x_diff < -self.pbox_lengths[0] / 2:
                    x_diff = x_diff + self.pbox_lengths[0]
                if x_diff > self.pbox_lengths[0] / 2:
                    x_diff = x_diff - self.pbox_lengths[0]

                if y_diff < -self.pbox_lengths[1] / 2:
                    y_diff = y_diff + self.pbox_lengths[1]
                if y_diff > self.pbox_lengths[1] / 2:
                    y_diff = y_diff - self.pbox_lengths[1]

                if z_diff < -self.pbox_lengths[2] / 2:
                    z_diff = z_diff + self.pbox_lengths[2]
                if z_diff > self.pbox_lengths[2] / 2:
                    z_diff = z_diff - self.pbox_lengths[2]

                # Compute distance
                r = sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                # Check if new particle is below rejection radius. If not, break out and try again
                if r <= r_reject:
                    k += 1  # Increment Halton counter
                    flag = 0  # New position not added (0 -> no longer outside reject r)
                    break

            # If flag true add new position
            if flag == 1:
                # Add new positions to arrays
                x[i] = x_new
                y[i] = y_new
                z[i] = z_new

                k += 1  # Increment Halton counter
                i += 1  # Increment particle number

        self.pos[:, 0] = x + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
        self.pos[:, 1] = y + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
        self.pos[:, 2] = z + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

    def kinetic_temperature(self):
        """
        Calculate the kinetic energy and temperature of each species.

        Returns
        -------
        K : numpy.ndarray
            Kinetic energy of each species. Shape=(``num_species``).

        T : numpy.ndarray
            Temperature of each species. Shape=(``num_species``).

        """
        K = zeros(self.num_species)
        T = zeros(self.num_species)
        const = 2.0 / (self.kB * self.species_num * self.dimensions)
        kinetic = 0.5 * self.masses * (self.vel * self.vel).transpose()

        species_start = 0
        species_end = 0
        for i, num in enumerate(self.species_num):
            species_end += num
            K[i] = kinetic[:, species_start:species_end].sum(axis=-1)
            T[i] = const[i] * K[i]
            species_start = species_end

        return K, T

    def potential_energies(self):
        """
        Calculate the potential energies of each species.

        Returns
        -------
        P : numpy.ndarray
            Potential energy of each species. Shape=(``num_species``).

        """
        P = zeros(self.num_species)

        species_start = 0
        species_end = 0
        for i, num in enumerate(self.species_num):
            species_end += num

            # TODO: Consider writing a numba function speedup in distance calculation
            species_charges = self.charges[species_start:species_end]
            uti = triu_indices(species_charges.size, k=1)
            species_charge2 = species_charges[uti[0]] * species_charges[uti[1]]
            species_distances = pdist(self.pos[species_start:species_end, :])
            potential = species_charge2 / self.fourpie0 / species_distances
            P[i] = potential.sum()

            species_start = species_end

        return P

    def remove_drift(self):
        """
        Enforce conservation of total linear momentum. Updates particles velocities
        """
        species_start = 0
        species_end = 0
        momentum = self.masses * self.vel.transpose()
        for ic, nums in enumerate(self.species_num):
            species_end += nums
            P = momentum[:, species_start:species_end].sum(axis=1)
            self.vel[species_start:species_end, :] -= P / (nums * self.masses[species_end - 1])
            species_start = species_end


class Species:
    """
    Class used to store all the information of a single species.

    Attributes
    ----------
    name : str
        Species' name.

    number_density : float
        Species number density in appropriate units.

    num : int
        Number of particles of Species.

    mass : float
        Species' mass.

    charge : float
        Species' charge.

    Z : float
        Species charge number or ionization degree.

    ai : float
        Species Wigner - Seitz radius calculate from electron density.
        Used in the calculation of the effective coupling constant.

    ai_dens : float
        Species Wigner - Seitz radius calculate from the species density.

    coupling : float
        Species coupling constant

    plasma_frequency : float
        Species' plasma frequency.

    debye_length : float
        Species' Debye Length.

    cyclotron_frequency : float
        Species' cyclotron frequency.

    initial_velocity : numpy.ndarray
        Initial velocity in x,y,z directions.

    temperature : float
        Initial temperature of the species.

    temperature_eV : float
        Initial temperature of the species in eV.

    initial_velocity_distribution : str
        Type of distribution. Default = 'boltzmann'.

    initial_spatial_distribution : str
        Type of distribution. Default = 'uniform'.

    atomic_weight : float
        (Optional) Species mass in atomic units.

    concentration : float
        Species' concentration.

    mass_density : float
        (Optional) Species' mass density.


    """

    def __init__(self, input_dict: dict = None):
        """
        Parameters
        ----------
        input_dict: dict, optional
            Dictionary with species information.

        """
        self.ai = None
        self.ai_dens = None
        self.atomic_weight = None
        self.charge = None
        self.Z = None
        self.coupling = None
        self.concentration = None
        self.debye_length = None
        self.initial_spatial_distribution = "random_no_reject"
        self.initial_velocity_distribution = "boltzmann"
        self.initial_velocity = zeros(3)
        self.mass = None
        self.mass_density = None
        self.name = None
        self.num = None
        self.number_density = None
        self.cyclotron_frequency = None
        self.plasma_frequency = None
        self.temperature = None
        self.temperature_eV = None

        if input_dict:
            self.from_dict(input_dict)

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Species( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def from_dict(self, input_dict: dict):
        """
        Update attributes from input dictionary using the ``__dict__.update()`` method.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """
        self.__dict__.update(input_dict)
        if not isinstance(self.initial_velocity, ndarray):
            self.initial_velocity = array(self.initial_velocity)

    def calc_plasma_frequency(self, constant: float):
        """
        Calculate the plasma frequency.

        Parameters
        ----------
        constant : float
            Charged systems: Electrostatic constant  :math: `4\\pi \\epsilon_0` [mks]
            Neutral systems: :math: `1/n\\sigma^2`

        """
        self.plasma_frequency = sqrt(4.0 * pi * self.charge**2 * self.number_density / (self.mass * constant))

    def calc_debye_length(self, kB: float, constant: float):
        """
        Calculate the Debye Length.

        Parameters
        ----------
        kB : float
            Boltzmann constant.

        constant : float
            Charged systems: Electrostatic constant  :math: `4 \\pi \\epsilon_0` [mks]
            Neutral systems: :math: `1/n\\sigma^2`

        """
        self.debye_length = sqrt((self.temperature * kB * constant) / (4.0 * pi * self.charge**2 * self.number_density))

    def calc_cyclotron_frequency(self, magnetic_field_strength: float):
        """
        Calculate the cyclotron frequency. See `Wikipedia link <https://en.wikipedia.org/wiki/Lorentz_force>`_.

        Parameters
        ----------
        magnetic_field_strength : float
            Magnetic field strength.

        """
        self.cyclotron_frequency = self.charge * magnetic_field_strength / self.mass

    def calc_coupling(self, a_ws: float, z_avg: float, const: float):
        """
        Calculate the coupling constant between particles.

        Parameters
        ----------
        a_ws : float
            Total Wigner-Seitz radius.

        z_avg : float
            Average charge of the system.

        const : float
            Electrostatic * Thermal constants.

        """
        self.ai_dens = (3.0 / (4.0 * pi * self.number_density)) ** (1.0 / 3.0)
        self.ai = (self.charge / z_avg) ** (1.0 / 3.0) * a_ws if z_avg > 0 else self.ai_dens
        self.coupling = self.charge**2 / (self.ai * const * self.temperature)

    def pretty_print(self, potential_type: str = None, units: str = "mks"):
        """Print Species information in a user-friendly way.

        Parameters
        ----------
        potential_type: str
            Interaction potential. If 'LJ' it will print out the epsilon and sigma attributes.

        units: str
            Unit system used in the simulation. Default = 'mks'.

        """

        print(f"\tName: {self.name}")
        print(f"\tNo. of particles = {self.num} ")
        print(f"\tNumber density = {self.number_density:.6e} ", end="")
        print("[N/cc]" if units == "cgs" else "[N/m^3]")
        print(f"\tAtomic weight = {self.atomic_weight:.4f} [a.u.]")
        print(f"\tMass = {self.mass:.6e} ", end="")
        print("[g]" if units == "cgs" else "[kg]")
        print(f"\tMass density = {self.mass_density:.6e} ", end="")
        print("[g/cc]" if units == "cgs" else "[kg/m^3]")
        print(f"\tCharge number/ionization degree = {self.Z:.4f} ")
        print(f"\tCharge = {self.charge:.6e} ", end="")
        print("[esu]" if units == "cgs" else "[C]")
        print(f"\tTemperature = {self.temperature:.6e} [K] = {self.temperature_eV:.6e} [eV]")
        if potential_type == "lj":
            print(f"\tEpsilon = {self.epsilon:.6e} ", end="")
            print("[erg]" if units == "cgs" else "[J]")
            print(f"\tSigma = {self.sigma:.6e} ", end="")
            print("[cm]" if units == "cgs" else "[m]")

        print(f"\tDebye Length = {self.debye_length:.6e} ", end="")
        print("[cm]" if units == "cgs" else "[m]")
        print(f"\tPlasma Frequency = {self.plasma_frequency:.6e} [rad/s]")
        if self.cyclotron_frequency:
            print(f"\tCyclotron Frequency = {self.cyclotron_frequency:.6e} [rad/s]")
            print(f"\tbeta_c = {self.cyclotron_frequency / self.plasma_frequency:.4e}")
