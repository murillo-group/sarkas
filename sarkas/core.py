"""
Module containing the three basic classes: Parameters, Particles, Species.
"""

from copy import deepcopy
from numpy import array, cross, float64, int64, ndarray, pi, rint, sqrt, tanh, zeros
from scipy.constants import physical_constants
from scipy.linalg import norm

from .plasma import Species
from .utilities.exceptions import ParticlesError


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

    equilibration_steps : int
        Total number of equilibration timesteps.

    eq_dump_step : int
        Equilibration dump interval.

    magnetization_steps : int
        Total number of magnetization timesteps.

    mag_dump_step : int
        Magnetization dump interval.

    production_steps : int
        Total number of production timesteps.

    prod_dump_step : int
        Production dump interval.

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
        Total number density. Calculated from the sum of :attr:`Species.number_density`.

    total_num_ptcls : int
        Total number of particles. Calculated from the sum of :attr:`Species.num`.

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

    units_dict : dict
        Strings of the units used in the simulation.
    """

    def __init__(self, dic: dict = None):

        self.particles_input_file = None
        self.load_perturb = 0.0
        self.initial_lattice_config = "simple_cubic"
        self.load_rejection_radius = None
        self.load_halton_bases = None
        self.load_method = None
        self.potential_type = None
        self.units = None
        self.electron_magnetic_energy = None
        self.input_file = None
        self.units_dict = {}
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
        self.box_lengths = array([0.0, 0.0, 0.0])
        self.pbox_lengths = array([0.0, 0.0, 0.0])
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

        # Phases
        self.equilibration_phase = True
        self.electrostatic_equilibration = True
        self.magnetization_phase = False
        self.production_phase = True

        # Timing
        self.equilibration_steps = 0
        self.production_steps = 0
        self.magnetization_steps = 0
        self.eq_dump_step = 1
        self.prod_dump_step = 1
        self.mag_dump_step = 1

        # Control
        self.job_id = None
        self.job_dir = None
        self.log_file = None
        self.measure = False
        self.magnetized = False
        self.plot_style = None
        self.pre_run = False
        self.threading = False
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

    def __copy__(self):
        """Make a shallow copy of the object using copy by creating a new instance of the object and copying its __dict__."""
        # Create a new object
        _copy = type(self)(dic=self.__dict__)
        return _copy

    def __deepcopy__(self, memodict={}):
        """
        Make a deepcopy of the object.

        Parameters
        ----------
        memodict: dict
            Dictionary of id's to copies

        Returns
        -------
        _copy: :class:`sarkas.core.Parameters`
            A new Parameters class.
        """
        id_self = id(self)  # memorization avoids unnecessary recursion
        _copy = memodict.get(id_self)
        if _copy is None:
            _copy = type(self)()
            # Make a deepcopy of the mutable arrays using numpy copy function
            for k, v in self.__dict__.items():
                _copy.__dict__[k] = deepcopy(v, memodict)

        return _copy

    def calc_coupling_constant(self, species: list):
        """
        Calculate the coupling constant of each species and the total coupling constant. For more information see
        the theory pages.

        Parameters
        ----------
        species: list
            List of ``sarkas.plasma.Species`` objects.

        """
        z_avg = (self.species_charges.transpose()) @ self.species_concentrations

        for i, sp in enumerate(species):
            const = self.fourpie0 * self.kB
            sp.calc_coupling(self.a_ws, z_avg, const)
            self.species_couplings[i] = sp.coupling
            self.coupling_constant += sp.concentration * sp.coupling

    def calc_electron_properties(self, species: list):
        """Check whether the electrons are a dynamical species or not."""
        # Check for electrons as dynamical species
        if "e" not in self.species_names:
            electrons = {
                "name": "electron_background",
                "number_density": (
                    self.species_charges.transpose() @ self.species_concentrations * self.total_num_density / self.qe
                ),
            }
            if hasattr(self, "electron_temperature_eV"):
                electrons["temperature_eV"] = self.electron_temperature_eV
                electrons["temperature"] = self.eV2K * self.electron_temperature_eV
            elif hasattr(self, "electron_temperature"):
                electrons["temperature"] = self.electron_temperature
                electrons["temperature_eV"] = self.electron_temperature / self.eV2K
            else:
                electrons["temperature"] = self.total_ion_temperature
                electrons["temperature_eV"] = self.total_ion_temperature / self.eV2K
            electrons["mass"] = self.me
            electrons["Z"] = -1.0
            electrons["charge"] = electrons["Z"] * self.qe
            electrons["spin_degeneracy"] = 2.0
            electrons["num"] = (self.species_num.T @ self.species_charges / self.qe).astype(int64)
            e_species = Species(electrons)
            e_species.copy_params(self)
            e_species.calc_ws_radius()
            e_species.calc_plasma_frequency()
            e_species.calc_debye_length()
            e_species.calc_landau_length()
            if self.magnetized:
                b_mag = norm(self.magnetic_field)  # magnitude of B
                if self.units == "cgs":
                    b_mag /= self.c0

                e_species.calc_cyclotron_frequency(magnetic_field_strength=b_mag)
            # Electron should be the last species if not dynamical
            species.append(e_species)
        else:
            # Electron should be the first species if dynamical
            e_species = species[0]

        e_species.calc_debroglie_wavelength()
        e_species.calc_quantum_attributes(spin_statistics="fermi-dirac")
        # Electron WS radius
        e_species.a_ws = (3.0 / (4.0 * pi * e_species.number_density)) ** (1.0 / 3.0)
        # Brueckner parameters
        e_species.rs = e_species.a_ws / self.a0

        # Other electron parameters
        e_species.degeneracy_parameter = self.kB * e_species.temperature / e_species.Fermi_energy
        e_species.relativistic_parameter = self.hbar * e_species.Fermi_wavenumber / (self.me * self.c0)

        # Eq. 1 in Murillo Phys Rev E 81 036403 (2010)
        e_species.coupling = e_species.charge**2 / (
            self.fourpie0 * e_species.Fermi_energy * e_species.a_ws * sqrt(1.0 + e_species.degeneracy_parameter**2)
        )

        # Warm Dense Matter Parameter, Eq.3 in Murillo Phys Rev E 81 036403 (2010)
        e_species.wdm_parameter = 2.0 / (e_species.degeneracy_parameter + 1.0 / e_species.degeneracy_parameter)
        e_species.wdm_parameter *= 2.0 / (e_species.coupling + 1.0 / e_species.coupling)

        if self.magnetized:
            # Inverse temperature for convenience
            beta_e = 1.0 / (self.kB * e_species.temperature)

            e_species.magnetic_energy = self.hbar * e_species.cyclotron_frequency
            tan_arg = 0.5 * self.hbar * e_species.cyclotron_frequency * beta_e

            # Perpendicular correction
            e_species.horing_perp_correction = (e_species.plasma_frequency / e_species.cyclotron_frequency) ** 2
            e_species.horing_perp_correction *= 1.0 - tan_arg / tanh(tan_arg)
            e_species.horing_perp_correction += 1

            # Parallel correction
            e_species.horing_par_correction = 1 - (self.hbar * beta_e * e_species.plasma_frequency) ** 2 / 12.0

            # Quantum Anisotropy Parameter
            e_species.horing_delta = e_species.horing_perp_correction - 1
            e_species.horing_delta += (self.hbar * beta_e * e_species.cyclotron_frequency) ** 2 / 12.0
            e_species.horing_delta /= e_species.horing_par_correction

    def calc_parameters(self, species: list):
        """
        Assign the parsed parameters.

        Parameters
        ----------
        species : list
            List of :class:`sarkas.plasma.Species` .

        """

        self.set_species_attributes(species)
        self.create_species_arrays(species)

        if self.magnetized:
            self.magnetic_field = array(self.magnetic_field, dtype=float)
            self.calc_magnetic_parameters(species)

        self.sim_box_setup()

    def calc_magnetic_parameters(self, species: list):
        """
        Calculate cyclotron frequency in case of a magnetized simulation.

        Parameters
        ----------
        species: list,
            List of :class:`sarkas.plasma.Species`.

        """
        self.species_cyclotron_frequencies = zeros(self.num_species)
        for i, sp in enumerate(species):

            b_mag = norm(self.magnetic_field)
            if self.units == "cgs":
                b_mag /= self.c0

            sp.calc_cyclotron_frequency(magnetic_field_strength=b_mag)

            sp.beta_c = sp.cyclotron_frequency / sp.plasma_frequency
            self.species_cyclotron_frequencies[i] = sp.cyclotron_frequency

    def check_units(self):
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

        self.create_unit_dict()

    def create_unit_dict(self):
        """Make a dictionary whose values are strings of the units. It is used in pretty_print methods."""

        if self.units == "cgs":
            self.units_dict["number density"] = "[N/cc]" if self.dimensions == 3 else "[N/cm^2]"
            self.units_dict["weight"] = "[g]"
            self.units_dict["mass density"] = "[g/cc]" if self.dimensions == 3 else "[g/cm^2]"
            self.units_dict["charge"] = "[esu]"
            self.units_dict["energy"] = "[erg]"
            self.units_dict["length"] = "[cm]"
            self.units_dict["volume"] = "[cm^3]" if self.dimensions == 3 else "[cm^2]"
            self.units_dict["inverse length"] = "[1/cm]"
            self.units_dict["magnetic field strength"] = "[Gauss]"
        else:
            self.units_dict["density"] = "[N/m^3]" if self.dimensions == 3 else "[N/m^2]"
            self.units_dict["weight"] = "[kg]"
            self.units_dict["mass density"] = "[kg/cc]" if self.dimensions == 3 else "[kg/m^2]"
            self.units_dict["charge"] = "[C]"
            self.units_dict["energy"] = "[J]"
            self.units_dict["length"] = "[m]"
            self.units_dict["volume"] = "[m^3]" if self.dimensions == 3 else "[m^2]"
            self.units_dict["inverse length"] = "[1/m]"
            self.units_dict["magnetic field strength"] = "[Tesla]"

        self.units_dict["temperature"] = "[K]"
        self.units_dict["Hertz"] = "[1/s]"
        self.units_dict["frequency"] = "[rad/s]"
        self.units_dict["time"] = "[s]"
        self.units_dict["electron volt"] = "[eV]"

    def create_species_arrays(self, species: list):
        """
        Get species information into arrays for the postprocessing part.

        Parameters
        ----------
        species : list
            List of :class:`sarkas.plasma.Species` .

        """
        self.num_species = len(species)

        # Initialize the arrays containing species attributes. This is needed for postprocessing
        self.species_names = []
        self.species_num = zeros(self.num_species, dtype=int64)
        self.species_num_dens = zeros(self.num_species)
        self.species_concentrations = zeros(self.num_species)
        self.species_temperatures = zeros(self.num_species)
        self.species_temperatures_eV = zeros(self.num_species)
        self.species_masses = zeros(self.num_species)
        self.species_charges = zeros(self.num_species)
        self.species_plasma_frequencies = zeros(self.num_species)
        self.species_couplings = zeros(self.num_species)

        if self.potential_type == "lj":
            self.species_lj_sigmas = zeros(self.num_species)

        # Initialization of attributes
        self.total_num_ptcls = 0
        self.total_num_density = 0.0

        wp_tot_sq = 0.0
        lambda_D = 0.0

        for i, sp in enumerate(species):
            self.total_num_ptcls += sp.num
            self.total_num_density += sp.number_density
            self.species_concentrations[i] = sp.concentration
            self.species_names.append(sp.name)
            self.species_num[i] = sp.num
            self.species_masses[i] = sp.mass
            self.species_num_dens[i] = sp.number_density
            self.species_temperatures_eV[i] = sp.temperature_eV
            self.species_temperatures[i] = sp.temperature
            self.species_charges[i] = sp.charge
            self.species_plasma_frequencies[i] = sp.plasma_frequency
            self.QFactor += sp.QFactor / self.fourpie0

            wp_tot_sq += sp.plasma_frequency**2
            lambda_D += sp.debye_length**2

            if self.potential_type == "lj":
                self.species_lj_sigmas[i] = sp.sigma

        self.total_mass_density = self.species_masses.transpose() @ self.species_num_dens
        # Calculate total quantities
        self.total_net_charge = (self.species_charges.transpose()) @ self.species_num
        self.total_plasma_frequency = sqrt(wp_tot_sq)
        self.total_debye_length = sqrt(lambda_D)
        # Transform the list of species names into an array
        self.species_names = array(self.species_names)

        self.total_ion_temperature = (self.species_concentrations.transpose()) @ self.species_temperatures
        # Redundancy!!!
        self.T_desired = self.total_ion_temperature

        self.average_charge = (self.species_charges.transpose()) @ self.species_concentrations
        self.average_mass = (self.species_masses.transpose()) @ self.species_concentrations
        # Hydrodynamic Frequency
        self.hydrodynamic_frequency = sqrt(
            4.0 * pi * self.average_charge**2 * self.total_num_density / (self.fourpie0 * self.average_mass)
        )

    def from_dict(self, input_dict: dict) -> None:
        """
        Update attributes from input dictionary.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """
        self.__dict__.update(input_dict)

    def pretty_print(self):
        """
        Print simulation parameters in a user-friendly way.
        """

        box_a = self.box_lengths / self.a_ws
        pbox_a = self.pbox_lengths / self.a_ws
        msg = (
            f"\nSIMULATION AND INITIAL PARTICLE BOX:\n"
            f"Units: {self.units}\n"
            f"No. of non-zero box dimensions = {self.dimensions}\n"
            f"Wigner-Seitz radius = {self.a_ws:.6e} {self.units_dict['length']}\n"
            f"Box side along x axis = {box_a[0]:.6e} a_ws = {self.box_lengths[0]:.6e} {self.units_dict['length']}\n"
            f"Box side along y axis = {box_a[1]:.6e} a_ws = {self.box_lengths[1]:.6e} {self.units_dict['length']}\n"
            f"Box side along z axis = {box_a[2]:.6e} a_ws = {self.box_lengths[2]:.6e} {self.units_dict['length']}\n"
            f"Box Volume = {self.box_volume:.6e} {self.units_dict['volume']}\n"
            f"Initial particle box side along x axis = {pbox_a[0]:.6e} a_ws = {self.pbox_lengths[0]:.6e} {self.units_dict['length']}\n"
            f"Initial particle box side along y axis = {pbox_a[1]:.6e} a_ws = {self.pbox_lengths[1]:.6e} {self.units_dict['length']}\n"
            f"Initial particle box side along z axis = {pbox_a[2]:.6e} a_ws = {self.pbox_lengths[2]:.6e} {self.units_dict['length']}\n"
            f"Initial particle box Volume = {self.pbox_volume:.6e} {self.units_dict['volume']}\n"
            f"Boundary conditions: {self.boundary_conditions}\n"
        )
        if hasattr(self, "rand_seed"):
            msg += f"Random Seed = {self.rand_seed}\n"

        if self.magnetized:
            mag_msg = (
                f"\nMAGNETIC FIELD:\n"
                f"Magnetic Field = {self.magnetic_field} {self.units_dict['magnetic field strength']}\n"
                f"Magnetic Field Magnitude = {norm(self.magnetic_field):.4e} {self.units_dict['magnetic field strength']}\n"
                f"Magnetic Field Unit Vector = {self.magnetic_field / norm(self.magnetic_field)}\n"
            )

            msg += mag_msg
        restart = self.load_method
        restart_step = self.restart_step if self.restart_step else 0
        wp_dt = self.total_plasma_frequency * self.dt
        # Print Time steps information
        phase_dict = {
            "eq": ["equilibration", "equilibration_steps", "eq_dump_step"],
            "ma": ["magnetization", "magnetization_steps", "mag_dump_step"],
            "pr": ["production", "production_steps", "prod_dump_step"],
        }
        # Check for restart simulations
        phase_msg = "\nSIMULATION PHASES:"
        if restart[-7:] == "restart":
            phase_ls = phase_dict[restart[:2]]
            phase = phase_ls[0]
            steps = self.__dict__[phase_ls[1]]
            dump_step = self.__dict__[phase_ls[2]]

            phs_msg = (
                f"\nRestart step: {restart_step}\n"
                f"Total {phase} steps = {steps}\n"
                f"Total {phase} time = {steps * self.dt:.4e} {self.units_dict['time']} ~ {int(steps * wp_dt/(2.0 * pi))} plasma periods\n"
                f"snapshot interval step = {dump_step}\n"
                f"snapshot interval time = {dump_step * self.dt:.4e} {self.units_dict['time']} = {dump_step * wp_dt/(2.0 * pi):.4f} plasma periods\n"
                f"Total number of snapshots = {int(steps / dump_step)}"
            )

        else:
            phs_msg = ""
            for (key, phase_ls) in phase_dict.items():
                phase = phase_ls[0]
                steps = self.__dict__[phase_ls[1]]
                dump_step = self.__dict__[phase_ls[2]]
                if key == "eq" and not self.equilibration_phase:
                    continue
                elif key == "ma" and not self.magnetization_phase:
                    continue
                else:
                    phs_msg += (
                        f"\n{phase.capitalize()}:\n"
                        f"\tNo. of {phase} steps = {steps}\n"
                        f"\tTotal {phase} time = {steps * self.dt:.4e} {self.units_dict['time']} ~ {int(steps * wp_dt/(2.0 * pi))} plasma periods\n"
                        f"\tsnapshot interval step = {dump_step}\n"
                        f"\tsnapshot interval time = {dump_step * self.dt:.4e} {self.units_dict['time']} = {dump_step * wp_dt/(2.0 * pi):.4f} plasma periods\n"
                        f"\tTotal number of snapshots = {int(steps / dump_step)}"
                    )

        phase_msg += phs_msg
        msg += phase_msg
        print(msg)

    def set_species_attributes(self, species: list):
        """
        Set species attributes that have not been defined in the input file.

        Parameters
        ----------
        species: list,
            List of :class:`sarkas.plasma.Species`.

        """
        # Loop over species and assign missing attributes
        # Collect species properties in single arrays

        tot_num_ptcls = 0
        for i, sp in enumerate(species):

            tot_num_ptcls += sp.num
            # Calculate the mass of the species from the atomic weight if given
            if sp.atomic_weight:
                # Choose between atomic mass constant or proton mass
                # u = const.physical_constants["atomic mass constant"][0]
                sp.mass = self.mp * sp.atomic_weight
            else:
                sp.atomic_weight = sp.mass / self.mp

            # Calculate the mass of the species from the mass density if given
            if sp.mass_density:
                Av = physical_constants["Avogadro constant"][0]
                sp.number_density = sp.mass_density * Av / sp.atomic_weight

            if not hasattr(sp, "number_density"):
                raise AttributeError(f"\nSpecies {sp.name} number density not defined")

            # Calculate the temperature in K if eV has been provided and vice versa
            if sp.temperature_eV:
                sp.temperature = self.eV2K * sp.temperature_eV
            else:
                # Convert to eV and save
                sp.temperature_eV = sp.temperature / self.eV2K

            # Calculate the species charge based on the inputs
            if sp.charge:
                sp.Z = sp.charge / self.qe
            elif sp.Z:
                sp.charge = sp.Z * self.qe
            elif sp.epsilon:
                # Lennard-Jones potentials don't have charge but have the equivalent epsilon.
                sp.charge = sqrt(sp.epsilon)
                sp.Z = 1.0
            else:
                sp.charge = 0.0
                sp.Z = 0.0

            if sp.mass_density is None:
                sp.mass_density = sp.mass * sp.number_density

            # Q^2 factor see eq.(2.10) in Ballenegger et al. J Chem Phys 128 034109 (2008).
            sp.QFactor = sp.num * sp.charge**2  # In case of LJ this is zero

            sp.copy_params(self)
            sp.calc_ws_radius()
            sp.calc_plasma_frequency()
            sp.calc_debye_length()
            sp.calc_landau_length()

        # Calculate species concentrations
        for i, sp in enumerate(species):
            sp.concentration = float(sp.num / tot_num_ptcls)

    def setup(self, species) -> None:
        """
        Setup simulations' parameters.

        Parameters
        ----------
        species : list
            List of :class:`sarkas.plasma.Species` objects.

        """
        self.check_units()
        self.calc_parameters(species)
        self.calc_coupling_constant(species)
        self.calc_electron_properties(species)

    def sim_box_setup(self):
        """Calculate initial particle's and simulation's box parameters."""
        # Simulation Box Parameters
        # Wigner-Seitz radius calculated from the total number density
        # Calculate initial particle's and simulation's box parameters
        if self.np_per_side:
            if not isinstance(self.np_per_side, ndarray):
                self.np_per_side = array(self.np_per_side, dtype=int64)

            if rint(self.np_per_side.prod()) != self.total_num_ptcls:
                raise ParticlesError("Number of particles per dimension does not match total number of particles.")

            if self.dimensions != 3:
                new_array = zeros(3, dtype=int64)
                for d in range(self.dimensions):
                    new_array[d] = self.np_per_side[d]

                del self.np_per_side
                self.np_per_side = new_array.copy()

            self.pbox_lengths = self.np_per_side / self.total_num_density ** (1.0 / self.dimensions)

        else:
            self.pbox_lengths = zeros(3, dtype=float64)
            self.np_per_side = zeros(3, dtype=int64)
            for d in range(self.dimensions):
                self.pbox_lengths[d] = (self.total_num_ptcls / self.total_num_density) ** (1.0 / self.dimensions)
                self.np_per_side[d] = rint(self.total_num_ptcls ** (1.0 / self.dimensions))

        self.LPx, self.LPy, self.LPz = self.pbox_lengths.ravel()

        # The following if are needed if you define Lx, Ly, Lz in the input file
        if self.Lx == 0.0:
            self.box_lengths[0] = self.pbox_lengths[0]
            self.Lx = self.box_lengths[0]
        else:
            self.box_lengths[0] = self.Lx

        if self.Ly == 0.0:
            self.box_lengths[1] = self.pbox_lengths[1]
            self.Ly = self.box_lengths[1]
        else:
            self.box_lengths[1] = self.Ly

        if self.Lz == 0.0:
            self.box_lengths[2] = self.pbox_lengths[2]
            self.Lz = self.box_lengths[2]
        else:
            self.box_lengths[2] = self.Lz

        # Dev Note: The following are useful for future geometries.
        # Dev Note: Do we really need it?
        self.e1 = array([self.Lx, 0.0, 0.0])
        self.e2 = array([0.0, self.Ly, 0.0])
        self.e3 = array([0.0, 0.0, self.Lz])

        self.ep1 = array([self.LPx, 0.0, 0.0])
        self.ep2 = array([0.0, self.LPy, 0.0])
        self.ep3 = array([0.0, 0.0, self.LPz])

        if self.dimensions == 3:
            self.a_ws = (3.0 / (4.0 * pi * self.total_num_density)) ** (1.0 / 3.0)
            self.box_volume = abs(cross(self.e1, self.e2).dot(self.e3))
            self.pbox_volume = abs(cross(self.ep1, self.ep2).dot(self.ep3))

        elif self.dimensions == 2:
            self.a_ws = 1.0 / sqrt(pi * self.total_num_density)

            self.box_volume = abs(cross(self.e1, self.e2)[-1])
            self.pbox_volume = abs(cross(self.ep1, self.ep2)[-1])
        else:
            self.a_ws = 2.0 / self.total_num_density
            self.box_volume = self.Lx
            self.pbox_volume = self.LPx
