import time
import yaml
import numpy as np
import os.path
import sys
import scipy.constants as const
import fdint

from sarkas.io.base import Checkpoint, Verbose
from sarkas.tools import postprocessing
from sarkas.potentials.base import Potential
from sarkas.time_evolution.integrators import Integrator
from sarkas.time_evolution.thermostats import Thermostat


class Simulation:

    def __init__(self):

        self.potential = Potential()
        self.integrator = Integrator()
        self.thermostat = Thermostat()
        self.params = Parameters()
        self.ptcls = Particles()
        self.species = []
        # self.checkpoint = Checkpoint()
        # self.verbose = Verbose()
        self.input_file = None

    def common_parser(self, filename):

        self.input_file = filename
        self.params.input_file = filename
        with open(filename, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)
            for lkey in dics:
                if lkey == "Particles":
                    for species in dics["Particles"]:
                        spec = Species()
                        for key, value in species["Species"].items():
                            if hasattr(spec, key):
                                spec.__dict__[key] = value
                            else:
                                setattr(spec, key, value)
                        self.species.append(spec)

                if lkey == "Potential":
                    for key, value in dics[lkey].items():
                        if hasattr(self.potential, key):
                            self.potential.__dict__[key] = value
                        else:
                            setattr(self.potential, key, value)

                if lkey == "Thermostat":
                    for key, value in dics[lkey].items():
                        if hasattr(self.thermostat, key):
                            self.thermostat.__dict__[key] = value
                        else:
                            setattr(self.thermostat, key, value)

                if lkey == "Integrator":
                    for key, value in dics[lkey].items():
                        if hasattr(self.integrator, key):
                            self.integrator.__dict__[key] = value
                        else:
                            setattr(self.integrator, key, value)

                if lkey == "Parameters":
                    for key, value in dics[lkey].items():
                        if hasattr(self.params, key):
                            self.params.__dict__[key] = value
                        else:
                            setattr(self.params, key, value)

                if lkey == "PostProcessing":
                    for key, value in dics[lkey].items():
                        if key == 'RadialDistributionFunction':
                            self.post_proc.rdf = postprocessing.RadialDistributionFunction()
                        if hasattr(self.postproc, key):
                            self.postproc.__dict__[key] = value
                        else:
                            setattr(self.postproc, key, value)

    def equilibrate(self, it_start):

        if self.params.verbose:
            print("\n------------- Equilibration -------------")
        time_init = time.time()
        potential_energy = self.integrator.equilibrate(it_start, self.ptcls, self.checkpoint)
        time_eq = time.time()
        self.verbose.time_stamp("Equilibration", time_eq - time_init)

        return potential_energy

    def evolve(self, potential_energy):
        """Time evolution.

        """
        ##############################################
        # Prepare for Production Phase
        ##############################################

        # Open output files
        if self.params.load_method == "restart":
            it_start = self.params.load_restart_step
        else:
            it_start = 0
            # Restart the pbc counter
            self.ptcls.pbc_cntr.fill(0.0)

        if self.params.verbose:
            print("\n------------- Production -------------")

        # Update measurement flag for rdf
        self.potential.measure = True
        self.checkpoint.dump(True, self.ptcls, potential_energy, 0)

        time_init = time.time()
        potential_energy = self.integrator.produce(it_start, self.ptcls, self.checkpoint)
        time_end = time.time()

        self.verbose.time_stamp("Production", time_end - time_init)

        return potential_energy

    def initialization(self):
        time_start = time.time()
        self.integrator.setup(self.params, self.thermostat, self.potential)
        self.thermostat.setup(self.params)
        self.potential.setup(self.params)

        self.ptcls.setup(self.params, self.species)

        # For restart and pva backups.
        self.checkpoint.save_pickle(self.params)
        self.verbose.sim_setting_summary(self)  # simulation setting summary
        time_end = time.time()
        self.verbose.time_stamp("Initialization", time_start - time_end)

    def pre_processing(self):
        pass

    def post_processing(self, time0):

        ##############################################
        # Finalization Phase
        ##############################################
        time_init = time.time()
        rdf = postprocessing.RadialDistributionFunction(self.params, self.species, self.potential.rc)
        rdf.save(self.ptcls.rdf_hist)
        rdf.plot(show=False)
        time_end = time.time()
        self.verbose.time_stamp("Post Processing", time_end - time_init)

        time_tot = time.time()
        self.verbose.time_stamp("Total", time_tot - time0)

    def run(self):

        time0 = time.time()
        self.initialization()
        if not self.params.load_method == 'prod_restart':
            if self.params.load_method == "therm_restart":
                it_start = self.params.load_therm_restart_step
            else:
                it_start = 0

            potential_energy = self.equilibrate(it_start)

        else:

            if not self.potential.method == "FMM":
                potential_energy = self.potential.calc_pot_acc(self.ptcls)
            # else:
            # potential_energy = self.potential.calc_pot_acc_fmm(self.ptcls, self.params)

        self.evolve(potential_energy)
        self.post_processing(time0)

    def setup(self, other_inputs=None):
        """Setup all simulations subclasses.

        Parameters
        ----------
        other_inputs : dict (optional)
            Dictionary with additional simulations options.

        """
        # save the type
        self.params.potential_type = self.potential.type
        self.params.magnetized = self.integrator.magnetized
        self.params.integrator = self.integrator.type
        self.params.dt = self.integrator.dt
        self.params.thermostat = self.thermostat.type

        self.params.setup(self.species, other_inputs)

        self.checkpoint = Checkpoint(self.params, self.species)
        self.verbose = Verbose(self.params)

        self.postproc.setup(self.params, self.species)


class Parameters:
    """
    Simulation's Parameters.

    Attributes
    ----------
    aws : float
        Wigner-Seitz radius. Calculated from the ``total_num_density`` .

    box_volume : float
        Box volume.

    dimensions : int
        Number of non-zero dimensions.

    fourpie0: float
        Electrostatic constant :math: `4\pi \espilon_0`.

    species : list
        List of Species objects with species' information.

    load_method : str
        Particles loading described in Species.

    load_rand_seed : int
        Seed of random number generator.

    load_restart_step : int
        Restart time step.

    load_r_reject : float
        Rejection radius to avoid placing particles to close to each other.

    load_perturb : float
        Strength of initial perturbation.

    load_halton_bases : array, shape(3)
        Array of 3 ints each of which is a base for the Halton sequence.

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

    L : float
        Smallest box length.

    Lx : float
        Box length in the :math:`x` direction.

    Ly : float
        Box length in the :math:`y` direction.

    Lz : float
        Box length in the :math:`z` direction.

    Lv : array, shape(3)
        Box length in each direction.

    e1 : float
        Unit vector in the :math:`x` direction.

    e2 : float
        Unit vector in the :math:`y` direction.

    e3 : float
        Unit vector in the :math:`z` direction.

    N : int
        Total number of particles same as ``tot_num_ptcls``.

    ptcls_input_file : str
        User defined input file containing particles' data.

    input_file : str
        YAML Input file with all the simulation's parameters.

    Te : float
        Equilibrium electron temperature. Defined in Potential module.

    Ti : float
        Total Equilibrium Ion temperature.

    T_desired : float
        Equilibrium temperature.

    tot_net_charge : float
        Total charge in the system.

    total_num_density : float
        Total number density. Calculated from the sum of ``Species.num_density``.

    total_num_ptcls : int
        Total number of particles. Calculated from the sum of ``Species.num``.

    wp : float
        Total Plasma frequency. Defined in Potential module.

    force : func
        Function for force calculation. Assigned in Potential module.
    """

    def __init__(self):
        # Container of Species
        self.ne = 0.0
        self.Lx = 0.0
        self.Ly = 0.0
        self.Lz = 0.0
        self.box_lengths = []
        self.box_volume = 0.0
        self.input_file = None
        self.dimensions = 3
        self.J2erg = 1.0e+7  # erg/J
        self.eps0 = const.epsilon_0
        self.fourpie0 = 4.0 * np.pi * self.eps0
        self.mp = const.physical_constants["proton mass"][0]
        self.me = const.physical_constants["electron mass"][0]
        self.qe = const.physical_constants["elementary charge"][0]
        self.hbar = const.hbar
        self.hbar2 = self.hbar ** 2
        self.c0 = const.physical_constants["speed of light in vacuum"][0]
        self.eV2K = const.physical_constants["electron volt-kelvin relationship"][0]
        self.eV2J = const.physical_constants["electron volt-joule relationship"][0]
        self.a0 = const.physical_constants["Bohr radius"][0]
        self.kB = const.Boltzmann
        self.aws = 0.0
        self.tot_net_charge = 0.0
        self.QFactor = 0.0
        self.num_species = 1
        self.total_ion_temperature = 0.0
        self.T_desired = 0.0
        self.total_num_density = 0.0
        self.total_num_ptcls = 0
        self.species = []
        #
        # Control
        self.measure = False
        self.verbose = True
        self.simulations_dir = "Simulations"
        self.job_dir = None
        self.production_dir = 'Production'
        self.equilibration_dir = 'Equilibration'
        self.preprocessing_dir = "PreProcessing"
        self.postprocessing_dir = "PostProcessing"
        self.prod_dump_dir = 'dumps'
        self.eq_dump_dir = 'dumps'
        self.job_id = None
        self.log_file = None
        self.np_per_side = []
        self.pre_run = False

    def setup(self, species, args=None):
        """
        Setup simulations' parameters.

        Parameters
        ----------
        species
        args : dict
            Input arguments
        """
        if args is None:
            args = {"simulations_dir": "Simulations",
                    "job_dir": os.path.basename(self.input_file).split('.')[0],
                    "production_dir": 'Production',
                    "equilibration_dir": 'Equilibration',
                    "preprocessing_dir": "PreProcessing",
                    "postprocessing_dir": "PostProcessing",
                    "prod_dump_dir": 'dumps',
                    "eq_dump_dir": 'dumps',
                    }
        self.create_directories(args)

        self.calc_parameters(species)
        self.calc_electron_properties()
        self.calc_coupling_constant(species)

    def calc_parameters(self, species):
        """ Assign the parsed parameters"""
        # Physical constants
        if self.units == "cgs":
            self.kB *= self.J2erg
            self.c0 *= 1e2  # cm/s
            # Coulomb to statCoulomb conversion factor. See https://en.wikipedia.org/wiki/Statcoulomb
            C2statC = 1.0e-01 * self.c0
            self.hbar = self.J2erg * self.hbar
            self.hbar2 = self.hbar ** 2
            self.qe *= C2statC
            self.me *= 1.0e3
            self.eps0 = 1.0
            self.fourpie0 = 1.0
            self.a0 *= 1e2

        self.num_species = len(species)
        # Loop over species and assign missing attributes
        # Collect species properties in single arrays
        self.species_num = np.zeros(self.num_species, dtype=int)
        self.species_concentrations = np.zeros(self.num_species)
        self.species_temperatures = np.zeros(self.num_species)
        self.species_masses = np.zeros(self.num_species)
        self.species_charges = np.zeros(self.num_species)
        self.species_names = []

        self.species_wp = np.zeros(self.num_species)
        self.species_num_dens = np.zeros(self.num_species)

        wp_tot_sq = 0.0
        lambda_D = 0.0

        if self.magnetized:
            self.species_cyclotron_frequencies = np.zeros(self.num_species)

        for i, sp in enumerate(species):
            self.total_num_ptcls += sp.num

            if sp.atomic_weight is not None:
                # Choose between atomic mass constant or proton mass
                # u = const.physical_constants["atomic mass constant"][0]
                if self.units == "cgs":
                    self.mp *= 1e3
                    sp.mass = self.mp * sp.atomic_weight
                elif self.units == "mks":
                    sp.mass = self.mp * sp.atomic_weight

            if sp.mass_density is not None:
                Av = const.physical_constants["Avogadro constant"][0]
                sp.number_density = sp.mass_density * Av / sp.atomic_weight
                self.total_num_density += sp.number_density

            assert sp.number_density is not None, "{} number density not defined".format(sp.name)

            self.total_num_density += sp.number_density

            self.species_names.append(sp.name)
            self.species_num[i] = sp.num
            self.species_masses[i] = sp.mass
            if hasattr(sp, 'temperature_eV'):
                sp.temperature = self.eV2K * sp.temperature_eV

            self.species_temperatures[i] = sp.temperature

            if sp.charge is not None:
                self.species_charges[i] = sp.charge
                sp.Z = sp.charge / self.qe
            elif sp.Z is not None:
                self.species_charges[i] = sp.Z * self.qe
                sp.charge = sp.Z * self.qe
            else:
                self.species_charges[i] = 0.0
                sp.charge = 0.0
                sp.Z = 0.0

            # Q^2 factor see eq.(2.10) in Ballenegger et al. J Chem Phys 128 034109 (2008)
            sp.QFactor = sp.num * sp.charge ** 2
            self.QFactor += sp.QFactor / self.fourpie0

            if self.magnetized:
                if self.units == "cgs":
                    sp.calc_cyclotron_frequency(self.magnetic_field_strength/self.c0)
                else:
                    sp.calc_cyclotron_frequency(self.magnetic_field_strength)
                self.species_cyclotron_frequencies[i] = sp.omega_c

            # Calculate the (total) plasma frequency
            if not self.potential_type == "LJ":
                sp.calc_plasma_frequency(self.fourpie0)
                wp_tot_sq += sp.wp ** 2
                sp.calc_debye_length(self.kB, self.fourpie0)
                lambda_D += sp.debye_length ** 2

            self.species_wp[i] = sp.wp
            self.species_num_dens[i] = sp.number_density

        for i, sp in enumerate(species):
            sp.concentration = sp.num / self.total_num_ptcls
            self.species_concentrations[i] = float(sp.num / self.total_num_ptcls)

        self.total_net_charge = np.transpose(self.species_charges) @ self.species_num
        self.total_ion_temperature = np.transpose(self.species_concentrations) @ self.species_temperatures
        self.total_plasma_frequency = np.sqrt(wp_tot_sq)
        self.total_debye_length = np.sqrt(lambda_D)
        self.tot_mass_density = self.species_masses.transpose() @ self.species_num_dens

        # Simulation Box Parameters
        # Wigner-Seitz radius calculated from the total number density
        self.aws = (3.0 / (4.0 * np.pi * self.total_num_density)) ** (1. / 3.)

        if len(self.np_per_side) != 0:
            msg = "Number of particles per dimension does not match total number of particles."
            assert int(np.prod(self.np_per_side)) == self.total_num_ptcls, msg

            self.Lx = self.aws * self.np_per_side[0] * (4.0 * np.pi / 3.0) ** (1.0 / 3.0)
            self.Ly = self.aws * self.np_per_side[1] * (4.0 * np.pi / 3.0) ** (1.0 / 3.0)
            self.Lz = self.aws * self.np_per_side[2] * (4.0 * np.pi / 3.0) ** (1.0 / 3.0)
        else:
            self.Lx = self.aws * (4.0 * np.pi * self.total_num_ptcls / 3.0) ** (1.0 / 3.0)
            self.Ly = self.aws * (4.0 * np.pi * self.total_num_ptcls / 3.0) ** (1.0 / 3.0)
            self.Lz = self.aws * (4.0 * np.pi * self.total_num_ptcls / 3.0) ** (1.0 / 3.0)

        self.box_lengths = np.array([self.Lx, self.Ly, self.Lz])  # box length vector

        # Dev Note: The following are useful for future geometries
        self.e1 = np.array([self.Lx, 0.0, 0.0])
        self.e2 = np.array([0.0, self.Ly, 0.0])
        self.e3 = np.array([0.0, 0.0, self.Lz])

        self.box_volume = abs(np.dot(np.cross(self.e1, self.e2), self.e3))

        self.dimensions = np.count_nonzero(self.box_lengths)  # no. of dimensions

        # Redundancy!!!
        self.T_desired = self.total_ion_temperature

        # # boundary Conditions
        # if self.BC.pbc_axes:
        #     self.BC.pbc_axes_indx = np.zeros(len(self.BC.pbc_axes))
        #     for (ij, bc) in enumerate(self.BC.pbc_axes):
        #         if bc == "x":
        #             self.BC.pbc_axes_indx[ij] = 0
        #         elif bc == "y":
        #             self.BC.pbc_axes_indx[ij] = 1
        #         elif bc == "z":
        #             self.BC.pbc_axes_indx[ij] = 2
        #
        # if self.BC.mm_axes:
        #     print("\nOnly Periodic Boundary Conditions are supported. Bye!")
        #     sys.exit()
        #     self.BC.mm_axes_indx = np.zeros(len(self.BC.mm_axes), dtype=np.int)
        #     for (ij, bc) in enumerate(self.BC.mm_axes):
        #         if bc == "x":
        #             self.BC.mm_axes_indx[ij] = 0
        #         elif bc == "y":
        #             self.BC.mm_axes_indx[ij] = 1
        #         elif bc == "z":
        #             self.BC.mm_axes_indx[ij] = 2
        #
        # if self.BC.open_axes:
        #     self.BC.open_axes_indx = np.zeros(len(self.BC.open_axes), dtype=np.int)
        #     for (ij, bc) in enumerate(self.BC.open_axes):
        #         if bc == "x":
        #             self.BC.open_axes_indx[ij] = 0
        #         elif bc == "y":
        #             self.BC.open_axes_indx[ij] = 1
        #         elif bc == "z":
        #             self.BC.open_axes_indx[ij] = 2

    def create_directories(self, args):
        """
        Check for undefined control variables and create output directory and its subdirectories.

        Parameters
        ---------
        args: dict
            Input arguments.

        """
        # Check for directories
        for key, value in args.items():
            if hasattr(self, key):
                self.__dict__[key] = value

        # Check if the directories exist
        if not os.path.exists(self.simulations_dir):
            os.mkdir(self.simulations_dir)

        if self.job_id is None:
            self.job_id = self.job_dir

        self.job_dir = os.path.join(self.simulations_dir, self.job_dir)
        if not os.path.exists(self.job_dir):
            os.mkdir(self.job_dir)

        # Equilibration directory and sub_dir
        self.equilibration_dir = os.path.join(self.job_dir, self.equilibration_dir)
        if not os.path.exists(self.equilibration_dir):
            os.mkdir(self.equilibration_dir)

        self.eq_dump_dir = os.path.join(self.equilibration_dir, 'dumps')
        if not os.path.exists(self.eq_dump_dir):
            os.mkdir(self.eq_dump_dir)

        # Production dir and sub_dir
        self.production_dir = os.path.join(self.job_dir, self.production_dir)
        if not os.path.exists(self.production_dir):
            os.mkdir(self.production_dir)

        self.prod_dump_dir = os.path.join(self.production_dir, "dumps")
        if not os.path.exists(self.prod_dump_dir):
            os.mkdir(self.prod_dump_dir)

        # Postprocessing dir
        self.postprocessing_dir = os.path.join(self.job_dir, self.postprocessing_dir)
        if not os.path.exists(self.postprocessing_dir):
            os.mkdir(self.postprocessing_dir)

        if self.log_file is None:
            self.log_file = os.path.join(self.job_dir, 'log.out')

    def calc_electron_properties(self):

        twopi = 2.0 * np.pi
        beta_i = 1.0 / (self.kB * self.total_ion_temperature)

        if not hasattr(self, "Te"):
            # if the electron temperature is not defined. The total ion temperature will be used for it.
            self.Te = self.total_ion_temperature

        self.ne = self.species_charges.transpose() @ self.species_concentrations * self.total_num_density / self.qe
        # Calculate electron gas properties
        fdint_fdk_vec = np.vectorize(fdint.fdk)
        fdint_ifd1h_vec = np.vectorize(fdint.ifd1h)
        beta_e = 1. / (self.kB * self.Te)
        lambda_DB = np.sqrt(twopi * self.hbar2 * beta_e / self.me)
        lambda3 = lambda_DB ** 3
        # chemical potential of electron gas/(kB T). See eq.(4) in Ref.[3]_
        self.eta_e = fdint_ifd1h_vec(lambda3 * np.sqrt(np.pi) * self.ne / 4.0)
        # Thomas-Fermi length obtained from compressibility. See eq.(10) in Ref. [3]_
        self.lambda_TF = np.sqrt(self.fourpie0 * np.sqrt(np.pi) * lambda3 / (
                8.0 * np.pi * self.qe ** 2 * beta_e * fdint_fdk_vec(k=-0.5, phi=self.eta_e)))

        self.rs = self.aws / self.a0
        kF = (3.0 * np.pi**2 * self.ne) ** (1./3.)
        self.fermi_energy = self.hbar2 * kF ** 2 / (2.0 * self.me)
        self.electron_degeneracy_parameter = self.kB * self.Te / self.fermi_energy
        self.relativistic_parameter = self.hbar * kF / (self.me * self.c0)

    def calc_coupling_constant(self,species):

        z_avg = np.transpose(self.species_charges) @ self.species_concentrations
        self.species_couplings = np.zeros(self.num_species)
        self.coupling_constant = 0.0
        for i, sp in enumerate(species):
            const = self.fourpie0 * self.kB
            sp.calc_coupling(self.aws, z_avg, const)
            self.species_couplings[i] = sp.coupling
            self.coupling_constant += sp.concentration * sp.coupling


class Particles:
    """
    Particles class.

    Parameters
    ----------
    params : object
        Simulation's parameters

    Attributes
    ----------
    pos : array
        Particles' positions.

    vel : array
        Particles' velocities.

    acc : array
        Particles' accelerations.

    params : class
        Simulation's parameters.

    tot_num_ptcls: int
        Total number of particles.

    box_lengths : array
        Box sides' lengths.

    mass : array
        Mass of each particle.

    charge : array
        Charge of each particle.

    species_id : array, shape(N,)
        Species identifier.

    species_name : list
        Species' names.

    species_num : array
        Number of particles of each species.

    rdf_nbins : int
        Number of bins for radial pair distribution.

    no_grs : int
        Number of independent :math:`g_{ij}(r)`.

    rdf_hist : array
        Histogram array for the radial pair distribution function.
    """

    def __init__(self):
        pass

    def setup(self, params, species):
        """
        Initialize the attributes
        """
        self.prod_dump_dir = params.prod_dump_dir
        self.eq_dump_dir = params.eq_dump_dir
        self.box_lengths = params.box_lengths
        self.total_num_ptcls = params.total_num_ptcls
        self.num_species = params.num_species
        self.species_num = params.species_num
        self.dimensions = params.dimensions

        if hasattr(params, "rand_seed"):
            self.rnd_gen = np.random.Generator(np.random.PCG64(params.rand_seed))
        else:
            self.rnd_gen = np.random.Generator(np.random.PCG64(123456789))

        self.pos = np.zeros((self.total_num_ptcls, params.dimensions))
        self.vel = np.zeros((self.total_num_ptcls, params.dimensions))
        self.acc = np.zeros((self.total_num_ptcls, params.dimensions))

        self.pbc_cntr = np.zeros((self.total_num_ptcls, params.dimensions))

        self.names = np.empty(self.total_num_ptcls, dtype=str)
        self.id = np.zeros(self.total_num_ptcls, dtype=int)

        self.species_init_vel = np.zeros((params.num_species, 3))
        self.species_thermal_velocity = np.zeros((params.num_species, 3))

        self.masses = np.zeros(self.total_num_ptcls)  # mass of each particle
        self.charges = np.zeros(self.total_num_ptcls)  # charge of each particle

        # No. of independent rdf
        self.no_grs = int(self.num_species * (self.num_species + 1) / 2)
        self.rdf_nbins = params.rdf_nbins
        self.rdf_hist = np.zeros((self.rdf_nbins, self.num_species, self.num_species))

        self.update_attributes(species, params.kB)

        self.load(params)

    def load(self, params):
        """
        Initialize particles' positions and velocities.
        Positions are initialized based on the load method while velocities are chosen
        from a Maxwell-Boltzmann distribution.

        """
        # Particles Position Initialization
        if params.verbose:
            print('\nAssigning initial positions according to {}'.format(params.load_method))

        if params.load_method == 'prod_restart':
            msg = "Restart step not defined. Please define restart_step."
            assert params.load_restart_step is not None, msg
            assert type(params.load_restart_step) is int, "Only integers are allowed."

            self.load_from_restart(False, params.load_restart_step)

        elif params.load_method == 'eq_restart':
            msg = "Therm Restart step not defined. Please define restart_step"
            assert params.load_therm_restart_step is not None, msg
            assert type(params.load_therm_restart_step) is int, "Only integers are allowed."

            self.load_from_restart(True, params.load_therm_restart_step)

        elif params.load_method == 'file':
            msg = 'Input file not defined. Please define particle_input_file.'
            assert params.ptcls_input_file is not None, msg
            self.load_from_file(params.ptcls_input_file)

        # position distribution.
        elif params.load_method == 'lattice':
            self.lattice(params.load_perturb)

        elif params.load_method == 'random_reject':
            self.random_reject(params.load_r_reject)

        elif params.load_method == 'halton_reject':
            self.halton_reject(params.load_halton_bases, params.load_r_reject)

        elif params.load_method in ['uniform', 'random_no_reject']:
            self.pos = self.uniform_no_reject([0.0, 0.0, 0.0], params.box_lengths)

        else:
            raise AttributeError('Incorrect particle placement scheme specified.')

    def gaussian(self, mean, sigma, num_ptcls):
        """Initialize particles' velocities according to a Maxwell-Boltzmann distribution.

        Parameters
        ----------
        num_ptcls
        mean
        sigma
        """
        return self.rnd_gen.normal(mean, sigma, (num_ptcls, 3))

    def update_attributes(self, species, kB):
        # Assign particles attributes
        species_end = 0
        for ic, sp in enumerate(species):
            species_start = species_end
            species_end += sp.num

            self.names[species_start:species_end] = sp.name
            self.masses[species_start:species_end] = sp.mass

            if hasattr(sp, 'charge'):
                self.charges[species_start:species_end] = sp.charge
            else:
                self.charges[species_start:species_end] = 1.0

            self.id[species_start:species_end] = ic

            if hasattr(sp, "init_vel"):
                self.species_init_vel[ic, :] = sp.init_vel

            if sp.initial_velocity_distribution == "boltzmann":
                if isinstance(sp.temperature, (int, float)):
                    sp_temperature = np.ones(self.dimensions) * sp.temperature

                self.species_thermal_velocity[ic] = np.sqrt(kB * sp_temperature / sp.mass)
                self.vel[species_start:species_end, :] = self.gaussian(sp.initial_velocity,
                                                                       self.species_thermal_velocity[ic], sp.num)

    def load_from_restart(self, equilibration, it):
        """
        Load particles' data from a checkpoint of a previous run

        Parameters
        ----------
        it : int
            Timestep.

        equilibration: bool
            Flag for restart phase.

        """
        if equilibration:
            file_name = os.path.join(self.eq_dump_dir, "checkpoint_" + str(it) + ".npz")
            data = np.load(file_name, allow_pickle=True)
            self.id = data["species_id"]
            self.names = data["species_name"]
            self.pos = data["pos"]
            self.vel = data["vel"]
            self.acc = data["acc"]

        else:
            file_name = os.path.join(self.prod_dump_dir, "checkpoint_" + str(it) + ".npz")
            data = np.load(file_name, allow_pickle=True)
            self.id = data["species_id"]
            self.species_name = data["species_name"]
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
        pv_data = np.loadtxt(f_name)
        if not (pv_data.shape[0] == self.total_num_ptcls):
            print("Number of particles is not same between input file and initial p & v data file.")
            print("From the input file: N = ", self.total_num_ptcls)
            print("From the initial p & v data: N = ", pv_data.shape[0])
            sys.exit()
        self.pos[:, 0] = pv_data[:, 0]
        self.pos[:, 1] = pv_data[:, 1]
        self.pos[:, 2] = pv_data[:, 2]

        self.vel[:, 0] = pv_data[:, 3]
        self.vel[:, 1] = pv_data[:, 4]
        self.vel[:, 2] = pv_data[:, 5]

    def uniform_no_reject(self, mins, maxs):
        """
        Randomly distribute particles along each direction.

        Returns
        -------
        pos : array
            Particles' positions.

        """

        return self.rnd_gen.uniform( mins, maxs, (self.total_num_ptcls,3))

    def lattice(self, perturb):
        """
        Place particles in a simple cubic lattice with a slight perturbation ranging
        from 0 to 0.5 times the lattice spacing.

        Parameters
        ----------
        perturb : float
            Value of perturbation, p, such that 0 <= p <= 1.

        Notes
        -----
        Author: Luke Stanek
        Date Created: 5/6/19
        Date Updated: 6/2/19
        Updates: Added to S_init_schemes.py for Sarkas import
        """

        # Check if perturbation is below maximum allowed. If not, default to maximum perturbation.
        if perturb > 1:
            print('Warning: Random perturbation must not exceed 1. Setting perturb = 1.')
            perturb = 1  # Maximum perturbation

        print('Initializing particles with maximum random perturbation of {} times the lattice spacing.'.format(
            perturb * 0.5))

        # Determining number of particles per side of simple cubic lattice
        part_per_side = self.total_num_ptcls ** (1. / 3.)  # Number of particles per side of cubic lattice

        # Check if total number of particles is a perfect cube, if not, place more than the requested amount
        if round(part_per_side) ** 3 != self.total_num_ptcls:
            part_per_side = np.ceil(self.total_num_ptcls ** (1. / 3.))
            print('\nWARNING: Total number of particles requested is not a perfect cube.')
            print('Initializing with {} particles.'.format(int(part_per_side ** 3)))

        dx_lattice = self.box_lengths[0] / (self.total_num_ptcls ** (1. / 3.))  # Lattice spacing
        dz_lattice = self.box_lengths[1] / (self.total_num_ptcls ** (1. / 3.))  # Lattice spacing
        dy_lattice = self.box_lengths[2] / (self.total_num_ptcls ** (1. / 3.))  # Lattice spacing

        # Start timer
        start = time.time()

        # Create x, y, and z position arrays
        x = np.arange(0, self.box_lengths[0], dx_lattice) + 0.5 * dx_lattice
        y = np.arange(0, self.box_lengths[1], dy_lattice) + 0.5 * dy_lattice
        z = np.arange(0, self.box_lengths[2], dz_lattice) + 0.5 * dz_lattice

        # Create a lattice with appropriate x, y, and z values based on arange
        X, Y, Z = np.meshgrid(x, y, z)

        # Perturb lattice
        X += self.rnd_gen.uniform(-0.5, 0.5, np.shape(X)) * perturb * dx_lattice
        Y += self.rnd_gen.uniform(-0.5, 0.5, np.shape(Y)) * perturb * dy_lattice
        Z += self.rnd_gen.uniform(-0.5, 0.5, np.shape(Z)) * perturb * dz_lattice

        # Flatten the meshgrid values for plotting and computation
        self.pos[:, 0] = X.ravel()
        self.pos[:, 1] = Y.ravel()
        self.pos[:, 2] = Z.ravel()

        # End timer
        end = time.time()
        print('Lattice creation took: {:1.4e} sec'.format(end - start))

    def random_reject(self, r_reject):
        """
        Place particles by sampling a uniform distribution from 0 to L (the box length)
        and uses a rejection radius to avoid placing particles to close to each other.

        Parameters
        ----------
        r_reject : float
            Value of rejection radius.

        Notes
        -----
        Author: Luke Stanek
        Date Created: 5/6/19
        Date Updated: N/A
        Updates: N/A

        """

        # Initialize Arrays
        x = np.array([])
        y = np.array([])
        z = np.array([])

        # Set first x, y, and z positions
        x_new = self.rnd_gen.uniform(0, self.box_lengths[0])
        y_new = self.rnd_gen.uniform(0, self.box_lengths[1])
        z_new = self.rnd_gen.uniform(0, self.box_lengths[2])

        # Append to arrays
        x = np.append(x, x_new)
        y = np.append(y, y_new)
        z = np.append(z, z_new)

        # Particle counter
        i = 0

        start = time.time()  # Start timer for placing particles
        # Loop to place particles
        while i < self.total_num_ptcls - 1:

            # Set x, y, and z positions
            x_new = self.rnd_gen.uniform(0.0, self.box_lengths[0])
            y_new = self.rnd_gen.uniform(0.0, self.box_lengths[1])
            z_new = self.rnd_gen.uniform(0.0, self.box_lengths[2])

            # Check if particle was place too close relative to all other current particles
            for j in range(len(x)):

                # Flag for if particle is outside of cutoff radius (True -> not inside rejection radius)
                flag = 1

                # Compute distance b/t particles for initial placement
                x_diff = x_new - x[j]
                y_diff = y_new - y[j]
                z_diff = z_new - z[j]

                # periodic condition applied for minimum image
                if x_diff < - self.box_lengths[0] / 2:
                    x_diff = x_diff + self.box_lengths[0]
                if x_diff > self.box_lengths[0] / 2:
                    x_diff = x_diff - self.box_lengths[0]

                if y_diff < - self.box_lengths[1] / 2:
                    y_diff = y_diff + self.box_lengths[1]
                if y_diff > self.box_lengths[1] / 2:
                    y_diff = y_diff - self.box_lengths[1]

                if z_diff < -self.box_lengths[2] / 2:
                    z_diff = z_diff + self.box_lengths[2]
                if z_diff > self.box_lengths[2] / 2:
                    z_diff = z_diff - self.box_lengths[2]

                # Compute distance
                r = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)

                # Check if new particle is below rejection radius. If not, break out and try again
                if r <= r_reject:
                    flag = 0  # new position not added (False -> no longer outside reject r)
                    break

            # If flag true add new position
            if flag == 1:
                x = np.append(x, x_new)
                y = np.append(y, y_new)
                z = np.append(z, z_new)

                # Increment particle number
                i += 1

        self.pos[:, 0] = x
        self.pos[:, 1] = y
        self.pos[:, 2] = z

        end = time.time()
        print('Uniform distribution with rejection radius took : {:1.4e} sec'.format(end - start))

    def halton_reject(self, bases, r_reject):
        """
        Place particles according to a Halton sequence from 0 to L (the box length)
        and uses a rejection radius to avoid placing particles to close to each other.

        Parameters
        ----------
        bases : array
            Array of 3 ints each of which is a base for the Halton sequence.
            Defualt: bases = np.array([2,3,5])

        r_reject : float
            Value of rejection radius.

        Notes
        -----
        Author: Luke Stanek
        Date Created: 5/6/19
        Date Updated: N/A
        Updates: N/A

        """

        # Get bases
        b1, b2, b3 = bases

        # Allocate space and store first value from Halton
        x = np.array([0])
        y = np.array([0])
        z = np.array([0])

        # Initialize particle counter and Halton counter
        i = 1
        k = 1

        # Start timer
        start = time.time()  # Start timer for placing particles

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
                n = np.floor(n / b1)
            x_new = self.box_lengths[0] * r1  # new x value

            # Determine y coordinate
            f2 = 1
            r2 = 0
            while m > 0:
                f2 /= b2
                r2 += f2 * (m % int(b2))
                m = np.floor(m / b2)
            y_new = self.box_lengths[1] * r2  # new y value

            # Determine z coordinate
            f3 = 1
            r3 = 0
            while p > 0:
                f3 /= b3
                r3 += f3 * (p % int(b3))
                p = np.floor(p / b3)
            z_new = self.box_lengths[2] * r3  # new z value

            # Check if particle was place too close relative to all other current particles
            for j in range(len(x)):

                # Flag for if particle is outside of cutoff radius (1 -> not inside rejection radius)
                flag = 1

                # Compute distance b/t particles for initial placement
                x_diff = x_new - x[j]
                y_diff = y_new - y[j]
                z_diff = z_new - z[j]

                # Periodic condition applied for minimum image
                if x_diff < - self.box_lengths[0] / 2:
                    x_diff = x_diff + self.box_lengths[0]
                if x_diff > self.box_lengths[0] / 2:
                    x_diff = x_diff - self.box_lengths[0]

                if y_diff < -self.box_lengths[1] / 2:
                    y_diff = y_diff + self.box_lengths[1]
                if y_diff > self.box_lengths[1] / 2:
                    y_diff = y_diff - self.box_lengths[1]

                if z_diff < -self.box_lengths[2] / 2:
                    z_diff = z_diff + self.box_lengths[2]
                if z_diff > self.box_lengths[2] / 2:
                    z_diff = z_diff - self.box_lengths[2]

                # Compute distance
                r = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)

                # Check if new particle is below rejection radius. If not, break out and try again
                if r <= r_reject:
                    k += 1  # Increment Halton counter
                    flag = 0  # New position not added (0 -> no longer outside reject r)
                    break

            # If flag true add new positiion
            if flag == 1:
                # Add new positions to arrays
                x = np.append(x, x_new)
                y = np.append(y, y_new)
                z = np.append(z, z_new)

                k += 1  # Increment Halton counter
                i += 1  # Increment particle number

        self.pos[:, 0] = x
        self.pos[:, 1] = y
        self.pos[:, 2] = z

        # End timer
        end = time.time()
        print("Particles' positioned according to Halton method took: {:1.4e}".format(end - start))

    def kinetic_temperature(self, kB):

        K = np.zeros(self.num_species)
        T = np.zeros(self.num_species)
        const = 2.0 / (kB * self.species_num * self.dimensions)
        kinetic = 0.5 * self.masses * (self.vel * self.vel).transpose()

        species_start = 0
        species_end = 0
        for i, num in enumerate(self.species_num):
            species_end += num
            K[i] = np.sum(kinetic[:, species_start:species_end])
            T[i] = const[i] * K[i]
            species_start = species_end

        return K, T

    def remove_drift(self):
        """
        Enforce conservation of total linear momentum. Updates ``ptcls.vel``

        Parameters
        ----------
        vel: array
            Particles' velocities.

        nums: array
            Number of particles of each species.

        masses: array
            Mass of each species.

        """
        species_start = 0
        species_end = 0
        momentum = self.masses * self.vel.transpose()
        for ic, nums in enumerate(self.species_num):
            species_end += nums
            P = np.sum(momentum[:, species_start:species_end], axis=1)
            self.vel[species_start:species_end, :] -= P / (nums * self.masses[species_end - 1])
            species_start = species_end


class Species:
    """
    Particle Species.

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
            Species charge number.

        atomic_weight : float
            Species atomic weight.

        initial_velocity: array
            Initial velocity in x,y,z directions.
    """

    def __init__(self):
        self.name = None
        self.number_density = None
        self.charge = None
        self.mass = None
        self.num = None
        self.concentration = None
        self.mass_density = None
        self.load_method = None
        self.atomic_weight = None
        self.initial_velocity_distribution = 'boltzmann'
        self.initial_spatial_distribution = 'random_no_reject'
        self.Z = None
        self.initial_velocity = np.zeros(3)

    def calc_plasma_frequency(self, fourpie0):
        self.wp = np.sqrt(4.0 * np.pi * self.charge ** 2 * self.number_density / (self.mass * fourpie0) )

    def calc_debye_length(self, kB, fourpie0):
        self.debye_length = np.sqrt( (self.temperature * kB * fourpie0)
                                     / (4.0 * np.pi * self.charge ** 2 * self.number_density) )

    def calc_cyclotron_frequency(self, magnetic_field_strength):
        #  See https://en.wikipedia.org/wiki/Lorentz_force
        self.omega_c = self.charge * magnetic_field_strength / self.mass

    def calc_coupling(self, aws, z_avg, const):
        self.ai = (self.charge / z_avg) ** (1./3.) * aws
        self.coupling = self.charge ** 2 / (self.ai * const * self.temperature)