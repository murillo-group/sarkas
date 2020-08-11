"""
Module for handling Params class
"""
import yaml
import numpy as np
import os.path
import sys
import scipy.constants as const

from sarkas.time_evolution.integrators import Integrator
from sarkas.time_evolution.thermostats import Thermostat


class Params:
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
        self.L = 0.0
        self.Lv = []
        self.input_file = None
        self.box_volume = 0.0
        self.dimensions = 3
        self.J2erg = 1.0e+7  # erg/J
        self.eps0 = const.epsilon_0
        self.fourpie0 = 4.0 * np.pi * self.eps0
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
        self.Ti = 0.0
        self.T_desired = 0.0
        self.total_num_density = 0
        self.total_num_ptcls = 0
        self.species = []
        #
        self.pppm = self.P3M()
        self.BC = self.BC()
        self.Magnetic = self.Magnetic()
        self.potential = self.Potential()
        self.integrator = Integrator()
        self.control = self.Control()
        self.thermostat = Thermostat()
        self.Langevin = self.Langevin()
        self.PostProcessing = self.PostProcessing()

    class BC:
        """Boundary Conditions.

        Attributes
        ----------
        pbc_axes : list
            Axes with Periodic Boundary Conditions.

        mm_axes : list
            Axes with Momentum Mirror Conditions.

        open_axes: list
            Axes with Open Boundary Conditions.

        pbc_axes_indx : array
            Indexes of axes with Periodic Boundary Conditions.

        mm_axes_indx : array
            Indexes of axes with Momentum Mirror Conditions.

        open_axes_indx: array
            Indexes of axes with Open Boundary Conditions.
        """

        def __init__(self):
            self.pbc_axes = []
            self.mm_axes = []
            self.open_axes = []
            self.pbc_axes_indx = []
            self.mm_axes_indx = []
            self.open_axes_indx = []

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

            init_vel: array
                Initial velocity in x,y,z directions.
        """

        def __init__(self):
            self.name = None
            self.number_density = None
            self.charge = None
            self.mass = None
            self.num = None
            self.load_method = None
            pass

    class Magnetic:
        """ 
        Parameters for a plasma in a constant magnetic field

        Attributes
        ----------
        on : bool
            Flag for magnetized plasma.

        elec_therm : int
            Thermalize electrostatic forces first?
            0 = False, 1 = True (default).

        Neq_mag : int
            Number of equilibration steps with magnetic field on.

        Bfield : float
            Strength of Magnetic Field.
        """

        def __init__(self):
            self.on = False
            self.elec_therm = True
            self.Neq_mag = 10000
            self.BField = 0.0

    class Potential:
        """
        Parameters specific to potential choice.

        Attributes
        ----------
        matrix : array
            Potential's parameters.

        method : str
            Algorithm to use for force calculations.
            "PP" = Linked Cell List (default).
            "P3M" = Particle-Particle Particle-Mesh.

        rc : float
            Cutoff radius.

        type : str
            Interaction potential: LJ, Yukawa, EGS, Coulomb, QSP, Moliere.

        """

        def __init__(self):
            self.method = "PP"

    class P3M:
        """ 
        P3M Algorithm parameters.

        Attributes
        ----------
        aliases : array, shape(3)
            Number of aliases in each direction.

        cao : int
            Charge assignment order.

        on : bool
            Flag.

        MGrid : array, shape(3), int
            Number of mesh point in each direction.

        Mx : int
            Number of mesh point along the :math:`x` axis.

        My : int
            Number of mesh point along the :math:`y` axis.

        Mz : int
            Number of mesh point along the :math:`z` axis.

        mx_max : int
            Number of aliases along the reciprocal :math:`x` direction.

        my_max : int
            Number of aliases along the reciprocal :math:`y` direction.

        mz_max : int
            Number of aliases along the reciprocal :math:`z` direction.

        G_ew : float
            Ewald parameter.

        G_k : array
            Optimized Green's function.

        hx : float
            Mesh spacing in :math:`x` direction.

        hy : float
            Mesh spacing in :math:`y` direction.

        hz : float
            Mesh spacing in :math:`z` direction.

        PP_err : float
            Force error due to short range cutoff.

        PM_err : float
            Force error due to long range cutoff.

        F_err : float
            Total force error.

        kx_v : array
            Array of :math:`k_x` values.

        ky_v : array
            Array of :math:`k_y` values.

        kz_v : array
            Array of :math:`k_z` values.

        """

        def __init__(self):
            self.on = False

    class Langevin:
        """
        Parameters for Langevin Dynamics.

        Attributes
        ----------
            on : int
                Flag. 0 = False, 1 = True.

            type : str
                Type of Langevin. 

            gamma : float
                Langeving gamma.
        """

        def __init__(self):
            self.on = 0

    class Control:
        """
        General simulation parameters

        Attributes
        ----------
            units : str
                cgs or mks.

            dt : float
                timestep. Same as ``Params.dt``.

            Nsteps : int
                Number of simulation timesteps.

            BC : str
                Boundary Condition. 'Periodic' only.

            dump_step : int
                Production Snapshot interval.

            therm_dump_step : int
                Thermalization Snapshot interval.

            np_per_side : array
                Number of particles per box length. Default= :math: `N_{tot}^{1/3}`
                Note that :math: `N_x x N_y x N_z = N_{tot}`

            writexyz : str
                Flag for XYZ file for OVITO. Default = False.

            verbose : str
                Flag for verbose screen output.
            
            checkpoint_dir : str
                Path to the directory where the outputs of the current simulation will be stored.
                Default = "Simulations/UnNamedRun"

            log_file : str
                File name for log output. Default = log.out

            pre_run : bool
                Flag for initial estimation of simulation parameters.

            simulations_dir : str
                Path to the directory where all future simulations will be stored. Default = cwd +  "Simulations"

            dump_dir : str
                Path to the directory where simulations' dumps will be stored.
                Default = "Simulations/UnNamedRun/Production"

            therm_dir : str
                Path to the directory where simulations' dumps will be stored.
                Default = "Simulations/UnNamedRun/Thermalization"

        """

        def __init__(self):
            self.measure = False
            self.writexyz = False
            self.verbose = True
            self.simulations_dir = "Simulations"
            self.job_dir = None
            self.production_dir = 'Production'
            self.equilibration_dir = 'Equilibration'
            self.preprocessing_dir = "PreProcessing"
            self.postprocessing_dir = "PostProcessing"
            self.job_id = None
            self.log_file = None
            self.np_per_side = []
            self.pre_run = False

    class PostProcessing:

        def __init__(self):
            self.rdf_nbins = 100
            self.ssf_no_ka_values = np.array([5, 5, 5], dtype=int)
            self.dsf_no_ka_values = np.array([5, 5, 5], dtype=int)
            self.hermite_order = 10

    def setup(self, args):
        """
        Setup simulations' parameters.

        Parameters
        ----------
        args : dict
            Input arguments
        """
        self.input_file = args["input_file"]

        # Parse parameters from input file
        self.common_parser(self.input_file)
        self.assign_attributes()
        self.create_directories(args)

        # Coulomb potential
        if self.potential.type == "Coulomb":
            from sarkas.potentials import coulomb
            coulomb.setup(self)

        # Yukawa potential
        if self.potential.type == "Yukawa":
            from sarkas.potentials import yukawa
            yukawa.setup(self)

        # exact gradient-corrected screening (EGS) potential
        if self.potential.type == "EGS":
            from sarkas.potentials import egs
            egs.setup(self)

        # Lennard-Jones potential
        if self.potential.type == "LJ":
            from sarkas.potentials import lennardjones612 as lj
            lj.setup(self)

        # Moliere potential
        if self.potential.type == "Moliere":
            from sarkas.potentials import moliere
            moliere.setup(self)

        # QSP potential
        if self.potential.type == "QSP":
            from sarkas.potentials import qsp
            qsp.setup(self)

        return

    def common_parser(self, filename):
        """
        Parse common parameters from input file

        Parameters
        ----------
        filename : str
            Input file's name.

        """
        self.input_file = filename

        with open(filename, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)
            for lkey in dics:
                if lkey == "Particles":
                    for species in dics["Particles"]:
                        spec = self.Species()
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

                if lkey == "P3M":
                    self.pppm.on = True
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if key == "MGrid":
                                self.pppm.MGrid = np.array(value)
                                self.pppm.Mx = self.pppm.MGrid[0]
                                self.pppm.My = self.pppm.MGrid[1]
                                self.pppm.Mz = self.pppm.MGrid[2]
                            if key == "cao":
                                self.pppm.cao = int(value)
                            if key == "aliases":
                                self.pppm.aliases = np.array(value, dtype=int)
                                self.pppm.mx_max = self.pppm.aliases[0]
                                self.pppm.my_max = self.pppm.aliases[1]
                                self.pppm.mz_max = self.pppm.aliases[2]
                            if key == "alpha_ewald":
                                self.pppm.G_ew = float(value)

                if lkey == "Thermostat":
                    for key, value in dics[lkey].items():
                        if hasattr(self.thermostat, key):
                            self.thermostat.__dict__[key] = value
                        else:
                            setattr(self.thermostat, key, value)

                if lkey == "Magnetized":
                    self.Magnetic.on = True
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if key == "B_Gauss":
                                G2T = 1e-4
                                self.Magnetic.BField = float(value) * G2T

                            if key == "B_Tesla":
                                self.Magnetic.BField = float(value)

                            if key == "electrostatic_thermalization":
                                # 1 = true, 0 = false
                                self.Magnetic.elec_therm = value

                            if key == "Neq_mag":
                                # Number of equilibration of magnetic degrees of freedom
                                self.Magnetic.Neq_mag = int(value)

                if lkey == "Integrator":
                    for key, value in dics[lkey].items():
                        if hasattr(self.integrator, key):
                            self.integrator.__dict__[key] = value
                        else:
                            setattr(self.integrator, key, value)

                if lkey == "Langevin":
                    self.Langevin.on = 1
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if key == 'type':
                                self.Langevin.type = value
                            if key == 'gamma':
                                self.Langevin.gamma = float(value)

                if lkey == "PostProcessing":
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if key == 'rdf_nbins':
                                self.PostProcessing.rdf_nbins = int(value)
                            if key == 'ssf_no_ka_values':
                                self.PostProcessing.ssf_no_ka_values = value
                            if key == 'dsf_no_ka_values':
                                self.PostProcessing.dsf_no_ka_values = value

                if lkey == "BoundaryConditions":
                    for key, value in dics[lkey].items():
                            if key == "periodic":
                                self.BC.pbc_axes = value

                            if key == "momentum_mirror":
                                self.BC.mm_axes = value

                            if key == "open":
                                self.BC.open_axes = value

                if lkey == "Control":
                    for key, value in dics[lkey].items():
                        if hasattr(self.control, key):
                            self.control.__dict__[key] = value
                        else:
                            setattr(self.control, key, value)

        # Check for conflicts in case of magnetic field
        if self.Magnetic.on and self.Magnetic.elec_therm:
            self.integrator.mag_type = value
            self.integrator.type = 'Verlet'

        # Check for thermostat temperatures
        if not hasattr(self.thermostat, 'temperatures'):
            self.thermostat.temperatures = np.zeros(len(self.species))
            for i, sp in enumerate(self.species):
                self.thermostat.temperatures[i] = sp.temperature

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
            if hasattr(self.control, key):
                self.control.__dict__[key] = value

        # Check if the directories exist
        if not os.path.exists(self.control.simulations_dir):
            os.mkdir(self.control.simulations_dir)

        self.control.job_dir = os.path.join(self.control.simulations_dir, self.control.job_dir)
        if not os.path.exists(self.control.job_dir):
            os.mkdir(self.control.job_dir)

        # Equilibration directory and sub_dir
        self.control.equilibration_dir = os.path.join(self.control.job_dir, self.control.equilibration_dir)
        if not os.path.exists(self.control.equilibration_dir):
            os.mkdir(self.control.equilibration_dir)

        self.control.eq_dump_dir = os.path.join(self.control.equilibration_dir, 'dumps')
        if not os.path.exists(self.control.eq_dump_dir):
            os.mkdir(self.control.eq_dump_dir)

        # Production dir and sub_dir
        self.control.production_dir = os.path.join(self.control.job_dir, self.control.production_dir)
        if not os.path.exists(self.control.production_dir):
            os.mkdir(self.control.production_dir)

        self.control.prod_dump_dir = os.path.join(self.control.production_dir, "dumps")
        if not os.path.exists(self.control.prod_dump_dir):
            os.mkdir(self.control.prod_dump_dir)

        # Postprocessing dir
        self.control.postprocessing_dir = os.path.join(self.control.job_dir, self.control.postprocessing_dir)
        if not os.path.exists(self.control.postprocessing_dir):
            os.mkdir(self.control.postprocessing_dir)

        if self.control.log_file is None:
            self.control.log_file = os.path.join(self.control.job_dir, 'log.out')

    def assign_attributes(self):
        """ Assign the parsed parameters"""
        # Physical constants
        if self.control.units == "cgs":
            self.kB *= self.J2erg
            self.c0 *= 1e2  # cm/s
            if not (self.potential.type == "LJ"):
                # Coulomb to statCoulomb conversion factor. See https://en.wikipedia.org/wiki/Statcoulomb
                C2statC = 1.0e-01 * self.c0
                self.hbar = self.J2erg * self.hbar
                self.hbar2 = self.hbar ** 2
                self.qe *= C2statC
                self.me *= 1.0e3
                self.eps0 = 1.0
                self.fourpie0 = 1.0
                self.a0 *= 1e2
        if self.control.job_id is None:
            self.control.job_id = self.control.job_dir
        self.num_species = len(self.species)
        # Loop over species and assign missing attributes
        for i, sp in enumerate(self.species):
            self.total_num_ptcls += sp.num
            if hasattr(sp, "number_density"):
                self.total_num_density += sp.number_density

            if hasattr(sp, "atomic_weight"):
                # Choose between atomic mass constant or proton mass
                # u = const.physical_constants["atomic mass constant"][0]
                self.mp = const.physical_constants["proton mass"][0]

                if self.control.units == "cgs":
                    self.mp *= 1e3
                    sp.mass = self.mp * sp.atomic_weight
                elif self.control.units == "mks":
                    sp.mass = self.mp * sp.atomic_weight

            if hasattr(sp, "mass_density"):
                Av = const.physical_constants["Avogadro constant"][0]
                sp.number_density = sp.mass_density * Av / sp.atomic_weight
                self.total_num_density += sp.number_density
        # Concentrations arrays and ions' total temperature
        self.Ti = 0.0
        for i, sp in enumerate(self.species):
            sp.concentration = sp.num / self.total_num_ptcls
            self.Ti += sp.concentration * sp.temperature

        # Wigner-Seitz radius calculated from the total density
        self.aws = (3.0 / (4.0 * np.pi * self.total_num_density)) ** (1. / 3.)

        if not (self.potential.type == "LJ"):
            for ic in range(self.num_species):

                self.species[ic].charge = self.qe

                if hasattr(self.species[ic], "Z"):
                    self.species[ic].charge *= self.species[ic].Z

                if self.Magnetic.on:
                    if self.control.units == "cgs":
                        #  See https://en.wikipedia.org/wiki/Lorentz_force
                        self.species[ic].omega_c = self.species[ic].charge * self.Magnetic.BField / self.species[
                            ic].mass
                        self.species[ic].omega_c = self.species[ic].omega_c / self.c0
                    elif self.control.units == "mks":
                        self.species[ic].omega_c = self.species[ic].charge * self.Magnetic.BField / self.species[
                            ic].mass

                # Q^2 factor see eq.(2.10) in Ballenegger et al. J Chem Phys 128 034109 (2008)
                self.species[ic].QFactor = self.species[ic].num * self.species[ic].charge ** 2

                self.QFactor += self.species[ic].QFactor
                self.tot_net_charge += self.species[ic].charge * self.species[ic].num

        # Calculate electron number density from the charge neutrality condition in case of Yukawa or EGS potential
        if self.potential.type == "Yukawa" or self.potential.type == "EGS":
            for ic in range(self.num_species):
                if hasattr(self.species[ic], "Z"):
                    self.ne += self.species[ic].Z * self.species[ic].number_density

        # Simulation Box Parameters
        if len(self.control.np_per_side) != 0:
            msg = "Number of particles per dimension does not match total number of particles."
            assert int(np.prod(self.control.np_per_side)) == self.total_num_ptcls, msg

            self.Lx = self.aws * self.control.np_per_side[0] * (4.0 * np.pi / 3.0) ** (1.0 / 3.0)
            self.Ly = self.aws * self.control.np_per_side[1] * (4.0 * np.pi / 3.0) ** (1.0 / 3.0)
            self.Lz = self.aws * self.control.np_per_side[2] * (4.0 * np.pi / 3.0) ** (1.0 / 3.0)
        else:
            self.Lx = self.aws * (4.0 * np.pi * self.total_num_ptcls / 3.0) ** (1.0 / 3.0)
            self.Ly = self.aws * (4.0 * np.pi * self.total_num_ptcls / 3.0) ** (1.0 / 3.0)
            self.Lz = self.aws * (4.0 * np.pi * self.total_num_ptcls / 3.0) ** (1.0 / 3.0)

        self.Lv = np.array([self.Lx, self.Ly, self.Lz])  # box length vector

        # Dev Note: The following are useful for future geometries
        self.e1 = np.array([self.Lx, 0.0, 0.0])
        self.e2 = np.array([0.0, self.Ly, 0.0])
        self.e3 = np.array([0.0, 0.0, self.Lz])

        self.box_volume = abs(np.dot(np.cross(self.e1, self.e2), self.e3))

        self.dimensions = np.count_nonzero(self.Lv)  # no. of dimensions

        self.T_desired = self.Ti

        # boundary Conditions
        if self.BC.pbc_axes:
            self.BC.pbc_axes_indx = np.zeros(len(self.BC.pbc_axes))
            for (ij, bc) in enumerate(self.BC.pbc_axes):
                if bc == "x":
                    self.BC.pbc_axes_indx[ij] = 0
                elif bc == "y":
                    self.BC.pbc_axes_indx[ij] = 1
                elif bc == "z":
                    self.BC.pbc_axes_indx[ij] = 2

        if self.BC.mm_axes:
            print("\nOnly Periodic Boundary Conditions are supported. Bye!")
            sys.exit()
            self.BC.mm_axes_indx = np.zeros(len(self.BC.mm_axes), dtype=np.int)
            for (ij, bc) in enumerate(self.BC.mm_axes):
                if bc == "x":
                    self.BC.mm_axes_indx[ij] = 0
                elif bc == "y":
                    self.BC.mm_axes_indx[ij] = 1
                elif bc == "z":
                    self.BC.mm_axes_indx[ij] = 2

        if self.BC.open_axes:
            self.BC.open_axes_indx = np.zeros(len(self.BC.open_axes), dtype=np.int)
            for (ij, bc) in enumerate(self.BC.open_axes):
                if bc == "x":
                    self.BC.open_axes_indx[ij] = 0
                elif bc == "y":
                    self.BC.open_axes_indx[ij] = 1
                elif bc == "z":
                    self.BC.open_axes_indx[ij] = 2
