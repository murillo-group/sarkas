import yaml
import numpy as np
import sys
import scipy.constants as const
import S_pot_Coulomb as Coulomb
import S_pot_Yukawa as Yukawa
import S_pot_LJ as LJ
import S_pot_EGS as EGS
import S_pot_Moliere as Moliere
import S_pot_QSP as QSP

class Params:
    """
    Simulation's Parameters.
    
    Attributes
    ----------
        aws : float
            Wigner-Seitz radius. Calculated from the ``total_num_density`` .

        box_volume : float
            Box volume.

        d : int
            Number of non-zero dimensions.
        
        dq : float
            Minimum wavenumber defined as :math:`2\pi/L` .
        
        dt : int
            Timestep.

        dump_step : int
            Snapshot interval.

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
            Box length.

        Lx : float
            Box length in the :math:`x` direction.

        Ly : float
            Box length in the :math:`y` direction.

        Lz : float
            Box length in the :math:`z` direction.

        Lv : array, shape(3)
            Box length in each direction.

        Lmax_v : array, shape(3)
            Maximum box length in each direction.

        Lmin_v : array, shape(3)
            Minimum box length in each direction.
        
        N : int
            Total number of particles same as ``tot_num_ptcls``.
        
        Neq : int
            Total number of equilibration steps.

        Nq : int
            Number of wavenumbers.
        
        Nt : int
            Number of production time steps.
        
        P3M : class
            P3M algorithm's parameters.

        q_max : int
            Maximum wavenumber.
        
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
        
        units : str
            Choice of units mks or cgs.

        wq : float
            Total Plasma frequency.

        force : func
            Function for force calculation.
    """

    def __init__(self):
        # Container of Species
        self.species = []
        #
        self.load = []
        self.P3M = self.P3M()
        self.Magnetic = self.Magnetic()
        self.Potential = self.Potential()
        self.Integrator = self.Integrator()
        self.Control = self.Control()
        self.Thermostat = self.Thermostat()
        self.Langevin = self.Langevin()


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

        """

        def __init__(self):
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
        """
        def __init__(self):
            self.on = False
            self.elec_therm = True
            self.Neq_mag = 10000

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

    class Integrator:
        """
        Integrator's parameters.

        Attributes
        ----------
            type : str
                Integrator type. 
        """
        def __init__(self):
            pass

    class Thermostat:
        """
        Thermostat's parameters

        Attributes
        ----------
            on : int
                Flag. 1 = True, 0 = False.

            type : int
                Berendsen only.

            tau : float
                Berendsen parameter :math:`\tau`.

            timestep : int
                Number of timesteps to wait before turning on Berendsen.
                (default) = 0
        """
        def __init__(self):
            self.on = 1
            self.timestep = 0

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

            Nstep : int
                Number of simulation timesteps. Same as ``Params.Nt``.

            BS : str
                Boundary Condition. 'Periodic' only.

            dump_step : int
                Snapshot interval.

            writexyz : str
                Flag for XYZ file for OVITO. "no" or "yes".

            verbose : str
                Flag for verbose screen output.
            
            checkpoint_dir : str
                Directory to store simulation's output files.

            screen_output : bool
                Flag to print to screen. default = False.

            log_file : str
                File name for log output.
        """
        def __init__(self):
            self.units = None
            self.dt = None
            self.Nstep = None
            self.Neq = None
            self.BC = "periodic"
            self.dump_step = 1
            self.screen_output = False
            self.writexyz = "no"
            self.verbose = "yes"
            self.checkpoint_dir = "Checkpoint"
            self.log_file = self.checkpoint_dir + "/log.out"

    def setup(self, filename):
        """
        Setup simulations' parameters.

        Parameters
        ----------
        filename : str
            Input file's name.

        """

        self.total_num_ptcls = 0
        self.total_num_density = 0

        # Parse parameters from input file
        self.common_parser(filename)

        self.N = self.total_num_ptcls

        # Coulomb potential
        if (self.Potential.type == "Coulomb"):
            Coulomb.setup(self)

        # Yukawa potential
        if (self.Potential.type == "Yukawa"):
            Yukawa.setup(self,filename)
        
        # exact gradient-corrected screening (EGS) potential
        if (self.Potential.type == "EGS"):
            EGS.EGS_setup(self,filename)

        # Lennard-Jones potential
        if (self.Potential.type == "LJ"):
            LJ.LJ_setup(self,filename)

        # Moliere potential
        if (self.Potential.type == "Moliere"):
            Moliere.Moliere_setup(self,filename)

        # QSP potential
        if (self.Potential.type == "QSP"):
            QSP.setup(self)

        self.Potential.LL_on = 1       # linked list on
        if not hasattr(self.Potential, "rc"):
            print("The cut-off radius is not defined. L/2 = ", self.L/2, "will be used as rc")
            self.Potential.rc = self.L/2.
            self.Potential.LL_on = 0       # linked list off

        if (self.Potential.method == "PP" and self.Potential.rc > self.L/2.):
            print("The cut-off radius is > L/2. L/2 = ", self.L/2, "will be used as rc")
            self.Potential.rc = self.L/2.
            self.Potential.LL_on = 0       # linked list off

        self.T_desired = self.Ti

        return

    # read input data which does not depend on potential type. 
    def common_parser(self, filename):
        """
        Parse common parameters from input file

        Parameters
        ----------
        filename : str
            Input file's name.

        """
        with open( filename, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)

            for lkey in dics:
                if (lkey == "Particles"):
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():

                            if (key == "species"):
                                spec = self.Species()
                                self.species.append(spec)
                                ic = len(self.species) - 1

                                for key, value in value.items():
                                    if (key == "name"):
                                        self.species[ic].name = value

                                    if (key == "number_density"):
                                        self.species[ic].num_density = float(value)
                                        self.total_num_density += self.species[ic].num_density

                                    if (key == "mass"):
                                        self.species[ic].mass = float(value)

                                    if (key == "num"):
                                        self.species[ic].num = int(value)
                                        self.total_num_ptcls += self.species[ic].num

                                    if (key == "Z"):
                                        self.species[ic].Z = float(value)

                                    if (key == "temperature"):
                                        self.species[ic].temperature = float(value)

                                    if (key == "A"):
                                        self.species[ic].atomic_weight = float(value)

                                    if (key == "temperature_eV"):
                                        # Conversion factor from eV to Kelvin
                                        eV2K = const.physical_constants["electron volt-kelvin relationship"][0]
                                        self.species[ic].temperature = float(value)*eV2K

                            if (key == "load"):
                                for key, value in value.items():
                                    if (key == "method"):
                                        self.load_method = value

                                    if (key == "rand_seed"):
                                        self.load_rand_seed = int(value)

                                    if (key == 'restart_step'):
                                        self.load_restart_step = int(value)
                                        self.load_rand_seed = 1

                                    if (key == 'r_reject'):
                                        self.load_r_reject = float(value)

                                    if (key == 'perturb'):
                                        self.load_perturb = float(value)

                                    if (key == 'halton_bases'):
                                        self.load_halton_bases = np.array(value)

                if (lkey == "Potential"):
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if (key == "type"):
                                self.Potential.type = value

                            if (key == "method"):
                                self.Potential.method = value

                            if (key == "rc"):
                                self.Potential.rc = float(value)

                if (lkey == "P3M"):
                    self.P3M.on = True
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if (key == "MGrid"):
                                self.P3M.MGrid = np.array(value)
                                self.P3M.Mx = self.P3M.MGrid[0]
                                self.P3M.My = self.P3M.MGrid[1]
                                self.P3M.Mz = self.P3M.MGrid[2]
                            if (key == "cao"):
                                self.P3M.cao = int(value)
                            if (key == "aliases"):
                                self.P3M.aliases = np.array(value,dtype=int)
                                self.P3M.mx_max = self.P3M.aliases[0]
                                self.P3M.my_max = self.P3M.aliases[1]
                                self.P3M.mz_max = self.P3M.aliases[2]
                            if (key == "alpha_ewald"):
                                self.P3M.G_ew = float(value)
                                
                if (lkey == "Thermostat"):
                    self.Thermostat.on = 1
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if (key == 'type'):
                                self.Thermostat.type = value
                            # If Berendsen
                            if (key == 'tau'):
                                # Notice tau_t should be a number between 0 (not included) and 1
                                if ( float(value) > 0.0) :
                                    self.Thermostat.tau = float(value)
                                else :
                                    print("\nBerendsen tau parameter must be positive")
                                    sys.exit()

                            if (key == 'timestep'):
                                # Number of timesteps to wait before turning on Berendsen
                                self.Thermostat.timestep = int(value)
                
                if (lkey == "Magnetized"):
                    self.Magnetic.on = True
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if (key == "B_Gauss"):
                                G2T = 1e-4
                                self.BField = float(value)*G2T

                            if (key == "B_Tesla"):
                                self.BField = float(value)

                            if (key == "electrostatic_thermalization"):
                                # 1 = true, 0 = false
                                self.Magnetic.elec_therm = int(value)

                            if (key == "Neq_mag"):
                                # Number of equilibration of magnetic degrees of freedom
                                self.Magnetic.Neq_mag = int(value)

                if (lkey == "Integrator"):
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if (key == 'type'):
                                self.Integrator.type = value

                if (lkey == "Langevin"):
                    self.Langevin.on = 1
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if (key == 'type'):
                                self.Langevin.type = value
                            if (key == 'gamma'):
                                self.Langevin.gamma = float(value)

                if (lkey == "Control"):
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            # Units
                            if (key == "units"):
                                self.Control.units = value
                                self.units = self.Control.units

                            # timestep
                            if (key == "dt"):
                                self.Control.dt = float(value)
                                self.dt = self.Control.dt

                            # Number of simulation timesteps    
                            if (key == "Nstep"):
                                self.Control.Nstep = int(value)
                                self.Control.Nt = self.Control.Nstep
                                self.Nt = self.Control.Nstep

                            # Number of equilibration timesteps
                            if (key == "Neq"):
                                self.Control.Neq = int(value)
                                self.Neq = self.Control.Neq

                            # Periodic Boundary Condition
                            if (key == "BC"):
                                self.Control.BC = value
                                if (self.Control.BC == "periodic"):
                                    self.Control.PBC = 1
                                else:
                                    self.Control.PBC = 0
                            # Saving interval
                            if (key == "dump_step"):
                                self.Control.dump_step = int(value)
                                self.dump_step = self.Control.dump_step
                            
                            # Write the XYZ file, Yes/No
                            if (key == "writexyz"):
                                if (value is False):
                                    self.Control.writexyz = 0
                                if (value is True):
                                    self.Control.writexyz = 1
                            
                            # verbose screen print out
                            if (key == "verbose"):
                                if (value is False):
                                    self.Control.verbose = 0
                                if (value is True):
                                    self.Control.verbose = 1
                            
                            # Directory where to store Checkpoint files
                            if (key =="output_dir"):
                                self.Control.checkpoint_dir = value
                            
                            if (key == "screen_output" ):
                                self.Control.screen_output = True

                            # Filenames appendix 
                            if (key =="fname_app"):
                                self.Control.fname_app = value
                            else:
                                self.Control.fname_app = self.Control.checkpoint_dir

        self.num_species = len(self.species)
        # Physical constants
        if (self.Control.units == "cgs"):
            self.J2erg = 1.0e+7  # erg/J
            self.kB = const.Boltzmann*self.J2erg
            self.eV2K = const.physical_constants["electron volt-kelvin relationship"][0]
            self.c0 = const.physical_constants["speed of light in vacuum"][0]
            if not (self.Potential.type=="LJ"):
                # Coulomb to statCoulomb conversion factor. See https://en.wikipedia.org/wiki/Statcoulomb
                C2statC = 1.0/(10.0*self.c0)
                self.hbar = self.J2erg*const.hbar
                self.hbar2 = self.hbar**2
                self.qe = C2statC*const.physical_constants["elementary charge"][0]
                self.me = 1.0e3*const.physical_constants["electron mass"][0] # grams   
                self.eps0 = 1.0

        elif (self.Control.units == "mks"):
            self.kB = const.Boltzmann
            self.eV2K = const.physical_constants["electron volt-kelvin relationship"][0]
            self.c0 = const.physical_constants["speed of light in vacuum"][0]
            if not (self.Potential.type=="LJ"):
                # Coulomb to statCoulomb conversion factor. See https://en.wikipedia.org/wiki/Statcoulomb
                self.hbar = const.hbar
                self.hbar2 = self.hbar**2
                self.qe = const.physical_constants["elementary charge"][0]
                self.me = const.physical_constants["electron mass"][0]
                self.eps0 = const.epsilon_0

        # Charge systems' parameters.
        self.QFactor = 0.0
        self.tot_net_charge = 0.0

        # Check mass input
        for ic in range(self.num_species):
            if hasattr (self.species[ic], "atomic_weight"):
                # Choose between atomic mass constant or proton mass
                # u = const.physical_constants["atomic mass constant"][0]
                mp = const.physical_constants["proton mass"][0]

                if (self.Control.units == "cgs"):
                    self.species[ic].mass = mp*1e3*self.species[ic].atomic_weight
                elif (self.Control.units == "mks"):
                    self.species[ic].mass = mp*self.species[ic].atomic_weight

        # Concentrations arrays and ions' total temperature
        nT = 0.
        for ic in range( self.num_species):
            self.species[ic].concentration = self.species[ic].num/self.total_num_ptcls
            nT += self.species[ic].concentration*self.species[ic].temperature

        self.Ti = nT

        # Wigner-Seitz radius calculated from the total density
        self.aws = (3.0/(4.0*np.pi*self.total_num_density))**(1./3.)

        if not (self.Potential.type == "LJ"):
            for ic in range(self.num_species):

                if (self.Control.units == "cgs"):
                    C2statC = 1.0/(10.0*const.physical_constants["speed of light in vacuum"][0])
                    qe = C2statC*const.physical_constants["elementary charge"][0]
                elif (self.Control.units == "mks"):
                    qe = const.physical_constants["elementary charge"][0]

                self.species[ic].charge = qe
                

                if hasattr(self.species[ic], "Z"):
                    self.species[ic].charge *= self.species[ic].Z

                if (self.Magnetic.on):
                    if (self.Control.units == "cgs"):
                        self.species[ic].omega_c = self.species[ic].charge*self.BField/self.species[ic].mass
                        self.species[ic].omega_c = self.species[ic].omega_c/const.physical_constants["speed of light in vacuum"][0]
                    elif (self.Control.units == "mks"):                 
                        self.species[ic].omega_c = self.species[ic].charge*self.BField/self.species[ic].mass

                # Q^2 factor see eq.(2.10) in Ballenegger et al. J Chem Phys 128 034109 (2008)
                self.species[ic].QFactor = self.species[ic].num*self.species[ic].charge**2
                
                self.QFactor += self.species[ic].QFactor
                self.tot_net_charge += self.species[ic].charge*self.species[ic].num


        # Calculate electron number density from the charge neutrality condition in case of Yukawa or EGS potential
        if (self.Potential.type == "Yukawa" or self.Potential.type == "EGS"):
            self.ne = 0 
            for ic in range(self.num_species):
                if hasattr(self.species[ic], "Z"):
                    self.ne += self.species[ic].Z*self.species[ic].num_density
        
        # Simulation Box Parameters
        self.L = self.aws*(4.0*np.pi*self.total_num_ptcls/3.0)**(1.0/3.0)      # box length
        self.N = self.total_num_ptcls
        L = self.L
        self.Lx = L
        self.Ly = L
        self.Lz = L
        self.Lv = np.array([L, L, L])              # box length vector
        
        self.box_volume = self.Lx*self.Ly*self.Lz

        self.d = np.count_nonzero(self.Lv)              # no. of dimensions
        self.Lmax_v = np.array([L, L, L])
        self.Lmin_v = np.array([0.0, 0.0, 0.0])

        # lowest wavenumber for S(q) and S(q,w)
        self.dq = 2.0*np.pi
        if (self.L > 0.):
            self.dq = 2.*np.pi/self.L
        # Max wavenumber for S(q) and S(q,w)
        self.q_max = 30                   # hardcode
        if (self.aws > 0):
            self.q_max = 30.0/self.aws       # hardcode, wave vector
        self.Nq = 3.0*int(self.q_max/self.dq)

        return
    
