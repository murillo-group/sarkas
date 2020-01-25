'''
S_params.py

a code to read Sarkas input file with a YAML format.

species - species to use in the simulation
    name: species name
    mass: species mass
    charge: species charge = e*Z
    Z: degree of ionization
    Temperature: desired species temperature

load - particles loading described in Species
    species_name: should be one of names in Species
    num: number of particles of the species to load
    num_density: number density of the species
    method: particle loading method. Files, restart file mathemathical mehtods.

potential - Two body potential
    type: potential type. Yukawa, EGS are available
    Gamma: plasma prameter Gamma
    kappa: plasma parameter kappa

Thermostat - Thermostat to equilibrize the system
    type: Thermostat type. Only Berendsen for now.

Integrator
    type: Verlet only.

Langevin
    type: Langevin model type. Only BBK for now.

control - general setup values for the simulations
    dt: timestep
    Neq: number of steps to make the system equblizing
    Nstep: number of steps for data collecting
    BC: Boundary condition. periodic only
    units: cgs, mks
    dump_step: output step
'''

import yaml
import numpy as np
import sys

import scipy.constants as const
# force are here
import S_pot_Coulomb as Coulomb
import S_pot_Yukawa as Yukawa
import S_pot_LJ as LJ
import S_pot_EGS as EGS
import S_pot_Moliere as Moliere
import S_pot_QSP as QSP

class Params:
    def __init__(self):
        self.species = []
        self.load = []
        self.magnetized = False

    class Species:
        def __init__(self):
            pass

    class Magnetic:
        on = 1
        elec_therm = 0
        Neq_mag = 0

    class Potential:
        method = "PP"
#        def __init__(self):
#            pass

    class P3M:
        on = False

    class Integrator:
        def __init__(self):
            pass

    class Thermostat:
            on = 0

    class Langevin:
            on = 0

    class Control:
        units = None
        dt = None
        Nstep = None
        Neq = None
        BC = "periodic"
        dump_step = 1
        writexyz = "no"
        verbose = "yes"
        checkpoint_dir = "Checkpoint"

    def setup(self, filename):
        # setup general parameters for all potential types and units
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
        with open(filename, 'r') as stream:
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
                    self.magnetized = True
                    self.Magnetic.on = 1
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if (key == "B_Gauss"):
                                G2T = 1e-4
                                self.BField = float(value)*G2T
                            if (key == "BField"):
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
        self.neutrality = 0.0

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

                if (self.magnetized):
                    if (self.Control.units == "cgs"):
                        self.species[ic].omega_c = self.species[ic].charge*self.BField/self.species[ic].mass
                        self.species[ic].omega_c = self.species[ic].omega_c/const.physical_constants["speed of light in vacuum"][0]
                    elif (self.Control.units == "mks"):                 
                        self.species[ic].omega_c = self.species[ic].charge*self.BField/self.species[ic].mass

                # Q^2 factor see eq.(2.10) in Ballenegger et al. J Chem Phys 128 034109 (2008)
                self.species[ic].QFactor = self.species[ic].num*self.species[ic].charge**2
                
                self.QFactor += self.species[ic].QFactor
                self.neutrality += self.species[ic].charge*self.species[ic].num


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
    
