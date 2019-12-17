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
import fdint

import S_units as units
import S_constants as const  # empty. 
# force are here
import S_pot_Yukawa as Yukawa
import S_pot_LJ as LJ

class Params:
    def __init__(self):
        self.species = []
        self.load = []

    class Species:
        def __init__(self):
            pass

    class Potential:
        method = "PP"
#        def __init__(self):
#            pass

    class P3M:
        def __init__(self):
            pass

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

        self.common_parser(filename)
        self.N = self.total_num_ptcls
        # Yukawa potential
        if (self.Potential.type == "Yukawa"):
            self.Yukawa_setup(filename)
        
        # exact gradient-corrected screening (EGS) potential
        if (self.Potential.type == "EGS"):
            print("Not yet prepared!")
            sys.exit()

        # Lennard-Jones potential
        if (self.Potential.type == "LJ"):
            self.LJ_setup(filename)

        self.Potential.LL_on = 1       # linked list on
        if not hasattr(self.Potential, "rc"):
            print("The cut-off radius is not defined. L/2 = ", self.L/2, "will be used as rc")
            self.Potential.rc = self.L/2.
            self.Potential.LL_on = 0       # linked list off

        if (self.Potential.rc > self.L/2.):
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

                if (lkey == "Thermostat"):
                    self.Thermostat.on = 1
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if (key == 'type'):
                                self.Thermostat.type = value

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
                            if (key == "units"):
                                self.Control.units = value
                                self.units = self.Control.units
                                units.setup(self)

                            if (key == "dt"):
                                self.Control.dt = float(value)
                                self.dt = self.Control.dt

                            if (key == "Nstep"):
                                self.Control.Nstep = int(value)
                                self.Control.Nt = self.Control.Nstep
                                self.Nt = self.Control.Nstep

                            if (key == "Neq"):
                                self.Control.Neq = int(value)
                                self.Neq = self.Control.Neq

                            if (key == "BC"):
                                self.Control.BC = value
                                if (self.Control.BC == "periodic"):
                                    self.Control.PBC = 1

                            if (key == "dump_step"):
                                self.Control.dump_step = int(value)
                                self.dump_step = self.Control.dump_step

                            if (key == "writexyz"):
                                if (value is False):
                                    self.Control.writexyz = 0
                                if (value is True):
                                    self.Control.writexyz = 1

                            if (key == "verbose"):
                                if (value is False):
                                    self.Control.verbose = 0
                                if (value is True):
                                    self.Control.verbose = 1

                            if (key =="output_dir"):
                                self.Control.checkpoint_dir = value
                                    

        self.num_species = len(self.species)
        for ic in range(self.num_species):
            self.species[ic].charge = const.elementary_charge
            if hasattr(self.species[ic], "Z"):
                self.species[ic].charge = const.elementary_charge*self.species[ic].Z

        self.ai = 0.0
        if (self.total_num_density > 0.):
            self.ai = (3/(4*np.pi*self.total_num_density))**(1./3.)

        self.ne = 0 # number of electron
        for ic in range(self.num_species):
            if hasattr(self.species[ic], "Z"):

                self.ne += self.species[ic].Z*self.species[ic].num_density
        nT = 0.
        for i in range(self.num_species):
            nT += self.species[i].num*self.species[i].temperature

        self.Ti = nT/self.total_num_ptcls


        self.L = self.ai*(4.0*np.pi*self.total_num_ptcls/3.0)**(1.0/3.0)      # box length
        self.N = self.total_num_ptcls
        L = self.L
        self.Lx = L
        self.Ly = L
        self.Lz = L
        self.Lv = np.array([L, L, L])              # box length vector
        self.d = np.count_nonzero(self.Lv)              # no. of dimensions
        self.Lmax_v = np.array([L, L, L])
        self.Lmin_v = np.array([0.0, 0.0, 0.0])

        self.dq = 2*np.pi
        if (self.L > 0.):
            self.dq = 2.*np.pi/self.L

        self.q_max = 30                   # hardcode
        if (self.ai > 0):
            self.q_max = 30/self.ai       # hardcode, wave vector
        self.Nq = 3*int(self.q_max/self.dq)

#        self.rc = self.ai*params.Potential.rc

        return
    
    # Yukawa potential
    def Yukawa_setup(self, filename):
        # Yukawa_matrix[0,0,0] : kappa,
        # Yukawa_matrix[1,0,0] : Gamma,
        # Yukawa_matrix[2,:,:] : ij matrix for foce & potential calc.
        Yukawa_matrix = np.zeros((3, self.num_species, self.num_species)) 
        # open the input file to read Yukawa parameters

        ai = (3./(4*np.pi*ni))**(1./3)

        with open(filename, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)
            for lkey in dics:
                if (lkey == "Potential"):
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if (key == "kappa"):
                                self.Potential.kappa = float(value)

                            if (key == "Gamma"):
                                self.Potential.Gamma = float(value)

                            if (key == "elec_temperature"):
                                self.Te = float(value)

        # if kappa is not given calculate it from the electron temperature
        if hasattr(self, "kappa"):
            Yukawa_matrix[0,:,:] = self.Potential.kappa/ai
        else:
            if not hasattr(self, "Te"):
                print("Electron temperature is not defined. 1st species temperature ", self.species[0].temperature, \
                        "will be used as the electron temperature.")
                self.Te = self.species[0].temperature

            # Using MKS relation to obtain kappa and Gamma
            if (self.Control.units == "cgs"):
                units.mks_units()

            k = const.kb
            e = const.elementary_charge
            hbar = const.hbar
            m_e = const.elec_mass
            e_0 = const.epsilon_0

            if (self.Control.units == "cgs"):
                units.cgs_units() # back to the input nits.

            Te  = self.Te
            Ti = self.Ti

            if (self.Control.units == "cgs"):
                ne = self.ne*1.e6    # /cm^3 --> /m^3
                ni = self.total_num_density*1.e6

            if (self.Control.units == "mks"):
                ne = self.ne    # /cm^3 --> /m^3
                ni = self.total_num_density
            fdint_fdk_vec = np.vectorize(fdint.fdk)
            fdint_dfdk_vec = np.vectorize(fdint.dfdk)
            fdint_ifd1h_vec = np.vectorize(fdint.ifd1h)
            beta = 1./(k*Te)

            eta = fdint_ifd1h_vec(np.pi**2*(beta*hbar**2/(m_e))**(3/2)/np.sqrt(2)*ne) #eq 4 inverted

            lambda_TF = np.sqrt((4*np.pi**2*e_0*hbar**2)/(m_e*e**2)*np.sqrt(2*beta*hbar**2/m_e)/(4*fdint_fdk_vec(k=-0.5, phi=eta))) 

            Yukawa_matrix[0, :, :] = 1./self.lambda_TF # kappa/ai

        for i in range(self.num_species):
            Zi = self.species[i].Z
            for j in range(self.num_species):
                Zj = self.species[j].Z
                Yukawa_matrix[1, i, j] = (Zi*Zj)*e**2/(4*np.pi*e_0*ai*k*Ti) # Gamma

                if (self.Control.units == "cgs"):
                    Yukawa_matrix[2, i, j] = (Zi*Zj)*const.elementary_charge**2

                if (self.Control.units == "mks"):
                    Yukawa_matrix[2, i, j] = (Zi*Zj)*const.elementary_charge**2/(4*np.pi*const.epsilon_0)
            
        # Calculate the plasma frequency
        if (self.Control.units == "cgs"):
            self.lambda_TF = lambda_TF*100  # meter to centimeter
            wp = np.sqrt(4*np.pi*self.species[0].charge**2*self.total_num_density/self.species[0].mass)
            self.wp = wp

        elif (self.Control.units == "mks"):
            self.lambda_TF = lambda_TF
            wp = np.sqrt(self.species[0].charge**2*self.total_num_density/(self.species[0].mass*const.epsilon_0))
            self.wp = wp

        if (self.Potential.method == "PP"):
            self.force = Yukawa.Yukawa_force_PP

        if (self.Potential.method == "P3M"):
            self.force = Yukawa.Yukawa_force_P3M
            # P3M parameters
            self.P3M.Mx = 64     # hardcode
            self.P3M.My = 64     # hardcode
            self.P3M.Mz = 64     # hardcode
            self.P3M.hx = self.Lx/self.P3M.Mx
            self.P3M.hy = self.Ly/self.P3M.My
            self.P3M.hz = self.Lz/self.P3M.Mz
            self.P3M.p = 6       # hardcode
            self.P3M.mx_max = 3  # hardcode
            self.P3M.my_max = 3  # hardcode
            self.P3M.mz_max = 3  # hardcode
            # Ewald parameters
            self.P3M.G = 0.46/self.ai #hardcode
            self.P3M.G_ew = self.P3M.G

            # Optimized Green's Function
            self.P3M.G_k, self.P3M.kx_v, self.P3M.ky_v, self.P3M.kz_v, self.P3M.A_pm = Yukawa.gf_opt(self)

        self.Potential.matrix = Yukawa_matrix

        return

    def LJ_setup(self, filename):
#        bohr = 5.2917721067e-11
#        hartree = 4.36e-18
#        const.kb = 3.1668105011874e-06
        LJ_matrix = np.zeros((2, self.num_species, self.num_species))

        with open(filename, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)

            for lkey in dics:
                if (lkey == "Potential"):
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if (key == "epsilon"):
                                lj_m1 = np.array(value)

                            if (key == "sigma"):
                                lj_m2 = np.array(value)


        for j in range(self.num_species):
            for i in range(self.num_species):
                idx = i*self.num_species + j
                LJ_matrix[0, i, j] =  lj_m1[idx]
                LJ_matrix[1, i, j] =  lj_m2[idx]

        self.Potential.matrix = LJ_matrix
        self.force = LJ.potential_and_force
        return
