""" 
Module to handle particles' class.
"""
import os
import numpy as np
import time
import sys

DEBUG = 0


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

    def __init__(self, params):
        """ 
        Initialize the attributes
        """
        self.params = params
        self.checkpoint_dir = params.Control.dump_dir
        self.therm_checkpoint_dir = params.Control.therm_dump_dir
        self.box_lengths = params.Lv
        self.tot_num_ptcls = params.total_num_ptcls
        self.num_species = params.num_species

        if params.load_rand_seed is not None:
            self.rnd_gen = np.random.Generator(np.random.PCG64(params.load_rand_seed))
        else:
            self.rnd_gen = np.random.Generator(np.random.PCG64(123456789))

        self.pos = np.zeros((self.tot_num_ptcls, 3))
        self.vel = np.zeros((self.tot_num_ptcls, 3))
        self.acc = np.zeros((self.tot_num_ptcls, 3))
        self.pbc_cntr = np.zeros((self.tot_num_ptcls, 3))

        self.species_name = np.empty(self.tot_num_ptcls, dtype='object')
        self.species_id = np.zeros((self.tot_num_ptcls,), dtype=int)
        self.species_num = np.zeros(params.num_species, dtype=int)
        self.species_conc = np.zeros(params.num_species)
        self.species_mass = np.zeros(params.num_species)
        self.species_init_vel = np.zeros((params.num_species, 3))

        self.mass = np.zeros(self.tot_num_ptcls)  # mass of each particle
        self.charge = np.zeros(self.tot_num_ptcls)  # charge of each particle

        # No. of independent rdf
        self.no_grs = int(self.num_species * (self.num_species + 1) / 2)
        self.rdf_nbins = params.PostProcessing.rdf_nbins
        self.rdf_hist = np.zeros((self.rdf_nbins, self.num_species, self.num_species))

        # Assign particles attributes
        species_end = 0
        for ic, sp in enumerate(params.species):
            species_start = species_end
            species_end += sp.num

            self.species_num[ic] = sp.num
            self.species_conc[ic] = sp.num / self.tot_num_ptcls
            self.species_mass[ic] = sp.mass

            self.species_name[species_start:species_end] = sp.name
            self.mass[species_start:species_end] = sp.mass

            if hasattr(sp, 'charge'):
                self.charge[species_start:species_end] = sp.charge
            else:
                self.charge[species_start:species_end] = 1.0

            self.species_id[species_start:species_end] = ic

            if hasattr(sp, "init_vel"):
                self.species_init_vel[ic, :] = sp.init_vel

    def load(self, params):
        """
        Initialize particles' positions and velocities.
        Positions are initialized based on the load method while velocities are chosen
        from a Maxwell-Boltzmann distribution.

        """

        """
        Dev Notes: Here numba does not help at all. In fact loading is slower with numba. 
        It could be made faster if we made load a function and not a method of Particles.
        but in that case we would have to pass each parameter individually.
        """

        load_method = params.load_method
        if params.load_method == 'restart':
            if params.load_restart_step is None:
                raise AttributeError("Restart step not defined. Please define restart_step in YAML file.")
            if not type(params.load_restart_step) is int:
                raise TypeError("Only integers are allowed.")
            self.load_from_restart(params.load_restart_step)

        elif params.load_method == 'therm_restart':
            if params.load_therm_restart_step is None:
                raise AttributeError("Therm Restart step not defined. Please define restart_step in YAML file.")
            if not type(params.load_therm_restart_step) is int:
                raise TypeError("Only integers are allowed.")
            self.load_from_therm_restart(params.load_therm_restart_step)

        elif params.load_method == 'file':
            if params.ptcls_input_file is None:
                raise AttributeError('Input file not defined. Please define particle_input_file in YAML file.')
            self.load_from_file(params.ptcls_input_file)
        else:
            # Particles Velocities Initialization
            if params.Control.verbose:
                print('\nAssigning initial velocities from a Maxwell-Boltzmann distribution')

            species_end = 0
            for ic, sp in enumerate(params.species):
                Vsig = np.sqrt(params.kB * params.Thermostat.temperatures[ic] / sp.mass)
                species_start = species_end
                species_end = species_start + sp.num
                vel_0 = self.species_init_vel[ic, :]

                self.vel[species_start:species_end, 0] = self.rnd_gen.normal(vel_0[0], Vsig, sp.num)
                self.vel[species_start:species_end, 1] = self.rnd_gen.normal(vel_0[1], Vsig, sp.num)
                self.vel[species_start:species_end, 2] = self.rnd_gen.normal(vel_0[2], Vsig, sp.num)

            # Particles Position Initialization
            if params.Control.verbose:
                print('\nAssigning initial positions according to {}'.format(params.load_method))

            # position distribution. 
            if load_method == 'lattice':
                self.lattice(params.load_perturb)

            elif load_method == 'random_reject':
                self.random_reject(params.load_r_reject)

            elif load_method == 'halton_reject':
                self.halton_reject(params.load_halton_bases, params.load_r_reject)

            elif load_method == 'random_no_reject':
                self.random_no_reject()

            else:
                raise AttributeError('Incorrect particle placement scheme specified.')

        return

    def add(self):
        """ 
        Add more particles: specific species, and number of particles
        
        """
        pass

    def remove(self):
        """
        Remove particles: need to specify particle id

        """
        pass

    def update_attributes(self, params):
        """
        Update particles' attributes.

        Parameters
        ----------
        params : class
            `S_params` class containing the updated information.

        """
        species_end = 0

        self.species_name = np.empty(self.tot_num_ptcls, dtype='object')
        self.species_id = np.zeros((self.tot_num_ptcls,), dtype=int)
        self.species_num = np.zeros(params.num_species, dtype=int)
        self.species_mass = np.zeros(params.num_species)
        self.species_conc = np.zeros(params.num_species)
        self.mass = np.zeros(self.tot_num_ptcls)  # mass of each particle
        self.charge = np.zeros(self.tot_num_ptcls)  # charge of each particle

        for ic, sp in range(params.species):
            species_start = species_end
            species_end += sp.num

            self.species_num[ic] = sp.num
            self.species_mass[ic] = sp.mass
            self.species_conc[ic] = sp.num / self.tot_num_ptcls

            self.species_name[species_start:species_end] = sp.name
            self.mass[species_start:species_end] = sp.mass

            if hasattr(sp, 'charge'):
                self.charge[species_start:species_end] = sp.charge
            else:
                self.charge[species_start:species_end] = 1.0

            self.species_id[species_start:species_end] = ic

        return

    def load_from_restart(self, it):
        """
        Load particles' data from a checkpoint of a previous run

        Parameters
        ----------
        it : int
            Timestep.

        """
        file_name = os.path.join(self.checkpoint_dir, "S_checkpoint_" + str(it) + ".npz")
        data = np.load(file_name, allow_pickle=True)
        self.species_id = data["species_id"]
        self.species_name = data["species_name"]
        self.pos = data["pos"]
        self.vel = data["vel"]
        self.acc = data["acc"]
        self.pbc_cntr = data["cntr"]
        self.rdf_hist = data["rdf_hist"]

    def load_from_therm_restart(self, it):
        """
        Load particles' data from a checkpoint of a previous run

        Parameters
        ----------
        it : int
            Timestep.

        """
        file_name = os.path.join(self.therm_checkpoint_dir, "S_checkpoint_" + str(it) + ".npz")
        data = np.load(file_name, allow_pickle=True)
        self.species_id = data["species_id"]
        self.species_name = data["species_name"]
        self.pos = data["pos"]
        self.vel = data["vel"]
        self.acc = data["acc"]

    def load_from_file(self, f_name):
        """
        Load particles' data from a specific file.

        Parameters
        ----------
        f_name : str
            Filename
        """
        pv_data = np.loadtxt(f_name)
        if not (pv_data.shape[0] == self.tot_num_ptcls):
            print("Number of particles is not same between input file and initial p & v data file.")
            print("From the input file: N = ", self.tot_num_ptcls)
            print("From the initial p & v data: N = ", pv_data.shape[0])
            sys.exit()
        self.pos[:, 0] = pv_data[:, 0]
        self.pos[:, 1] = pv_data[:, 1]
        self.pos[:, 2] = pv_data[:, 2]

        self.vel[:, 0] = pv_data[:, 3]
        self.vel[:, 1] = pv_data[:, 4]
        self.vel[:, 2] = pv_data[:, 5]

    def random_no_reject(self):
        """
        Randomly distribute particles along each direction.

        Returns
        -------
        pos : array
            Particles' positions.

        """

        # np.random.seed(self.params.load_rand_seed) # Seed for random number generator

        self.pos[:, 0] = self.rnd_gen.uniform(0.0, self.box_lengths[0], self.tot_num_ptcls)
        self.pos[:, 1] = self.rnd_gen.uniform(0.0, self.box_lengths[1], self.tot_num_ptcls)
        self.pos[:, 2] = self.rnd_gen.uniform(0.0, self.box_lengths[2], self.tot_num_ptcls)

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
        part_per_side = self.tot_num_ptcls ** (1. / 3.)  # Number of particles per side of cubic lattice

        # Check if total number of particles is a perfect cube, if not, place more than the requested amount
        if round(part_per_side) ** 3 != self.tot_num_ptcls:
            part_per_side = np.ceil(self.tot_num_ptcls ** (1. / 3.))
            print('\nWARNING: Total number of particles requested is not a perfect cube.')
            print('Initializing with {} particles.'.format(int(part_per_side ** 3)))

        dx_lattice = self.box_lengths[0] / (self.tot_num_ptcls ** (1. / 3.))  # Lattice spacing
        dz_lattice = self.box_lengths[1] / (self.tot_num_ptcls ** (1. / 3.))  # Lattice spacing
        dy_lattice = self.box_lengths[2] / (self.tot_num_ptcls ** (1. / 3.))  # Lattice spacing

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
        while i < self.tot_num_ptcls - 1:

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
        while i < self.tot_num_ptcls:

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
