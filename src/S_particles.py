""" 
Module to handle particles' class.
"""
import numpy as np
from inspect import currentframe, getframeinfo
import time
import sys

DEBUG = 0

class Particles:
    """
    Particles class.

    Parameters
    ----------
    params : class
        Simulation's parameters. See ``S_params.py`` for more info.

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

    N : int
        Total number of particles.

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
    """

    def __init__(self, params):
        """ 
        Initialize the attributes
        """
        self.params = params
        self.N = params.total_num_ptcls

        iseed = params.load_rand_seed
        np.random.seed(seed=iseed)

        self.pos = np.zeros((self.N, 3)) 
        self.vel = np.zeros((self.N, 3)) 
        self.acc = np.zeros((self.N, 3)) 

        self.species_name = np.empty(self.N, dtype='object')
        self.species_id = np.zeros( (self.N,), dtype=int)
        self.species_num = np.zeros( self.params.num_species, dtype=int )

        self.mass = np.zeros(self.N) # mass of each particle
        self.charge = np.zeros(self.N) # charge of each particle
        return

    def assign_attributes(self,params):
        """ Assign particles attributes """

        self.species_name = np.empty(self.N, dtype='object')
        self.species_id = np.zeros( (self.N,), dtype=int)
        self.species_num = np.zeros( self.params.num_species, dtype=int )
        self.species_mass = np.zeros( self.params.num_species )

        self.mass = np.zeros(self.N) # mass of each particle
        self.charge = np.zeros(self.N) # charge of each particle
        
        species_start = 0
        species_end = 0
        
        ic_species = 0
        for i in range(params.num_species): 
            species_start = species_end
            species_end += self.params.species[i].num
           
            self.species_num[i] = self.params.species[i].num
            self.species_mass[i] = self.params.species[i].mass

            self.species_name[species_start:species_end] = self.params.species[i].name
            self.mass[species_start:species_end] = self.params.species[i].mass

            if hasattr (self.params.species[i],'charge'):
                self.charge[species_start:species_end] = self.params.species[i].charge
            else:
                self.charge[species_start:species_end] = 1.0
            
            self.species_id[species_start:species_end] = ic_species
            ic_species += 1

    def load(self):
        """
        Initialize particles' positions and velocities.
        Positions are initilized based on the load method while velocities are chosen 
        from a Maxwell-Boltzmann distribution.

        """
        
        """
        Dev Notes: Here numba does not help at all. In fact loading is slower with numba. 
        It could be made faster if we made load a function and not a method of Particles.
        but in that case we would have to pass all the parameters.
        """
        N = self.N
        Lx = self.params.Lx
        Ly = self.params.Ly
        Lz = self.params.Lz
        
        N_species = self.params.num_species

        load_method = self.params.load_method
        if (load_method == 'restart'):
            timestep = self.params.load_restart_step
            if(timestep == None):
                print("restart_step is not defined!!!")
                sys.exit()
            
            self.load_from_restart(timestep)

        elif (load_method == 'file'):
            if ( self.params.Control.screen_output):
                print('\nReading initial particle positions and velocities from file...')
            
            f_input = 'init.out'           # name of input file
            self.load_from_file(f_input, N)

        else:
            two_pi = 2.0*np.pi 

            # Particles Velocities Initialization
            if ( self.params.Control.screen_output):
                print('\nAssigning initial velocities from a Maxwell-Boltzmann distribution')
            potential_type = self.params.Potential.type
            units = self.params.Control.units

            Vsigma = np.zeros(N_species)
            for i in range(N_species):
                Vsigma[i] = np.sqrt(self.params.kB*self.params.Ti/self.params.species[i].mass)

            species_start = 0
            species_end = 0
            for ic in range(N_species):
                Vsig = Vsigma[ic]
                num_ptcls = self.params.species[ic].num
                species_start = species_end
                species_end = species_start + num_ptcls

                self.vel[species_start:species_end,0] = np.random.normal(0.0,Vsig,num_ptcls)
                self.vel[species_start:species_end,1] = np.random.normal(0.0,Vsig,num_ptcls)
                self.vel[species_start:species_end,2] = np.random.normal(0.0,Vsig,num_ptcls)
                
                # Enforce zero total momentum
                vx_mean = np.mean(self.vel[species_start:species_end, 0])
                vy_mean = np.mean(self.vel[species_start:species_end, 1])
                vz_mean = np.mean(self.vel[species_start:species_end, 2])

                self.vel[species_start:species_end, 0] -= vx_mean
                self.vel[species_start:species_end, 1] -= vy_mean
                self.vel[species_start:species_end, 2] -= vz_mean

            # Particles Position Initialization
            if ( self.params.Control.screen_output):
                print('\nAssigning initial positions according to {}'.format(load_method))

            # position distribution. 
            if (load_method == 'lattice'):
                self.lattice(self.N, self.params.load_perturb, self.params.load_rand_seed)

            elif (load_method == 'random_reject'):
                self.random_reject(self.N, self.params.load_r_reject, self.params.load_rand_seed)

            elif (load_method == 'halton_reject'):
                self.halton_reject(self.N, self.params.load_halton_bases, self.params.load_r_reject)

            elif (load_method == 'random_no_reject'):
                self.random_no_reject(self.N)

            else:
                if (params.Control.verbose):
                    print('\nIncorrect particle placement scheme specified... Using "random_no_reject"')
                self.random_no_reject(self.N)

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

    def update(self):
        pass

    def load_from_restart(self, it):
        """
        Load particles' data from a checkpoint of a previous run

        Parameters
        ----------
        it : int
            Timestep

        """
        file_name = self.params.Control.checkpoint_dir+"/"+"S_checkpoint_"+str(it)+".npz"
        data = np.load(file_name,allow_pickle=True)
        self.species_id = data["species_id"]
        self.species_name = data["species_name"]
        self.pos = data["pos"]
        self.vel = data["vel"]
        self.acc = data["acc"]

    def load_from_file(self, f_name, N):
        """
        Load particles' data from a specific file.

        Parameters
        ----------
        f_name : str
            Filename

        N : int
            Number of particles

        """
        pv_data = np.loadtxt(f_name)
        if not (pv_data.shape[0] == N):
            print("Number of particles is not same between input file and initial p & v data file.")
            print("From the input file: N = ", N)
            print("From the initial p & v data: N = ", pv_data.shape[0])
            sys.exit()
        self.pos[:, 0] = pv_data[:, 0]
        self.pos[:, 1] = pv_data[:, 1]
        self.pos[:, 2] = pv_data[:, 2]

        self.vel[:, 0] = pv_data[:, 3]
        self.vel[:, 1] = pv_data[:, 4]
        self.vel[:, 2] = pv_data[:, 5]

    def random_no_reject(self, N):
        """
        Randomly distribute particles along each direction.

        Parameters
        ----------
        N : int
            Number of particles.

        Returns
        -------
        pos : array
            Particles' positions.

        """

        #np.random.seed(self.params.load_rand_seed) # Seed for random number generator

        self.pos[:, 0] = self.params.Lx*np.random.random(N)
        self.pos[:, 1] = self.params.Ly*np.random.random(N)
        self.pos[:, 2] = self.params.Lz*np.random.random(N)

    def lattice(self, N, perturb, rand_seed):
        """ 
        Place particles in a simple cubic lattice with a slight perturbation ranging from 0 to 0.5 times the lattice spacing.

        Parameters
        ----------
        N_particles : int
            Number of particles to be placed.

        perturb : float
            Value of perturbation, p, such that 0 <= p <= 1.

        rand_seed : int
            Seed for random number generator. 
            Default: 1.

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
            perturb = 1 # Maximum perturbation
            
        print('Initializing particles with maximum random perturbation of {} times the aattice spacing.'.format(perturb*0.5))
        
        # Determining number of particles per side of simple cubic lattice
        part_per_side = (N)**(1./3.) # Number of particles per side of cubic lattice
        
        # Check if total number of particles is a perfect cube, if not, place more than the requested amount
        if round(part_per_side)**3 != N:
        
            part_per_side = np.ceil( N**(1./3.) )
            print('Warning: Total number of particles requested is not a perfect cube. Initializing with {} particles.'.format( int(part_per_side**3) ))

        #L = (4 * np.pi * N/3)**(1/3) # Box length normalized by weigner-steitz radius
        L = self.params.L

        d_lattice =  L/(N**(1./3.)) # Lattice spacing

        # Start timer
        start = time.time()
        
        # Create x, y, and z position arrays
        x = np.arange( 0, L, d_lattice ) + 0.5 * d_lattice
        y = np.arange( 0, L, d_lattice ) + 0.5 * d_lattice
        z = np.arange( 0, L, d_lattice ) + 0.5 * d_lattice

        # Create a lattice with approprate x, y, and z values based on arange
        X, Y, Z = np.meshgrid(x,y,z)

        # Random seed
        if ( self.params.Control.screen_output):
            print('Random number generator using rand_seed = {}'.format(rand_seed))
        np.random.seed(rand_seed) # Seed for random number generator
        
        # Perturb lattice
        X +=  np.random.uniform(-0.5, 0.5, np.shape(X)) * perturb * d_lattice
        Y +=  np.random.uniform(-0.5, 0.5, np.shape(Y)) * perturb * d_lattice
        Z +=  np.random.uniform(-0.5, 0.5, np.shape(Z)) * perturb * d_lattice

        # Flatten the meshgrid values for plotting and computation
        self.pos[:, 0] = X.ravel()
        self.pos[:, 1] = Y.ravel()
        self.pos[:, 2] = Z.ravel()
        
        # End timer
        end = time.time()
        print('Lattice Elapsed time: ', end - start)


    def random_reject(self, N, r_reject, rand_seed):
        """ 
        Place particles by sampling a uniform distribution from 0 to L (the box length)
        and uses a rejection radius to avoid placing particles to close to each other.
        
        Parameters
        ----------
        N : int
            Total number of particles to place.

        r_reject : float
            Value of rejection radius.

        rand_seed : int
            Seed for random number generator. 
            Default: 1.

        Notes
        -----    
        Author: Luke Stanek
        Date Created: 5/6/19
        Date Updated: N/A
        Updates: N/A

        """

        # Set random seed
        np.random.seed(rand_seed)
        
        # Determine box side length
        L = (4 * np.pi * N/3)**(1/3)
        L = self.params.L

        # Initialize Arrays
        x = np.array([])
        y = np.array([])
        z = np.array([])

        # Set first x, y, and z positions
        x_new = np.random.uniform(0, L)
        y_new = np.random.uniform(0, L)
        z_new = np.random.uniform(0, L)

        # Append to arrays
        x = np.append(x, x_new)
        y = np.append(y, y_new)
        z = np.append(z, z_new)

        # Particle counter
        i = 0

        start = time.time() # Start timer for placing particles
        bad_count = 0
        # Loop to place particles
        while i < N - 1:
            
            # Set x, y, and z positions
            x_new = np.random.uniform(0, L)
            y_new = np.random.uniform(0, L)
            z_new = np.random.uniform(0, L)   
            
            # Check if particle was place too close relative to all other current particles
            for j in range(len(x)):

                # Flag for if particle is outside of cutoff radius (True -> not inside rejection radius)
                flag = 1

                # Compute distance b/t particles for initial placement
                x_diff = x_new - x[j]
                y_diff = y_new - y[j]
                z_diff = z_new - z[j]

                # periodic condition applied for minimum image
                if(x_diff < -L/2):
                    x_diff = x_diff + L
                if(x_diff > L/2):
                    x_diff = x_diff - L
                    
                if(y_diff < -L/2):
                    y_diff = y_diff + L
                if(y_diff > L/2):
                    y_diff = y_diff - L
                    
                if(z_diff < -L/2):
                    z_diff = z_diff + L
                if(z_diff > L/2):
                    z_diff = z_diff - L

                # Compute distance
                r = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                # Check if new particle is below rejection radius. If not, break out and try again
                if r <= r_reject:
                    flag = 0 # new position not added (False -> no longer outside reject r)
                    break

            # If flag true add new positiion
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
        print('Random Elapsed time: ', end - start)

    def halton_reject(self, N, bases, r_reject):
        """ 
        Place particles according to a Halton sequence from 0 to L (the box length)
        and uses a rejection radius to avoid placing particles to close to each other.
    
        Parameters
        ----------
        N : int
            Total number of particles to place.

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

        # Determine box side length
        L = (4 * np.pi * N/3)**(1/3) # Box length normalized by weigner-steitz radius

        # Allocate space and store first value from Halton
        x = np.array([0])
        y = np.array([0])
        z = np.array([0])
        
        # Initialize particle counter and Halton counter
        i = 1
        k = 1

        # Start timer
        start = time.time() # Start timer for placing particles
        
        # Loop over all particles
        while i < N:

            # Increment particle counter
            n = k
            m = k
            p = k
            
            # Determine x coordinate
            f1 = 1
            r1 = 0
            while n > 0:
                f1 /= b1
                r1 += f1 * ( n % int(b1) )
                n = np.floor(n/b1)
            x_new = L*r1 # new x value
            
            # Determine y coordinate
            f2 = 1
            r2 = 0
            while m > 0:
                f2 /= b2
                r2 += f2 * ( m % int(b2) )
                m = np.floor(m/b2)
            y_new = L*r2 # new y value

            # Determine z coordinate
            f3 = 1
            r3 = 0
            while p > 0:
                f3 /= b3
                r3 += f3 * ( p % int(b3) )
                p = np.floor( p/b3 )
            z_new = L*r3 # new z value
        
            # Check if particle was place too close relative to all other current particles
            for j in range(len(x)):

                # Flag for if particle is outside of cutoff radius (1 -> not inside rejection radius)
                flag = 1

                # Compute distance b/t particles for initial placement
                x_diff = x_new - x[j]
                y_diff = y_new - y[j]
                z_diff = z_new - z[j]

                # Periodic condition applied for minimum image
                if(x_diff < -L/2):
                    x_diff = x_diff + L
                if(x_diff > L/2):
                    x_diff = x_diff - L
                    
                if(y_diff < -L/2):
                    y_diff = y_diff + L
                if(y_diff > L/2):
                    y_diff = y_diff - L
                    
                if(z_diff < -L/2):
                    z_diff = z_diff + L
                if(z_diff > L/2):
                    z_diff = z_diff - L

                # Compute distance
                r = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                # Check if new particle is below rejection radius. If not, break out and try again
                if r <= r_reject:
                    k += 1 # Increment Halton counter
                    flag = 0 # New position not added (0 -> no longer outside reject r)
                    break

            # If flag true add new positiion
            if flag == 1:
                
                # Add new positions to arrays
                x = np.append(x, x_new)
                y = np.append(y, y_new)
                z = np.append(z, z_new)
                
                k += 1 # Increment Halton counter
                i += 1 # Increment particle number

        self.pos[:, 0] = x
        self.pos[:, 1] = y
        self.pos[:, 2] = z

        # End timer        
        end = time.time()
        print('Halton Elapsed time: ', end - start)


