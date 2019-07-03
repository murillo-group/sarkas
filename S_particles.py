'''
S_particle.py

particle loading, adding, and removing

species_name
px,py,pz: position components
vx, vy, vz: velocity components
ax, ay, az: acc. components
charge
mass
'''
import numpy as np
from inspect import currentframe, getframeinfo
import time
import sys

DEBUG = 0

class particles:
    def __init__(self, params, total_num_part):

        

        N = total_num_part
        self.params = params
        #print(self.params)
        iseed = params.control[0].seed
        np.random.seed(seed=iseed)

        self.px = np.empty(N) 
        self.py = np.empty(N) 
        self.pz = np.empty(N) 

        self.vx = np.empty(N)
        self.vy = np.empty(N)
        self.vz = np.empty(N)

        self.ax = np.empty(N)
        self.ay = np.empty(N)
        self.az = np.empty(N)

        self.species_name = [None]*N

        self.mass = np.empty(N)
        self.charge = np.empty(N)


# initial whole particles loading.
# Here numba does not help at all. In fact loading is slower with numba. 
    def load(self, glb_vars, N):

        if (self.params.load[0].method == 'restart'):
            timestep = self.params.load[0].restart_step
            if(timestep == None):
                print("restart_step is not defined!!!")
                sys.exit()
            
            pos, vel, acc = self.load_from_restart(timestep)
            return pos, vel
            

        elif (self.params.load[0].method == 'file'):
            print('\nReading initial particle positions and velocities from file...')
            
            f_input = 'init.out'           # name of input file
            self.load_from_file(f_input, N)

            return np.transpose(np.array([self.px, self.py, self.pz])), np.transpose(np.array([self.vx, self.vy, self.vz]))

        else:

            if self.params.load[0].method == 'random_no_reject':
                print('\nAssigning random initial positions {}'.format(self.params.load[0].method))

            else:
                print('\nAssigning initial positions according to {}'.format(self.params.load[0].method))

            print('Assigning random initial velocities...')


            Lx = glb_vars.Lx
            Ly = glb_vars.Ly
            Lz = glb_vars.Lz

            idx_end = 0
            for i, load in enumerate(self.params.load):
                idx_start = idx_end
                idx_end += self.params.load[i].Num
                for j in range(idx_start, idx_end):
                    self.species_name[j] = self.params.load[i].species_name

            for i in range(N):
                for j, species in enumerate(self.params.species):

                    if(DEBUG):
                        frameinfo = getframeinfo(currentframe())
                        print( frameinfo.filename, frameinfo.lineno)
                        print("===", self.species_name[i], self.params.species[j].name)

                    if(self.species_name[i] == self.params.species[j].name):
                        self.mass[j] = self.params.species[j].mass
                        self.charge[j] = self.params.species[j].charge
                        break

            two_pi = 2*np.pi 

            potential_type = self.params.potential[0].type
            units = self.params.control[0].units

#   Here, we assume one species model. Will be expanded for multi-species in the future.
            if(potential_type == "Yukawa"):
                if(units == "cgs"):
                    #Vsig = np.sqrt(q1*q2/ai/mi/Gamma)
                    print("Please use 'Yukawa' units for Yukawa potential.")
                    sys.exit()

                elif(units == "mks"):
                    #Vsig = np.sqrt(q1*q2/ai/mi/Gamma/(4*np.pi*const.eps_0))
                    print("Please use 'Yukawa' units for Yukawa potential.")

                elif(units == "Yukawa"):
                    if(DEBUG):
                        frameinfo = getframeinfo(currentframe())
                        print(frameinfo.filename, frameinfo.lineno)
                    q1 = 1. 
                    q2 = 1.
                    mi = 1.
                    Gamma = glb_vars.Gamma
                    T_desired = 1/Gamma

                    Vsig = np.sqrt(1./Gamma/3)

                if(potential_type == "EGS"):
                    print("Will be implemented...")
                    sys.exit()
#                if(units == "cgs"):
#                  sig = np.sqrt(q1*q2/ai/mi/T_desired)
#                elif(units == "mks"):
#                  sig = np.sqrt(q1*q2/ai/mi/T_desired/(4*np.pi*const.eps_0))

            #Box-Muller transform to generate Gaussian random numbers from uniform random numbers 
            u1 = np.random.random(N)
            u2 = np.random.random(N)
            u3 = np.random.random(N)
            u4 = np.random.random(N)

            self.vx[:] = Vsig*np.sqrt(-2*np.log(u1))*np.cos(two_pi*u2) #distribution of vx
            self.vy[:] = Vsig*np.sqrt(-2*np.log(u1))*np.sin(two_pi*u2) #distribution of vy
            self.vz[:] = Vsig*np.sqrt(-2*np.log(u3))*np.cos(two_pi*u4) #distribution of vz
        
            #computing the mean of each velocity component to impose mean value of the velocity components to be zero
            vx_mean = np.mean(self.vx)
            vy_mean = np.mean(self.vy)
            vz_mean = np.mean(self.vz)

            #mean value of the velocity components to be zero
            self.vx -= vx_mean
            self.vy -= vy_mean
            self.vz -= vz_mean

            if (load.method == 'lattice'):
                self.lattice(N, self.params.load[0].perturb, self.params.load[0].rand_seed)

            elif (load.method == 'random_reject'):
                self.random_reject(N, self.params.load[0].r_reject, self.params.load[0].rand_seed)

            elif (load.method == 'halton_reject'):
                self.halton_reject(N, self.params.load[0].halton_bases, self.params.load[0].r_reject)

            elif (load.method == 'random_no_reject'):
                self.px = Lx*np.random.random(N)
                self.py = Ly*np.random.random(N)
                self.pz = Lz*np.random.random(N)

            else:
                print('Incorrect particle placement scheme specified... Using "random_no_reject"')
                self.px = Lx*np.random.random(N)
                self.py = Ly*np.random.random(N)
                self.pz = Lz*np.random.random(N)
               
            return np.transpose(np.array([self.px, self.py, self.pz])), np.transpose(np.array([self.vx, self.vy, self.vz]))


# add more particles: specific species, and number of particles
    def add(self):
        pass

# remove particles: need to specify particle id
    def remove(self):
        pass

    def update(self):
        pass

    def load_from_restart(self, it):
        file_name = "Checkpoint/S_checkpoint_"+str(it)+".npz"
        data = np.load(file_name)
        pos = data["pos"]
        vel = data["vel"]
        acc = data["acc"]
        params = data["params"]

        return pos, vel, acc


    def load_from_file(self, f_name, N):
        
        pv_data = np.loadtxt(f_name)
        if not (pv_data.shape[0] == N):
            print("Number of particles is not same between input file and initial p & v data file.")
            print("From the input file: N = ", N)
            print("From the initial p & v data: N = ", pv_data.shape[0])
            sys.exit()
        self.px = pv_data[:, 0]
        self.py = pv_data[:, 1]
        self.pz = pv_data[:, 2]

        self.vx = pv_data[:, 3]
        self.vy = pv_data[:, 4]
        self.vz = pv_data[:, 5]

    def lattice(self, N, perturb, rand_seed):
        ''' Place particles in a simple cubic lattice with a slight perturbation
        ranging from 0 to 0.5 times the lattice spacing.

        Parameters
        ----------
        N_particles : int
            Number of particles to be placed.

        perturb : float
            Value of perturbation, p, such that 0 <= p <= 1.

        rand_seed : int
            Seed for random number generator. 
            Default: 1.

        Returns
        -------
        x : array_like
            X positions for particles.
                
        y : array_like
            Y positions for particles.
            
        z : array_like
            Z positions for particles.
        Notes
        -----    
        Author: Luke Stanek
        Date Created: 5/6/19
        Date Updated: 6/2/19
        Updates: Added to S_init_schemes.py for Sarkas import
        '''
                          
        # Check if perturbation is below maximum allowed. If not, default to maximum perturbation.
        if perturb > 1:
            print('Warning: Random perturbation must not exceed 1. Setting perturb = 1.')
            perturb = 1 # Maximum perturbation
            
        print('Initializing particles with maximum random perturbation of {} times the lattice spacing.'.format(perturb*0.5))
        
        # Determining number of particles per side of simple cubic lattice
        part_per_side = (N)**(1/3) # Number of particles per side of cubic lattice
        
        # Check if total number of particles is a perfect cube, if not, place more than the requested amount
        if round(part_per_side)**3 != N:
        
            part_per_side = np.ceil( N**(1/3) )
            print('Warning: Total number of particles requested is not a perfect cube. Initializing with {} particles.'.format( int(part_per_side**3) ))

        L = (4 * np.pi * N/3)**(1/3) # Box length normalized by weigner-steitz radius

        d_lattice =  L/(N**(1/3)) # Lattice spacing

        # Start timer
        start = time.time()
        
        # Create x, y, and z position arrays
        x = np.arange( 0, L, d_lattice ) + 0.5 * d_lattice
        y = np.arange( 0, L, d_lattice ) + 0.5 * d_lattice
        z = np.arange( 0, L, d_lattice ) + 0.5 * d_lattice

        # Create a lattice with approprate x, y, and z values based on arange
        X, Y, Z = np.meshgrid(x,y,z)
        # Random seed
        # print('Random number generator using rand_seed = {}'.format(self.rand_seed))
        np.random.seed(1) # Seed for random number generator
        
        # Perturb lattice
        X +=  np.random.uniform(-0.5, 0.5, np.shape(X)) * perturb * d_lattice
        Y +=  np.random.uniform(-0.5, 0.5, np.shape(Y)) * perturb * d_lattice
        Z +=  np.random.uniform(-0.5, 0.5, np.shape(Z)) * perturb * d_lattice

        # Flatten the meshgrid values for plotting and computation
        self.px = X.ravel()
        self.py = Y.ravel()
        self.pz = Z.ravel()
        
        # End timer
        end = time.time()
        print('Lattice Elapsed time: ', end - start)

    def random_reject(self, N, r_reject, rand_seed):
        ''' Place particles by sampling a uniform distribution from 0 to L (the box length)
            and uses a rejection radius to avoid placing particles to close to each other.
        
        Parameters
        ----------
        N_part : int
            Total number of particles to place.

        r_reject : float
            Value of rejection radius.

        rand_seed : int
            Seed for random number generator. 
            Default: 1.

        Returns
        -------
        x : array_like
            X positions for particles.
            
        y : array_like
            Y positions for particles.
        
        z : array_like
            Z positions for particles.

        Notes
        -----    
        Author: Luke Stanek
        Date Created: 5/6/19
        Date Updated: N/A
        Updates: N/A

        '''
        
        # Determine box side length
        L = (4 * np.pi * N/3)**(1/3)

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
                
        self.px = x
        self.py = y
        self.pz = z

        end = time.time()
        print('Random Elapsed time: ', end - start)

    def halton_reject(self, N, bases, r_reject):

        ''' Place particles according to a Halton sequence from 0 to L (the box length)
            and uses a rejection radius to avoid placing particles to close to each other.
    
        Parameters
        ----------
        N_part : int
            Total number of particles to place.

        bases : array_like
            Array of 3 ints each of which is a base for the Halton sequence.
            Defualt: bases = np.array([2,3,5])

        r_reject : float
            Value of rejection radius.

        Returns
        -------
        x : array_like
            X positions for particles.
            
        y : array_like
            Y positions for particles.
        
        z : array_like
            Z positions for particles.

        Notes
        -----    
        Author: Luke Stanek
        Date Created: 5/6/19
        Date Updated: N/A
        Updates: N/A

        '''
        
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
        
        # Counter for bad placements
        bad_count = 0

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
                    bad_count += 1
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

        self.px = x
        self.py = y
        self.pz = z

        # End timer        
        end = time.time()
        print('Halton Elapsed time: ', end - start)


