'''
S_integrator.py

velocity integrator

Verlet: velocity Verlet
Verlet_with_Langevin: Verlet with Langevin damping
RK45: N/A
RK45_with_Langevin: N/A
'''

import numpy as np
import sys
import S_calc_force as calc_force
import S_constants as const

class Integrator:
    def __init__(self, params):

        self.calc_force = calc_force.force_pot

        self.params = params

        self.dt = params.Control.dt
        self.half_dt = 0.5*self.dt
        self.N = params.N
        self.d = params.d
        self.Lv = params.Lv
        self.PBC = params.Control.PBC
        self.Lmax_v = params.Lmax_v
        self.Lmin_v = params.Lmin_v
        self.N_species = self.params.num_species
        self.T = self.params.Ti

        if(params.Integrator.type == "Verlet"):
            if(params.Langevin.on):
                if(params.Langevin.type == "BBK"):
                    self.update = self.Verlet_with_Langevin     # currently only BBK.
                else:
                    print("No such Langevin type.")
                    sys.exit()

                self.g = self.params.Langevin.gamma
                self.c1 = (1. - 0.5*self.g*self.dt)
                self.c2 = 1./(1. + 0.5*self.g*self.dt)
                self.sqrt_dt = np.sqrt(self.dt)

            else:
                self.update = self.Verlet
        else:
            print("Only Verlet integrator is supported. Check your input file, integrator part.")
            sys.exit()

    def Verlet(self, ptcls):
        ''' Update particle position and velocity based on velocity verlet method.
        More information can be found here: https://en.wikipedia.org/wiki/Verlet_integration
        or on the Sarkas website. 
    
        Parameters
        ----------
        ptlcs: particles data. See S_particles.py for the detailed information

        Returns
        -------
        U : float
            Total potential energy
        '''
        # Import global parameters (is there a better way to do this?)


        # First half step velocity update
        ptcls.vel = ptcls.vel + ptcls.acc*self.half_dt

        # Full step position update
        ptcls.pos = ptcls.pos + ptcls.vel*self.dt
        # Periodic boundary condition
        N = len(ptcls.pos[:, 0])
        if self.PBC == 1:
            
            # Loop over all particles
            for i in np.arange(self.N):
                # Loop over dimensions (x=0, y=1, z=2)
                for p in np.arange(self.d):
                    
                    # If particle is outside of box in positive direction, wrap to negative side
                    if ptcls.pos[i, p] > self.Lmax_v[p]:
                        ptcls.pos[i, p] = ptcls.pos[i, p] - self.Lv[p]
                    
                    # If particle is outside of box in negative direction, wrap to positive side
                    if ptcls.pos[i, p] < self.Lmin_v[p]:
                        ptcls.pos[i, p] = ptcls.pos[i, p] + self.Lv[p]


        # Compute total potential energy and accleration for second half step velocity update                 
        U = self.calc_force(ptcls,self.params)
        
        #Second half step velocity update
        ptcls.vel = ptcls.vel + ptcls.acc*self.half_dt

        return U
    
    '''
    BBK integrator
    A. Brünger, C. L. Brooks III, M. Karplus, 
    Stochastic boundary conditions fro molecular dynamics simulations of ST2 water. Chem. Phys. Letters, 1984, 105 (5) 495-500.
    '''
    def Verlet_with_Langevin(self, ptcls): 
 
        beta1 = np.random.normal(0., 1., 3*self.N).reshape(self.N, 3)
        beta2 = np.random.normal(0., 1., 3*self.N).reshape(self.N, 3)

        Vsigma = np.zeros(self.N_species)
        for i in range(self.N_species):
                Vsigma[i] = np.sqrt(2.*self.g*const.kb*self.T/self.params.species[i].mass)

        # First half step velocity update
        species_start = 0
        species_end = 0
        for ic in range(self.N_species):
            Vsig = Vsigma[ic]
            num_ptcls = self.params.species[ic].num
            species_start = species_end
            species_end = species_start + num_ptcls

            ptcls.vel[species_start:species_end, :] = \
                self.c1*ptcls.vel[species_start:species_end, :] + self.half_dt*ptcls.acc[species_start:species_end, :] + \
                0.5*Vsig*self.sqrt_dt*beta1[species_start:species_end, :]

            # Full step position update
            ptcls.pos[species_start:species_end, :] += self.dt*ptcls.vel[species_start:species_end]


            # Periodic boundary condition
            if self.PBC == 1:
                
                # Loop over all particles
                for i in np.arange(self.N):
                    # Loop over dimensions (x=0, y=1, z=2)
                    for p in np.arange(self.d):
                        
                        # If particle is outside of box in positive direction, wrap to negative side
                        if ptcls.pos[i, p] > self.Lmax_v[p]:
                            ptcls.pos[i, p] = ptcls.pos[i, p] - self.Lv[p]
                        
                        # If particle is outside of box in negative direction, wrap to positive side
                        if ptcls.pos[i, p] < self.Lmin_v[p]:
                            ptcls.pos[i, p] = ptcls.pos[i, p] + self.Lv[p]


            # Compute total potential energy and accleration for second half step velocity update                 
            acc = ptcls.acc
            U = self.calc_force(ptcls)
            acc_new = ptcls.acc


            ptcls.vel[species_start:species_end, :] = \
                self.c1*self.c2*ptcls.vel[species_start:species_end, :] + \
                self.half_dt*(acc_new[species_start:species_end, :] + acc[species_start:species_end, :])*self.c2 + \
                self.c2*Vsig*self.sqrt_dt*beta2[species_start:species_end, :]

            # Second half step velocity update



        if(0):
            sig = np.sqrt(2. * g*const.kb*self.params.T_desired/const.proton_mass)

            c1 = (1. - 0.5*g*dt)
            c2 = 1./(1. + 0.5*g*dt)
            beta = np.random.normal(0., 1., 3*N).reshape(N, 3)

            ptcls.pos = ptcls.pos + c1*dt*ptcls.vel + 0.5*dt**2*ptcls.acc + 0.5*sig*dt**1.5*beta

            # periodic boundary condition
            if PBC == 1:
                for i in np.arange(N):
                    for p in np.arange(d):
                        if ptcls.pos[i, p] > Lmax_v[p]:
                            ptcls.pos[i, p] = ptcls.pos[i, p] - self.Lv[p]
                        if ptcls.pos[i, p] < Lmin_v[p]:
                            ptcls.pos[i, p] = ptcls.pos[i, p] + self.Lv[p]

            acc = ptcls.acc
            U = self.calc_force(ptcls)
            acc_new = ptcls.acc
            ptcls.vel = c1*c2*ptcls.vel + 0.5*dt*(acc_new + acc)*c2 + c2*sig*rtdt*beta

        return U

    def RK(self, ptcls):
        ''' Update particle position and velocity based on the 4th order Runge-Kutta method
        More information can be found here: 
        https://en.wikipedia.org/wiki/Runge–Kutta_methods
        or on the Sarkas website. 
    
        Parameters
        ----------
        ptlcs: particles data. See S_particles.py for the detailed information
        k1: the vel, acc at the beginng
        k2: the vel, acc at the middle
        k3: the vel, acc at the middle if the acc. at the beginning was k2
        k4: the vel, acc at the end if the acc. at the beginning was k3

        Returns
        -------
        U : float
            Total potential energy
        '''
        # Import global parameters (is there a better way to do this?)
        pass

    def RK45(self, ptcls):
        ''' Update particle position and velocity based on Explicit Runge-Kutta method of order 5(4). 
        More information can be found here: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html
        https://en.wikipedia.org/wiki/Runge–Kutta_methods
        or on the Sarkas website. 
    
        Parameters
        ----------
        ptlcs: particles data. See S_particles.py for the detailed information

        Returns
        -------
        U : float
            Total potential energy
        '''
        # Import global parameters (is there a better way to do this?)
        pass

    def RK45_with_Langevin(self, ptcls):
        pass
