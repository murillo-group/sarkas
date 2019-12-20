"""
S_integrator.py

Module of various types of integrators 

Verlet : velocity Verlet
Verlet_with_Langevin : Verlet with Langevin damping
RK45: N/A
RK45_with_Langevin: N/A
"""

import numpy as np
import numba as nb
import sys
import S_calc_force_pp as force_pp
import S_calc_force_pm as force_pm
import S_constants as const

class Integrator:
    def __init__(self, params):

        self.calc_force = PotentialAcceleration

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
                self.update = Verlet
        else:
            print("Only Verlet integrator is supported. Check your input file, integrator part.")
            sys.exit()


    def Verlet_with_Langevin(self, ptcls):
        """ Calculate particles dynamics using the Velocity Verlet algorithm and Langevin damping.

        Parameters
        ----------
        ptcls : class
                Particles' data. See S_particles for more information

        Return
        ------
        U : float
            Potential Energy

        """
        dt = self.params.Control.dt
        g = self.params.Langevin.gamma
        Gamma = self.params.Potential.Gamma
        Lmax_v = self.params.Lmax_v
        Lmin_v = self.params.Lmin_v
        Lv = self.parms.Lv
        PBC = self.params.PBC
        N = self.params.N
        d = self.params.d

        rtdt = np.sqrt(dt)

        sig = np.sqrt(2. * g*const.kb*self.params.T_desired/const.proton_mass)
        if(self.params.units == "Yukawa"):
            sig = np.sqrt(2. * g/(3*Gamma))

        c1 = (1. - 0.5*g*dt)
        c2 = 1./(1. + 0.5*g*dt)
        beta = np.random.normal(0., 1., 3*N).reshape(N, 3)

        ptcls.pos = ptcls.pos + c1*dt*ptcls.vel + 0.5*dt**2*ptcls.acc + 0.5*sig*dt**1.5*beta

        # periodic boundary condition
        if PBC == 1:
            for i in np.arange(N):
                for p in np.arange(d):
                    if ptcls.pos[i, p] > Lmax_v[p]:
                        ptcls.pos[i, p] = ptcls.pos[i, p] - Lv[p]
                    if ptcls.pos[i, p] < Lmin_v[p]:
                        ptcls.pos[i, p] = ptcls.pos[i, p] + Lv[p]

        acc = ptcls.acc
        U = PotentialAcceleration(ptcls,params)
        acc_new = ptcls.acc
        ptcls.vel = c1*c2*ptcls.vel + 0.5*dt*(acc_new + acc)*c2 + c2*sig*rtdt*beta
        return U

    def RK(self, ptcls):
        """ Update particle position and velocity based on the 4th order Runge-Kutta method
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
        """
        # Import global parameters (is there a better way to do this?)
        dt = self.glb_vars.dt
        half_dt = 0.5*dt
        N = self.glb_vars.N
        d = self.glb_vars.d
        Lv = self.glb_vars.Lv

        pass

    def RK45(self, ptcls):
        """ Update particle position and velocity based on Explicit Runge-Kutta method of order 5(4). 
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
        """

        # Import global parameters (is there a better way to do this?)
        # Yes use self.params or just pass params
        dt = self.glb_vars.dt
        N = self.glb_vars.N
        d = self.glb_vars.d
        Lv = self.glb_vars.Lv
        PBC = self.glb_vars.PBC
        Lmax_v = self.glb_vars.Lmax_v
        Lmin_v = self.glb_vars.Lmin_v
        pass

    def RK45_with_Langevin(self, ptcls):
        pass


def Verlet(ptcls,params):
    """ Update particle position and velocity based on velocity verlet method.
    More information can be found here: https://en.wikipedia.org/wiki/Verlet_integration
    or on the Sarkas website. 

    Parameters
    ----------
    ptlcs: particles data. See S_particles.py for the detailed information. 
    Returns
    -------
    U : float
        Total potential energy
    """
    # Import global parameters (is there a better way to do this?)


    # First half step velocity update
    ptcls.vel = ptcls.vel + 0.5*ptcls.acc*params.Control.dt

    # Full step position update
    ptcls.pos = ptcls.pos + ptcls.vel*params.Control.dt
    # Periodic boundary condition
    if params.Control.PBC == 1:
        EnforcePBC(ptcls.pos,params.Lv)
        
    # Compute total potential energy and accleration for second half step velocity update                 
    U = PotentialAcceleration(ptcls,params)
    
    #Second half step velocity update
    ptcls.vel = ptcls.vel + 0.5*ptcls.acc*params.Control.dt

    return U


@nb.njit
def EnforcePBC(pos, BoxVector):
    """ Enforce Periodic Boundary Conditions. 

    Parameters
    ----------
    pos : array
          particles' positions = ptcls.pos

    BoxVector : array
                Box Dimensions

    Return
    ------

    none
    
    """

    # Loop over all particles
    for i in np.arange(pos.shape[0]):
        for p in np.arange(pos.shape[1]):
            
            # If particle is outside of box in positive direction, wrap to negative side
            if pos[i, p] > BoxVector[p]:
                pos[i, p] = pos[i, p] - BoxVector[p]
            
            # If particle is outside of box in negative direction, wrap to positive side
            if pos[i, p] < 0.0:
                pos[i, p] = pos[i, p] + BoxVector[p]

    
def PotentialAcceleration(ptcls,params):
    """ Calculate the Potential and update particles' accelerations.

    Parameter
    ---------
    ptcls : class
            Particles' data. See S_particles for more information

    params : class
             Simulations data. See S_params for more information

    Return
    ------
    U : float
        Potential

    """
    
    if(params.Potential.LL_on):
        U_short, acc_s_r = force_pp.update(ptcls,params)
    else:
        U_short, acc_s_r = force_pp.update_0D(ptcls,params)
    
    ptcls.acc = acc_s_r

    U = U_short

    #if (params.P3M_flag):
    #    U_long, acc_l_r = force_pm.update(ptcls,params)
    #    U = U + U_long
    #    ptcls.acc = ptcls.acc + acc_l_r
    
    return U