'''
S_integrator.py

velocity integrator

Verlet: velocity Verlet
Verlet_with_Langevin: Verlet with Langevin damping
RK45: N/A
RK45_with_Langevin: N/A
'''

import numpy as np
import numba as nb
import sys
import S_calc_force_pp as force_pp
import S_calc_force_pm as force_pm
import S_constants as const

'''
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
'''
def Verlet(ptcls,params):
    ''' Update particle position and velocity based on velocity verlet method.
    More information can be found here: https://en.wikipedia.org/wiki/Verlet_integration
    or on the Sarkas website. 

    Parameters
    ----------
    ptlcs: particles data. See S_particles.py for the detailed information. 
    Returns
    -------
    U : float
        Total potential energy
    '''
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
    """ Calculate the Potential and update particle's accelerations.

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
        U_short, acc_s_r = force_pp.update(ptcls, params)
    else:
        U_short, acc_s_r = force_pp.update_0D(ptcls, params)
    
    ptcls.acc = acc_s_r

    U = U_short

    #if (params.P3M_flag):
    #    U_long, acc_l_r = force_pm.update(ptcls,params)
    #    U = U + U_long
    #    ptcls.acc = ptcls.acc + acc_l_r
    
    return U