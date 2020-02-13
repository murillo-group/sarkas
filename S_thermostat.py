'''
S_thermostat.py

Berendsen only.
'''
import numpy as np
import sys
import scipy.constants as const
'''
class Thermostat:
    def __init__(self, params):
        self.integrator = Integrator(params)

        self.params = params

        if(params.Thermostat.type == "Berendsen"):
            self.type = self.Berendsen
        else:
            print("Only Berendsen thermostat is supported. Check your input file, thermostat part.")
            sys.exit()

        if(params.Integrator.type == "Verlet"):
            self.integrator = self.integrator.Verlet
        else:
            print("Only Verlet integrator is supported. Check your input file, integrator part.")
            sys.exit()

    def update(self, ptcls, it):
        U = self.type(ptcls, it)
        return U
'''
def Berendsen(ptcls, params, it):
    """ 
    Update particle velocity based on Berendsen thermostat.
    
    Parameters
    ----------
    ptlcs : class
        Particles's data. See S_particles.py for the detailed information
    
    params : class
        Simulation parameters. See S_params for detail information

    it : int
        timestep
    
    Returns
    -------
    vel : array_like
        Thermostated particles' velocities
    
    References
    ---------
    [1] Berendsen et al. J Chem Phys 81 3684 (1984)

    """
    K, T = calc_kin_temp(ptcls, params)
    species_start = 0
    species_end = 0
    for i in range(params.num_species):
        species_end = species_start + params.species[i].num
        
        if (it <= params.Thermostat.timestep):
            fact = np.sqrt(params.T_desired/T[i])
        else:
            fact = np.sqrt( 1.0 + (params.T_desired/T[i] - 1.0)/params.Thermostat.tau)  # eq.(11)

        ptcls.vel[species_start:species_end,:] = ptcls.vel[species_start:species_end,:]*fact
        species_start = species_end

    return

def calc_kin_temp(ptcls,params):
    """ 
    Calculate the kinetic energy and temperature

    Parameters
    ----------
    ptlcs : class
        Particles's data. See S_particles.py for the detailed information
    
    params : class
        Simulation's parameters. See S_params for detail information

    Returns
    -------
    K : array_like
        Kinetic energy of each species

    T : array_like
        Temperature of each species

    Notes
    -----
    """

    K = np.zeros( params.num_species)
    T = np.zeros( params.num_species)

    species_start = 0
    species_end = 0
    for i in range(params.num_species):
        species_end = species_start + params.species[i].num
        K[i] = 0.5*params.species[i].mass*np.ndarray.sum(ptcls.vel[species_start:species_end, :]**2)
        T[i] = (2.0/3.0)*K[i]/params.kB/params.species[i].num
        species_start = species_end

    return K, T

def remove_drift(ptcls, params):
    """
    Enforce conservation of total linear momentum

    Parameters
    ----------
    vel: array_like
        Particles' velocities

    nums: array_like
        Number of particles for each species

    masses: array_like
        Mass of each species

    Returns
    -------
    vel: array_like
        Particles' velocities

    Notes
    -----

    """
    P = np.zeros( ( params.N , params.d ) )

    species_start = 0
    species_end = 0

    # Calculate total linear momentum
    for ic in range( params.num_species ):
        species_end = species_start + params.species[ic].num
        P[ic,:] = np.sum( ptcls.vel[species_start:species_end,:], axis = 0 )*params.species[ic].mass
        species_start = species_end

    if ( np.sum(P[:,0]) > 1e-39 or np.sum(P[:,1]) > 1e-39 or np.sum(P[:,2]) > 1e-39 ) : 
        # Remove tot momentum
        species_start = 0
        for ic in range( params.num_species ):
            species_end = species_start + params.species[ic].num
            ptcls.vel[species_start:species_end,:] -= P[ic,:]/(float(params.species[ic].num)*params.species[ic].mass )
            species_start = species_end
    
    return
