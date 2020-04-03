"""
Module containing various thermostat. Berendsen only for now.
"""
import numpy as np
import numba as nb

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
    ptcls : class
        Particles's data. See ``S_particles.py`` for more information.
    
    params : class
        Simulation parameters. See ``S_params.py`` for more information.

    it : int
        Timestep.
    
    References
    ----------
    .. [1] `H.J.C. Berendsen et al., J Chem Phys 81 3684 (1984) <https://doi.org/10.1063/1.448118>`_ 

    """
    # Dev Notes: this could be Numba'd
    K, T = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
    species_start = 0
    species_end = 0
    for i in range(params.num_species):
        species_end = species_start + params.species[i].num

        if it <= params.Thermostat.timestep:
            fact = np.sqrt(params.T_desired / T[i])
        else:
            fact = np.sqrt(1.0 + (params.T_desired / T[i] - 1.0) / params.Thermostat.tau)  # eq.(11)

        ptcls.vel[species_start:species_end, :] *= fact
        species_start = species_end

    return


@nb.njit
def calc_kin_temp(vel, nums, masses, kB):
    """ 
    Calculates the kinetic energy and temperature.

    Parameters
    ----------
    kB: float
        Boltzmann constant in chosen units.

    masses: array
        Mass of each species.

    nums: array
        Number of particles of each species.

    vel: array
        Particles' velocities.

    Returns
    -------
    K : array
        Kinetic energy of each species.

    T : array
        Temperature of each species.
    """

    num_species = len(masses)

    K = np.zeros(num_species)
    T = np.zeros(num_species)

    species_start = 0
    species_end = 0
    for i in range(num_species):
        species_end = species_start + nums[i]
        K[i] = 0.5 * masses[i] * np.sum(vel[species_start:species_end, :] ** 2)
        T[i] = (2.0 / 3.0) * K[i] / kB / nums[i]
        species_start = species_end

    return K, T


@nb.njit
def remove_drift(vel, nums, masses):
    """
    Enforce conservation of total linear momentum. Updates ``ptcls.vel``

    Parameters
    ----------
    vel: array
        Particles' velocities.

    nums: array
        Number of particles of each species.

    masses: array
        Mass of each species.

    """
    P = np.zeros((len(nums), vel.shape[1]))

    species_start = 0
    species_end = 0

    for ic in range(len(nums)):
        species_end = species_start + nums[ic]
        P[ic, :] = np.sum(vel[species_start:species_end, :], axis=0) * masses[ic]
        species_start = species_end

    if np.sum(P[:, 0]) > 1e-40 or np.sum(P[:, 1]) > 1e-40 or np.sum(P[:, 2]) > 1e-40:
        # Remove tot momentum
        species_start = 0
        for ic in range(len(nums)):
            species_end = species_start + nums[ic]
            vel[species_start:species_end, :] -= P[ic, :] / (float(nums[ic]) * masses[ic])
            species_start = species_end

    return
