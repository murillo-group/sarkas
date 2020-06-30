"""
Module containing various thermostat. Berendsen only for now.
"""
import numpy as np
import numba as nb


class Thermostat:
    """
    Parameters
    ----------

    params : class

    Attributes
    ----------
    kB : float
        Boltzmann constant in correct units.

    no_species : int
        Total number of species.

    species_np : array
        Number of particles of each species.

    species_mass : array
        Mass of each species.

    relaxation_rate: float
        Berendsen parameter tau.

    relaxation_timestep: int
        Timestep at which thermostat is turned on.

    T_desired: array
        Thermostating temperature of each species.

    type: str
        Thermostat type

    """
    def __init__(self, params):
        if params.Thermostat.on:
            self.kB = params.kB
            self.no_species = len(params.species)
            self.species_np = np.zeros(self.no_species)
            self.species_masses = np.zeros(self.no_species)
            self.relaxation_timestep = params.Thermostat.timestep
            self.relaxation_rate = params.Thermostat.tau
            self.T_desired = params.Thermostat.temperatures

            for i in range(self.no_species):
                self.species_np[i] = params.species[i].num
                self.species_masses[i] = params.species[i].mass

            if params.Thermostat.type == "Berendsen":
                self.type = Berendsen
            else:
                raise AttributeError("Only Berendsen thermostat is supported. Check your input file, thermostat part.")
        else:
            pass

    def update(self, vel, it):
        """
        Update particles' velocities according to the chosen thermostat

        Parameters
        ----------
        vel : ndarray
            Particles' velocities to be rescaled.

        it : int
            Current timestep.

        """
        K, T = calc_kin_temp(vel, self.species_np, self.species_masses, self.kB)
        self.type(vel, self.T_desired, T, self.species_np, self.relaxation_timestep, self.relaxation_rate, it)


@nb.njit
def Berendsen(vel, T_desired, T, species_np, therm_timestep, tau, it):
    """
    Update particle velocity based on Berendsen thermostat [Berendsen1984]_.

    Parameters
    ----------
    T : array
        Temperature of each species.

    vel : array
        Particles' velocities to rescale.

    T_desired : array
        Target temperature of each species.

    tau : float
        Scale factor.

    therm_timestep : int
        Timestep at which to turn on the thermostat.

    species_np : array
        Number of each species.

    it : int
        Current timestep.

    References
    ----------
    .. [Berendsen1984] `H.J.C. Berendsen et al., J Chem Phys 81 3684 (1984) <https://doi.org/10.1063/1.448118>`_

    """
    # Dev Notes: this could be Numba'd
    species_start = 0
    for i in range(len(species_np)):
        species_end = species_start + species_np[i]

        if it <= therm_timestep:
            fact = np.sqrt(T_desired[i] / T[i])
        else:
            fact = np.sqrt(1.0 + (T_desired[i] / T[i] - 1.0) / tau)  # eq.(11)

        vel[species_start:species_end, :] *= fact
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

