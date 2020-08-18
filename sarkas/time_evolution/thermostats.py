"""
Module containing various thermostat. Berendsen only for now.
"""
import numpy as np
from numba import njit


class Thermostat:
    """
    Thermostat object.

    Parameters
    ----------
    params : object
        Simulation's parameters

    Attributes
    ----------
    kB : float
        Boltzmann constant in correct units.

    no_species : int
        Total number of species.

    species_np : array
        Number of particles of each species.

    species_masses : array
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
    def __init__(self):
        self.temperatures = None
        self.type = None
        self.relaxation_rate = None
        self.relaxation_timestep = None
        self.kB = None
        self.species_num = None
        self.species_masses = None

    def setup(self, params):

        # run some checks
        if hasattr(self, 'temperatures_eV'):
            self.temperatures = params.eV2K * self.temperatures_eV

        if not isinstance(self.temperatures, np.ndarray):
            self.temperatures = np.array(self.temperatures)

        if hasattr(self, "tau"):
            self.relaxation_rate = 1.0/self.tau

        self.kB = params.kB
        self.species_num = params.species_num
        self.species_masses = params.species_masses

        assert self.type.lower() == "berendsen", "Only Berendsen thermostat is supported."

    def update(self, ptcls, it):
        """
        Update particles' velocities according to the chosen thermostat

        Parameters
        ----------
        vel : ndarray
            Particles' velocities to be rescaled.

        it : int
            Current timestep.

        """
        K, T = ptcls.kinetic_temperature(self.kB)
        berendsen(ptcls.vel, self.temperatures, T, self.species_num, self.relaxation_timestep,
                              self.relaxation_rate, it)


@njit
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

    num_species = len(nums)

    K = np.zeros(num_species)
    T = np.zeros(num_species)
    const = 2.0 / (kB * nums * vel.shape[1])
    kinetic_energies = 0.5 * masses * (vel ** 2).transpose()

    species_start = 0
    species_end = 0
    for i, num in enumerate(nums):
        species_end += num
        K[i] = np.sum(kinetic_energies[:, species_start:species_end])
        T[i] = const[i] * K[i]
        species_start = species_end

    return K, T


@njit
def berendsen(vel, T_desired, T, species_np, therm_timestep, tau, it):
    """
    Update particle velocity based on Berendsen thermostat [Berendsen1984]_.

    Parameters
    ----------
    T : array
        Instantaneous temperature of each species.

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

    if it < therm_timestep:
        fact = np.sqrt(T_desired / T)
    else:
        fact = np.sqrt(1.0 + (T_desired / T - 1.0) * tau)  # eq.(11)

    species_start = 0
    species_end = 0

    for i, num in enumerate(species_np):
        species_end += num
        vel[species_start:species_end, :] *= fact[i]
        species_start = species_end



