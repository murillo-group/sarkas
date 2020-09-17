"""
Module containing various thermostat. Berendsen only for now.
"""
import numpy as np
from numba import njit


class Thermostat:
    """
    Thermostat object.

    Attributes
    ----------
    kB : float
        Boltzmann constant in correct units.

    no_species : int
        Total number of species.

    species_np : numpy.ndarray
        Number of particles of each species.

    species_masses : numpy.ndarray
        Mass of each species.

    relaxation_rate: float
        Berendsen parameter tau.

    relaxation_timestep: int
        Timestep at which thermostat is turned on.

    type: str
        Thermostat type

    tau: float
        Berendsen parameter.

    """
    def __init__(self):
        self.temperatures = None
        self.temperatures_eV = None
        self.type = None
        self.relaxation_rate = None
        self.relaxation_timestep = None
        self.kB = None
        self.species_num = None
        self.species_masses = None
        self.tau = None
        self.eV_temp_flag = False
        self.K_temp_flag = False

    def from_dict(self, input_dict: dict) :
        """
        Update attributes from input dictionary.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """
        self.__dict__.update(input_dict)

        # Make sure list are turned into numpy arrays
        if self.temperatures_eV:
            if not isinstance(self.temperatures_eV, np.ndarray):
                self.temperatures_eV = np.array(self.temperatures_eV)
            self.eV_temp_flag = True

        if self.temperatures:
            if not isinstance(self.temperatures, np.ndarray):
                    self.temperatures = np.array(self.temperatures)
            self.K_temp_flag = True

    def setup(self, params):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        params: sarkas.base.Parameters
            Simulation's parameters

        """

        # Check whether you input temperatures in eV or K

        if self.eV_temp_flag:
            self.temperatures = params.eV2K * np.copy(self.temperatures_eV)
        elif self.K_temp_flag:
            self.temperatures_eV = np.copy(self.temperatures)/ params.eV2K
        else:
            # If you forgot to give thermostating temperatures
            print("WARNING: Equilibration temperatures not defined. I will use the species's temperatures.")
            self.temperatures = np.copy(params.species_temperatures)
            self.temperatures_eV = np.copy(self.temperatures)/ params.eV2K

        if self.tau:
            self.relaxation_rate = 1.0/self.tau

        if not self.temperatures.all():
            self.temperatures = np.copy(params.species_temperatures)

        self.kB = params.kB
        self.species_num = np.copy(params.species_num)
        self.species_masses = np.copy(params.species_masses)

        assert self.type.lower() == "berendsen", "Only Berendsen thermostat is supported."

    def update(self, ptcls, it):
        """
        Update particles' velocities according to the chosen thermostat

        Parameters
        ----------
        ptcls : sarkas.base.Particles
            Particles' data.

        it : int
            Current timestep.

        """
        K, T = ptcls.kinetic_temperature()
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

    masses: numpy.ndarray
        Mass of each species.

    nums: numpy.ndarray
        Number of particles of each species.

    vel: numpy.ndarray
        Particles' velocities.

    Returns
    -------
    K : numpy.ndarray
        Kinetic energy of each species.

    T : numpy.ndarray
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
    T : numpy.ndarray
        Instantaneous temperature of each species.

    vel : numpy.ndarray
        Particles' velocities to rescale.

    T_desired : numpy.ndarray
        Target temperature of each species.

    tau : float
        Scale factor.

    therm_timestep : int
        Timestep at which to turn on the thermostat.

    species_np : numpy.ndarray
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



