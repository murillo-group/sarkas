"""
Module containing various thermostat. Berendsen only for now.
"""
from warnings import warn
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

    berendsen_tau: float
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
        self.berendsen_tau = None
        self.eV_temp_flag = False
        self.K_temp_flag = False

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Thermostat( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def from_dict(self, input_dict: dict):
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
                self.temperatures_eV = np.array([self.temperatures_eV])
            self.eV_temp_flag = True

        if self.temperatures:
            if not isinstance(self.temperatures, np.ndarray):
                self.temperatures = np.array([self.temperatures])
            self.K_temp_flag = True

    def pretty_print(self):
        """Print Thermostat information in a user-friendly way."""
        print("Type: {}".format(self.type))
        print("First thermostating timestep, i.e. relaxation_timestep = {}".format(self.relaxation_timestep))
        print("Berendsen parameter tau: {:.3f} [timesteps]".format(self.berendsen_tau))
        print("Berendsen relaxation rate: {:.3f} [1/timesteps] ".format(self.relaxation_rate))
        # if not self.eV_temp_flag and not self.K_temp_flag:
        #     # If you forgot to give thermostating temperatures
        #     warn("Equilibration temperatures not defined. "
        #          "I will use the species's temperatures")
        print("Thermostating temperatures: ")
        for i, (t, t_ev) in enumerate(zip(self.temperatures, self.temperatures_eV)):
            print("Species ID {}: T_eq = {:.6e} [K] = {:.6e} [eV]".format(i, t, t_ev))

    def setup(self, params):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        params: sarkas.core.Parameters
            Simulation's parameters

        Raises
        ------
        ValueError
            If a thermostat different than Berendsen is chosen.

        """

        # Check whether you input temperatures in eV or K
        self.type = self.type.lower()

        if self.eV_temp_flag:
            self.temperatures = params.eV2K * np.copy(self.temperatures_eV)
        elif self.K_temp_flag:
            self.temperatures_eV = np.copy(self.temperatures) / params.eV2K
        else:
            self.temperatures = np.copy(params.species_temperatures)
            self.temperatures_eV = np.copy(self.temperatures) / params.eV2K

        if self.berendsen_tau:
            self.relaxation_rate = 1.0 / self.berendsen_tau
        else:
            self.berendsen_tau = 1.0 / self.relaxation_rate

        if not self.temperatures.all():
            self.temperatures = np.copy(params.species_temperatures)

        self.kB = params.kB
        self.species_num = np.copy(params.species_num)
        self.species_masses = np.copy(params.species_masses)

        if self.type != "berendsen":
            raise ValueError("Only Berendsen thermostat is supported.")

    def update(self, ptcls, it):
        """
        Update particles' velocities according to the chosen thermostat

        Parameters
        ----------
        ptcls : sarkas.core.Particles
            Particles' data.

        it : int
            Current timestep.

        """
        _, T = ptcls.kinetic_temperature()
        berendsen(ptcls.vel, self.temperatures, T, self.species_num, self.relaxation_timestep, self.relaxation_rate, it)


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

    # if it < therm_timestep:
    #     fact = np.sqrt(T_desired / T)
    # else:
    #     fact = np.sqrt(1.0 + (T_desired / T - 1.0) * tau)  # eq.(11)

    # branchless programming
    fact = 1.0 * (it < therm_timestep) + np.sqrt(1.0 + (T_desired / T - 1.0) * tau) * (it >= therm_timestep)
    species_start = 0
    species_end = 0

    for i, num in enumerate(species_np):
        species_end += num
        vel[species_start:species_end, :] *= fact[i]
        species_start += num
