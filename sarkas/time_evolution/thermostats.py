"""
Module containing various thermostat. Berendsen only for now.
"""

from copy import deepcopy
from numba import float64, int64, jit, void
from numpy import array, ndarray, sqrt
from scipy.constants import physical_constants


class Thermostat:
    """
    Thermostat object.

    Attributes
    ----------
    berendsen_tau: float
        Berendsen parameter.

    eV2K : float
        Conversion factor from eV to Kelvin.

    eV_temp_flag: bool
        Flag if the input temperatures are in eV. Default = False.

    K_temp_flag: bool
        Flag if the input temperatures are in K. Default = False.

    kB : float
        Boltzmann constant in correct units. Default is in J/K.

    relaxation_rate: float
        Berendsen parameter tau.

    relaxation_timestep: int
        Timestep at which thermostat is turned on.

    species_num : numpy.ndarray
        Number of particles of each species. Copy of :attr:`sarkas.core.Parameters.species_num`.

    species_masses : numpy.ndarray
        Mass of each species. Copy of :attr:`sarkas.core.Parameters.species_masses`.

    temperatures: numpy.ndarray
        Array of equilibration temperatures.

    temperatures_eV: numpy.ndarray
        Array of equilibration temperatures in eV.

    type: str
        Thermostat type.
    """

    berendsen_tau: float = None
    eV2K: float = physical_constants["electron volt-kelvin relationship"][0]
    eV_temp_flag: bool = False
    K_temp_flag: bool = False
    kB: float = physical_constants["Boltzmann constant"][0]
    relaxation_rate: int = None
    relaxation_timestep: int = None
    species_num: ndarray = None
    species_masses: ndarray = None
    temperatures = None
    temperatures_eV = None
    type: str = None

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Thermostat( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def __copy__(self):
        """Make a shallow copy of the object using copy by creating a new instance of the object and copying its __dict__."""
        # Create a new object
        _copy = type(self)()
        # copy the dictionary
        _copy.__dict__.update(self.__dict__)
        return _copy

    def __deepcopy__(self, memodict: dict = {}):
        """Make a deepcopy of the object.

        Parameters
        ----------
        memodict: dict
            Dictionary of id's to copies

        Returns
        -------
        _copy: :class:`sarkas.time_evolution.thermostat.Thermostat`
            A new Thermostat class.

        """
        id_self = id(self)  # memorization avoids unnecessary recursion
        _copy = memodict.get(id_self)
        if _copy is None:
            _copy = type(self)()
            _copy.temperatures = self.temperatures.copy()
            _copy.temperatures_eV = self.temperatures_eV.copy()
            _copy.type = deepcopy(self.type, memodict)
            _copy.relaxation_rate = deepcopy(self.relaxation_rate, memodict)
            _copy.relaxation_timestep = deepcopy(self.relaxation_timestep, memodict)
            _copy.species_num = self.species_num.copy()
            _copy.species_masses = self.species_masses.copy()
            _copy.berendsen_tau = deepcopy(self.berendsen_tau, memodict)
            _copy.eV_temp_flag = deepcopy(self.eV_temp_flag, memodict)
            _copy.K_temp_flag = deepcopy(self.K_temp_flag, memodict)
            _copy.kB = deepcopy(self.kB)
            memodict[id_self] = _copy
        return _copy

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
        # TODO: There should be a better way to check if the user passed eV or K
        if self.temperatures_eV:
            if not isinstance(self.temperatures_eV, ndarray):
                self.temperatures_eV = array([self.temperatures_eV])
            self.eV_temp_flag = True

        if self.temperatures:
            if not isinstance(self.temperatures, ndarray):
                self.temperatures = array([self.temperatures])
            self.K_temp_flag = True

    def pretty_print(self):
        """Print Thermostat information in a user-friendly way."""
        print("\nTHERMOSTAT: ")
        print(f"Type: {self.type}")
        print(f"First thermostating timestep, i.e. relaxation_timestep = {self.relaxation_timestep}")
        print(f"Berendsen parameter tau: {self.berendsen_tau:.3f} [timesteps]")
        print(f"Berendsen relaxation rate: {self.relaxation_rate:.3f} [1/timesteps] ")
        # if not self.eV_temp_flag and not self.K_temp_flag:
        #     # If you forgot to give thermostating temperatures
        #     warn("Equilibration temperatures not defined. "
        #          "I will use the species's temperatures")
        print("Thermostating temperatures: ")
        for i, (t, t_ev) in enumerate(zip(self.temperatures, self.temperatures_eV)):
            print(f"Species ID {i}: T_eq = {t:.6e} [K] = {t_ev:.6e} [eV]")

    def setup(self, params):
        """
        Assign attributes from simulation's parameters.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            Simulation's parameters

        Raises
        ------
        ValueError
            If a thermostat different than Berendsen is chosen.

        """

        # Check whether you input temperatures in eV or K
        self.type = self.type.lower()

        if self.eV_temp_flag:
            self.temperatures = self.eV2K * self.temperatures_eV.copy()
        elif self.K_temp_flag:
            self.temperatures_eV = self.temperatures.copy() / self.eV2K
        else:
            self.temperatures = params.species_temperatures.copy()
            self.temperatures_eV = self.temperatures.copy() / self.eV2K

        if self.berendsen_tau:
            self.relaxation_rate = 1.0 / self.berendsen_tau
        else:
            self.berendsen_tau = 1.0 / self.relaxation_rate

        if not self.temperatures.all():
            self.temperatures = params.species_temperatures.copy()

        self.kB = params.kB
        self.species_num = params.species_num.copy()
        self.species_masses = params.species_masses.copy()

        if self.type != "berendsen":
            raise ValueError("Only Berendsen thermostat is supported.")

    def update(self, ptcls, it):
        """
        Update particles' velocities according to the chosen thermostat

        Parameters
        ----------
        ptcls : :class:`sarkas.particles.Particles`
            Particles' data.

        it : int
            Current timestep.

        """
        _, T = ptcls.kinetic_temperature()
        berendsen(ptcls.vel, self.temperatures, T, self.species_num, self.relaxation_timestep, self.relaxation_rate, it)


@jit(void(float64[:, :], float64[:], float64[:], int64[:], int64, float64, int64), nopython=True)
def berendsen(vel, T_desired, T, species_np, therm_timestep, tau, it):
    """
    Numba'd function to update particle velocity based on Berendsen thermostat :cite:`Berendsen1984`.

    Parameters
    ----------
    vel : numpy.ndarray
        Particles' velocities to rescale.

    T_desired : numpy.ndarray
        Target temperature of each species.

    T : numpy.ndarray
        Instantaneous temperature of each species.

    species_np : numpy.ndarray
        Number of each species.

    therm_timestep : int
        Timestep at which to turn on the thermostat.

    tau : float
        Scale factor.

    it : int
        Current timestep.

    """

    # if it < therm_timestep:
    #     fact = sqrt(T_desired / T)
    # else:
    #     fact = sqrt(1.0 + (T_desired / T - 1.0) * tau)  # eq.(11)

    # branchless programming
    fact = 1.0 * (it < therm_timestep) + sqrt(1.0 + (T_desired / T - 1.0) * tau) * (it >= therm_timestep)
    species_start = 0
    species_end = 0

    for i, num in enumerate(species_np):
        species_end += num
        vel[species_start:species_end, :] *= fact[i]
        species_start += num
