"""
Module containing the basic class for handling the plasma's components.
"""

from copy import deepcopy
from numpy import array, ndarray, pi, sqrt, zeros

from .utilities.fdints import fdm1h, invfd1h


class Species:
    """
    Class used to store all the information of a single species.

    Attributes
    ----------
    name : str
        Species' name.

    number_density : float
        Species number density in appropriate units.

    num : int
        Number of particles of Species.

    mass : float
        Species' mass.

    charge : float
        Species' charge.

    Z : float
        Species charge number or ionization degree.

    ai : float
        Species Wigner - Seitz radius calculate from electron density.
        Used in the calculation of the effective coupling constant.

    ai_dens : float
        Species Wigner - Seitz radius calculate from the species density.

    coupling : float
        Species coupling constant

    plasma_frequency : float
        Species' plasma frequency.

    debye_length : float
        Species' Debye Length.

    cyclotron_frequency : float
        Species' cyclotron frequency.

    initial_velocity : numpy.ndarray
        Initial velocity in x,y,z directions.

    temperature : float
        Initial temperature of the species.

    temperature_eV : float
        Initial temperature of the species in eV.

    initial_velocity_distribution : str
        Type of distribution. Default = 'boltzmann'.

    initial_spatial_distribution : str
        Type of distribution. Default = 'uniform'.

    atomic_weight : float
        (Optional) Species mass in atomic units.

    concentration : float
        Species' concentration.

    mass_density : float
        (Optional) Species' mass density.


    """

    def __init__(self, input_dict: dict = None):
        """
        Parameters
        ----------
        input_dict: dict, optional
            Dictionary with species information.

        """
        self.ai = None
        self.ai_dens = None
        self.atomic_weight = None
        self.charge = None
        self.Z = None
        self.coupling = None
        self.concentration = None
        self.debye_length = None
        self.initial_spatial_distribution = "random_no_reject"
        self.initial_velocity_distribution = "boltzmann"
        self.initial_velocity = zeros(3)
        self.mass = None
        self.mass_density = None
        self.name = None
        self.num = None
        self.number_density = None
        self.cyclotron_frequency = None
        self.plasma_frequency = None
        self.temperature = None
        self.temperature_eV = None

        if input_dict:
            self.from_dict(input_dict)

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Species( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def __copy__(self):
        """Make a shallow copy of the object using copy by creating a new instance of the object and copying its __dict__."""
        # Create a new object
        _copy = type(self)()
        # copy the dictionary
        _copy.from_dict(input_dict=self.__dict__)
        return _copy

    def __deepcopy__(self, memodict: dict = {}):
        """Make a deepcopy of the object.

        Parameters
        ----------
        memodict: dict
            Dictionary of id's to copies

        Returns
        -------
        _copy: :class:`sarkas.plasma.Species`
            A new Species class.
        """
        id_self = id(self)  # memorization avoids unnecessary recursion
        _copy = memodict.get(id_self)
        if _copy is None:

            # Make a shallow copy of all attributes
            _copy = type(self)()
            # Make a deepcopy of the mutable arrays using numpy copy function
            for k, v in self.__dict__.items():
                _copy.__dict__[k] = deepcopy(v, memodict)

        return _copy

    def from_dict(self, input_dict: dict):
        """
        Update attributes from input dictionary using the ``__dict__.update()`` method.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """
        self.__dict__.update(input_dict)

        if not isinstance(self.initial_velocity, ndarray):
            self.initial_velocity = array(self.initial_velocity)

    def copy_params(self, params):
        """
        Copy physical constants from parameters class.

        Parameters
        ----------
        params: :class:`sarkas.core.Parameters`
            Parameters class.

        """
        self.kB = params.kB
        self.eV2K = params.eV2K
        self.eV2J = params.eV2J
        self.hbar = params.hbar
        self.c0 = params.c0

        self.dimensions = params.dimensions
        # Charged systems: Electrostatic constant  :math:`4 \\pi \\epsilon_0` [mks]
        # Neutral systems: :math:`1/n\\sigma^2`
        if hasattr(self, "sigma"):
            self.fourpie0 = 4.0 * pi * self.number_density * self.sigma**2
        else:
            self.fourpie0 = params.fourpie0

    def calc_plasma_frequency(self):
        """Calculate the plasma frequency."""
        if self.dimensions == 3:
            self.plasma_frequency = sqrt(4.0 * pi * self.charge**2 * self.number_density / (self.mass * self.fourpie0))
        else:
            self.plasma_frequency = sqrt(
                2.0 * pi * self.charge**2 * self.number_density / (self.ai_dens * self.mass * self.fourpie0)
            )

    def calc_debye_length(self):
        """Calculate the Debye Length."""
        self.debye_length = sqrt(
            (self.temperature * self.kB * self.fourpie0) / (4.0 * pi * self.charge**2 * self.number_density)
        )

    def calc_debroglie_wavelength(self):
        """Calculate the de Broglie wavelength."""
        self.deBroglie_wavelength = sqrt(2.0 * pi * self.hbar**2 / (self.kB * self.temperature * self.mass))

    def calc_cyclotron_frequency(self, magnetic_field_strength: float):
        """
        Calculate the cyclotron frequency. See `Wikipedia link <https://en.wikipedia.org/wiki/Lorentz_force>`_.

        Parameters
        ----------
        magnetic_field_strength : float
            Magnetic field strength.

        """

        self.cyclotron_frequency = abs(self.charge) * magnetic_field_strength / self.mass

    def calc_landau_length(self):
        """Calculate the Landau Length."""
        # Landau length 4pi e^2 beta. The division by fourpie0 is needed for MKS units
        self.landau_length = 4.0 * pi * self.charge**2 / (self.temperature * self.fourpie0 * self.kB)

    def calc_coupling(self, a_ws: float, z_avg: float, const: float):
        """
        Calculate the coupling constant between particles.

        Parameters
        ----------
        a_ws : float
            Total Wigner-Seitz radius.

        z_avg : float
            Average charge of the system.

        const : float
            Electrostatic * Thermal constants.

        """
        self.ai = (self.charge / z_avg) ** (1.0 / 3.0) * a_ws if z_avg > 0 else self.ai_dens
        self.coupling = self.charge**2 / (self.ai * const * self.temperature)

    def calc_ws_radius(self):
        if self.dimensions == 3:
            self.ai_dens = (3.0 / (4.0 * pi * self.number_density)) ** (1.0 / 3.0)
        else:
            self.ai_dens = 1.0 / sqrt(pi * self.number_density)

    def calc_quantum_attributes(self, spin_statistics: str = "fermi-dirac"):
        """
        Calculate the following quantum parameters:
        Dimensionless chemical potential, Fermi wavenumber, Fermi energy, Thomas-Fermi wavenumber.

        This function work only for electrons. Other species might give nonsensical values.

        Parameters
        ----------
        spin_statistics: str
            Spin statistic. Default and only option "fermi-dirac".

        Raises
        ------
        `ValueError`:
            If not Fermi-Dirac statistics.

        """

        if spin_statistics.lower() != "fermi-dirac":
            raise ValueError(f"{spin_statistics} spin_statistic not implemented yet.")

        self.spin_degeneracy = 2.0
        lambda3 = self.deBroglie_wavelength**3

        # chemical potential mu/(kB T), obtained by inverting the density equation.
        self.dimensionless_chemical_potential = invfd1h(lambda3 * sqrt(pi) * self.number_density / 4.0)

        # Thomas-Fermi length obtained from compressibility. See eq.(10) in Ref. [3]_
        lambda_TF_sq = lambda3 / self.landau_length
        lambda_TF_sq /= self.spin_degeneracy / sqrt(pi) * fdm1h(self.dimensionless_chemical_potential)
        self.ThomasFermi_wavelength = sqrt(lambda_TF_sq)

        # Fermi wave number
        self.Fermi_wavenumber = (3.0 * pi**2 * self.number_density) ** (1.0 / 3.0)

        # Fermi energy
        self.Fermi_energy = self.hbar**2 * self.Fermi_wavenumber**2 / (2.0 * self.mass)

    def pretty_print(self, potential_type: str = None, units: str = "mks"):
        """Print Species information in a user-friendly way.

        Parameters
        ----------
        potential_type: str
            Interaction potential. If 'LJ' it will print out the epsilon and sigma attributes.

        units: str
            Unit system used in the simulation. Default = 'mks'.

        """
        # Pre compute the units to be printed
        if units == "cgs":
            density_units = "[N/cc]" if self.dimensions == 3 else "[N/cm^2]"
            weight_units = "[g]"
            mass_dens_units = "[g/cc]" if self.dimensions == 3 else "[g/cm^2]"
            charge_units = "[esu]"
            energy_units = "[erg]"
            length_units = "[cm]"
            inv_length_units = "[1/cm]"
        else:
            density_units = "[N/m^3]" if self.dimensions == 3 else "[N/m^2]"
            weight_units = "[kg]"
            mass_dens_units = "[kg/cc]" if self.dimensions == 3 else "[kg/m^2]"
            charge_units = "[C]"
            energy_units = "[J]"
            length_units = "[m]"
            inv_length_units = "[1/m]"

        # Assemble the entire message to be printed
        if self.name == "electron_background":
            kf_xf = self.mass * self.c0**2 * (sqrt(1.0 + self.relativistic_parameter**2) - 1.0)
            mu_EF = self.dimensionless_chemical_potential * self.kB * self.temperature / self.Fermi_energy

            if self.cyclotron_frequency:
                b_ef = self.magnetic_energy / self.Fermi_energy
                b_t = self.magnetic_energy / (self.kB * self.temperature)

                elec_mag_msg = (
                    f"Electron cyclotron frequency: w_c = {self.cyclotron_frequency:.6e}\n"
                    f"Lowest Landau energy level: h w_c/2 = {0.5 * self.magnetic_energy:.6e}\n"
                    f"Electron magnetic energy gap: h w_c = {self.magnetic_energy:.6e} = {b_ef:.4e} E_F = {b_t:.4e} k_B T_e\n"
                )

            else:
                elec_mag_msg = ""

            msg = (
                f"ELECTRON BACKGROUND PROPERTIES:\n"
                f"Number density: n_e = {self.number_density:.6e} {density_units}\n"
                f"Wigner-Seitz radius: a_e = {self.a_ws:.6e} {length_units}\n"
                f"Temperature: T_e = {self.temperature:.6e} [K] = {self.temperature_eV:.6e} [eV]\n"
                f"de Broglie wavelength: lambda_deB = {self.deBroglie_wavelength:.6e} {length_units}\n"
                f"Thomas-Fermi length: lambda_TF = {self.ThomasFermi_wavelength:.6e}{length_units}\n"
                f"Fermi wave number: k_F = {self.Fermi_wavenumber:.6e} {inv_length_units}\n"
                f"Fermi Energy: E_F = {self.Fermi_energy / self.kB / self.eV2K:.6e} [eV]\n"
                f"Relativistic parameter: x_F = {self.relativistic_parameter:.6e} --> E_F = {(kf_xf / self.kB / self.eV2K):.6e} [eV]\n"
                f"Degeneracy parameter: Theta = {self.degeneracy_parameter:.6e}\n"
                f"Coupling: r_s = {self.rs:.6f},  Gamma_e = {self.coupling:.6f}\n"
                f"Warm Dense Matter parameter: W = {self.wdm_parameter:.4e}\n"
                f"Chemical potential: mu = {self.dimensionless_chemical_potential:.4e} k_B T_e = {mu_EF:.4e} E_F\n"
            )
            msg += elec_mag_msg

        else:
            if potential_type == "lj":
                pot_msg = (
                    f"\tEpsilon = {self.epsilon:.6e} {energy_units}\n"
                    f"\tSigma = {self.sigma:.6e} {length_units}\n"
                    f"\tEquivalent Plasma frequency = {self.plasma_frequency: .6e} [rad/s]\n"
                )
            else:
                pot_msg = (
                    f"\tDebye length = {self.debye_length: .6e} {length_units}\n"
                    f"\tPlasma frequency = {self.plasma_frequency: .6e} [rad/s]\n"
                )
            if self.cyclotron_frequency:
                mag_msg = (
                    f"\tCyclotron Frequency = {self.cyclotron_frequency:.6e} [rad/s]\n"
                    f"\tbeta_c = {self.cyclotron_frequency / self.plasma_frequency:.4e}"
                )
            else:
                mag_msg = ""

            msg = (
                f"\tName: {self.name}\n"
                f"\tNo. of particles = {self.num}\n"
                f"\tNumber density = {self.number_density:.6e} {density_units}\n"
                f"\tAtomic weight = {self.atomic_weight:.6e} [a.u.]\n"
                f"\tMass = {self.mass:.6e} {weight_units}\n"
                f"\tMass density = {self.mass_density:.6e} {mass_dens_units}\n"
                f"\tCharge number/ionization degree = {self.Z:.4f}\n"
                f"\tCharge = {self.charge:.6e} {charge_units}\n"
                f"\tTemperature = {self.temperature:.6e} [K] = {self.temperature_eV:.6e} [eV]\n"
            )
            msg = msg + pot_msg + mag_msg
        print(msg)
