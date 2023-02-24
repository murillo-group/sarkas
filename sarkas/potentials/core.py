"""
Module handling the potential class.
"""
from copy import deepcopy
from fmm3dpy import hfmm3d, lfmm3d
from numpy import array, inf, int64, ndarray, pi, sqrt, tanh
from warnings import warn

from ..utilities.exceptions import AlgorithmWarning
from ..utilities.fdints import fdm1h, invfd1h
from .force_pm import force_optimized_green_function as gf_opt
from .force_pm import update as pm_update
from .force_pp import update as pp_update
from .force_pp import update_0D as pp_update_0D


class Potential:
    r"""
    Parameters specific to potential choice.

    Attributes
    ----------
    a_rs : float
        Short-range cutoff to deal with divergence of the potential for r -> 0.

    box_lengths : array
        Pointer to :attr:`sarkas.core.Parameters.box_lengths`.

    box_volume : float
        Pointer to :attr:`sarkas.core.Parameters.box_volume`.

    force_error : float
        Force error due to the choice of the algorithm.

    fourpie0 : float
        Coulomb constant :math:`4 \pi \epsilon_0`.

    kappa : float
        Inverse screening length.

    linked_list_on : bool
        Flag for choosing the Linked cell list algorithm.

    matrix : numpy.ndarray
        Matrix of potential's parameters.

    measure : bool
        Flag for calculating the histogram for the radial distribution function.
        It is set to `False` during equilibration phase and changed to `True` during production phase.

    method : str
        Algorithm method. Choices = `["PP", "PPPM", "FMM", "Brute"]`. \n
        `"PP"` = Linked Cell List (default).
        `"PPPM"` = Particle-Particle Particle-Mesh.
        `"FMM"` = Fast Multipole Method.
        `"Brute"` = corresponds to calculating the distance between all pair of particles within a distance :math:`L/2`.

    pbox_lengths : numpy.ndarray
        Pointer to :attr:`sarkas.core.Parameters.pbox_lengths`

    pbox_volume : float
        Pointer to :attr:`sarkas.core.Parameters.pbox_lengths`

    pppm_on : bool
        Flag for turning on the PPPM algorithm.

    QFactor : float
        Sum of the squared of the charges.

    rc : float
        Cutoff radius for the Linked Cell List algorithm.

    screening_length_type : str
        Choice of ways to calculate the screening length. \n
        Choices = `[thomas-fermi, tf, debye, debye-huckel, db, moliere, custom, unscreened]`. \n
        Default = thomas-fermi

    screening_length : float
        Value of the screening length.

    total_net_charge : float
        Sum of all the charges.

    type : str
        Type of potential. \n
        Choices = [`"coulomb"`, `"egs"`, `"lennardjones"`, `"moliere"`, `"qsp"`].

    """

    a_rs: float = 0.0
    box_lengths: ndarray = None
    box_volume: float = 0.0
    force_error: float = 0.0
    fourpie0: float = 0.0
    kappa: float = None
    linked_list_on: bool = True
    matrix: ndarray = None
    measure: bool = False
    method: str = "pp"
    pbox_lengths: ndarray = None
    pbox_volume: float = 0.0
    pppm_on: bool = False
    pppm_aliases: ndarray = array([3, 3, 3], dtype=int64)
    pppm_alpha_ewald: float = 0.0
    pppm_cao: ndarray = array([3, 3, 3], dtype=int64)
    pppm_mesh: ndarray = array([8, 8, 8], dtype=int64)
    pppm_h_array: ndarray = array([1.0, 1.0, 1.0], dtype=float)
    pppm_pm_err: float = 0.0
    pppm_pp_err: float = 0.0
    QFactor: float = 0.0
    rc: float = None
    num_species: ndarray = None
    screening_length_type: str = "thomas-fermi"
    screening_length: float = None
    species_charges: ndarray = None
    species_masses: ndarray = None
    total_net_charge: float = 0.0
    total_num_density: float = 0.0
    total_num_ptcls: float = 0.0
    type: str = "yukawa"

    def __copy__(self):
        """
        Make a shallow copy of the object using copy by creating a new instance of the object and copying its __dict__.
        """
        # Create a new object
        _copy = type(self)()
        # copy the dictionary
        _copy.from_dict(input_dict=self.__dict__)
        return _copy

    def __deepcopy__(self, memodict={}):
        """
        Make a deepcopy of the object.

        Parameters
        ----------
        memodict: dict
            Dictionary of id's to copies

        Returns
        -------
        _copy: :class:`sarkas.potentials.core.Potential`
            A new Potential class.
        """
        id_self = id(self)  # memorization avoids unnecessary recursion
        _copy = memodict.get(id_self)
        if _copy is None:
            _copy = type(self)()
            # Make a deepcopy of the mutable arrays using numpy copy function
            for k, v in self.__dict__.items():
                _copy.__dict__[k] = deepcopy(v, memodict)

        return _copy

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Potential( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    @staticmethod
    def calc_electron_properties(params):
        """Calculate electronic parameters.
        See Electron Properties webpage in documentation website.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            Simulation's parameters.

        """

        warn(
            "Deprecated feature. It will be removed in the v2.0.0 release. \n"
            "Use parameters.calc_electron_properties(species). You need to pass the species list.",
            category=DeprecationWarning,
        )

        twopi = 2.0 * pi
        spin_degeneracy = 2.0  # g in the notes

        # Inverse temperature for convenience
        beta_e = 1.0 / (params.kB * params.electron_temperature)

        # Plasma frequency
        params.electron_plasma_frequency = sqrt(
            4.0 * pi * params.qe**2 * params.electron_number_density / (params.fourpie0 * params.me)
        )

        params.electron_debye_length = sqrt(
            params.fourpie0 / (4.0 * pi * params.qe**2 * params.electron_number_density * beta_e)
        )

        # de Broglie wavelength
        params.electron_deBroglie_wavelength = sqrt(twopi * params.hbar2 * beta_e / params.me)
        lambda3 = params.electron_deBroglie_wavelength**3

        # Landau length 4pi e^2 beta. The division by fourpie0 is needed for MKS units
        params.electron_landau_length = 4.0 * pi * params.qe**2 * beta_e / params.fourpie0

        # chemical potential of electron gas/(kB T), obtained by inverting the density equation.
        params.electron_dimensionless_chemical_potential = invfd1h(
            lambda3 * sqrt(pi) * params.electron_number_density / 4.0
        )

        # Thomas-Fermi length obtained from compressibility. See eq.(10) in Ref. [3]_
        lambda_TF_sq = lambda3 / params.electron_landau_length
        lambda_TF_sq /= spin_degeneracy / sqrt(pi) * fdm1h(params.electron_dimensionless_chemical_potential)
        params.electron_TF_wavelength = sqrt(lambda_TF_sq)

        # Electron WS radius
        params.electron_WS_radius = (3.0 / (4.0 * pi * params.electron_number_density)) ** (1.0 / 3.0)
        # Brueckner parameters
        params.electron_rs = params.electron_WS_radius / params.a0
        # Fermi wave number
        params.electron_Fermi_wavenumber = (3.0 * pi**2 * params.electron_number_density) ** (1.0 / 3.0)

        # Fermi energy
        params.electron_Fermi_energy = params.hbar2 * params.electron_Fermi_wavenumber**2 / (2.0 * params.me)

        # Other electron parameters
        params.electron_degeneracy_parameter = params.kB * params.electron_temperature / params.electron_Fermi_energy
        params.electron_relativistic_parameter = params.hbar * params.electron_Fermi_wavenumber / (params.me * params.c0)

        # Eq. 1 in Murillo Phys Rev E 81 036403 (2010)
        params.electron_coupling = params.qe**2 / (
            params.fourpie0
            * params.electron_Fermi_energy
            * params.electron_WS_radius
            * sqrt(1 + params.electron_degeneracy_parameter**2)
        )

        # Warm Dense Matter Parameter, Eq.3 in Murillo Phys Rev E 81 036403 (2010)
        params.wdm_parameter = 2.0 / (params.electron_degeneracy_parameter + 1.0 / params.electron_degeneracy_parameter)
        params.wdm_parameter *= 2.0 / (params.electron_coupling + 1.0 / params.electron_coupling)

        if params.magnetized:
            b_mag = sqrt((params.magnetic_field**2).sum())  # magnitude of B
            if params.units == "cgs":
                params.electron_cyclotron_frequency = params.qe * b_mag / params.c0 / params.me
            else:
                params.electron_cyclotron_frequency = params.qe * b_mag / params.me

            params.electron_magnetic_energy = params.hbar * params.electron_cyclotron_frequency
            tan_arg = 0.5 * params.hbar * params.electron_cyclotron_frequency * beta_e

            # Perpendicular correction
            params.horing_perp_correction = (params.electron_plasma_frequency / params.electron_cyclotron_frequency) ** 2
            params.horing_perp_correction *= 1.0 - tan_arg / tanh(tan_arg)
            params.horing_perp_correction += 1

            # Parallel correction
            params.horing_par_correction = 1 - (params.hbar * beta_e * params.electron_plasma_frequency) ** 2 / 12.0

            # Quantum Anisotropy Parameter
            params.horing_delta = params.horing_perp_correction - 1
            params.horing_delta += (params.hbar * beta_e * params.electron_cyclotron_frequency) ** 2 / 12
            params.horing_delta /= params.horing_par_correction

    def calc_screening_length(self, species):

        # Consistency
        self.screening_length_type = self.screening_length_type.lower()

        if self.screening_length_type in ["thomas-fermi", "tf"]:
            # Check electron properties
            if hasattr(self, "electron_temperature_eV"):
                self.electron_temperature = self.eV2K * self.electron_temperature_eV
            else:
                self.electron_temperature = species[-1].temperature

            self.screening_length = species[-1].ThomasFermi_wavelength
        elif self.screening_length_type in ["debye", "debye-huckel", "dh"]:
            self.screening_length = species[-1].debye_length
        elif self.screening_length_type in ["kappa", "from_kappa"]:
            self.screening_length = self.a_ws / self.kappa
        elif self.screening_length_type in ["qsp", "deBroglie"]:
            self.screening_length = species[0].deBroglie_wavelength / (2.0 * pi)
        elif self.screening_length_type in ["coulomb"]:
            self.screening_length = inf
        elif self.screening_length_type in ["custom"]:
            if self.screening_length is None:
                raise AttributeError("potential.screening_length not defined!")

        if not self.screening_length and not self.kappa:
            warn("You have not defined the screening_length nor kappa. I will use the Thomas-Fermi length")
            self.screening_length_type = "thomas-fermi"
            self.screening_length = species[-1].ThomasFermi_wavelength

    def copy_params(self, params):
        """
        Copy necessary parameters.

        Parameters
        ----------
        params: :class:`sarkas.core.Parameters`
            Simulation's parameters.

        """

        self.measure = params.measure
        self.units = params.units
        self.units_dict = params.units_dict
        self.dimensions = params.dimensions
        # Copy needed parameters
        self.box_lengths = params.box_lengths.copy()
        self.pbox_lengths = params.pbox_lengths.copy()
        self.box_volume = params.box_volume
        self.pbox_volume = params.pbox_volume

        # Needed physical constants
        self.fourpie0 = params.fourpie0
        self.a_ws = params.a_ws
        self.kB = params.kB
        self.eV2K = params.eV2K
        self.eV2J = params.eV2J
        self.hbar = params.hbar
        self.QFactor = params.QFactor
        self.T_desired = params.T_desired
        self.coupling_constant = params.coupling_constant

        self.total_num_ptcls = params.total_num_ptcls
        self.total_net_charge = params.total_net_charge
        self.total_num_density = params.total_num_density

        self.num_species = params.num_species
        self.species_charges = params.species_charges.copy()
        self.species_masses = params.species_masses.copy()

        if self.type == "lj":
            self.species_lj_sigmas = params.species_lj_sigmas.copy()

    def from_dict(self, input_dict: dict) -> None:
        """
        Update attributes from input dictionary.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """

        self.__dict__.update(input_dict)

    def method_pretty_print(self):
        """Print algorithm information."""

        msg = f"\nALGORITHM: {self.method}\n"
        # PP section
        if self.method != "fmm":
            pp_cells = (self.box_lengths / self.rc).astype(int)
            ptcls_in_loop = int(self.total_num_density * (self.dimensions * self.rc) ** self.dimensions)
            dim_const = (self.dimensions + 1) / 3.0 * pi
            pp_neighbors = int(self.total_num_density * dim_const * self.rc**self.dimensions)

            fmm_msg = (
                f"rcut = {self.rc / self.a_ws:.4f} a_ws = {self.rc:.6e} {self.units_dict['length']}\n"
                f"No. of PP cells per dimension = {pp_cells}\n"
                f"No. of particles in PP loop = {ptcls_in_loop}\n"
                f"No. of PP neighbors per particle = {pp_neighbors}"
            )

            msg += fmm_msg

        if self.method == "pppm":
            # PM Section
            h_a = self.pppm_h_array / self.a_ws
            halpha = self.pppm_h_array * self.pppm_alpha_ewald
            inv_halpha = (1.0 / halpha).astype(int)

            pppm_msg = (
                f"Charge assignment orders: {self.pppm_cao}\n "
                f"FFT aliases: {self.pppm_aliases}\n"
                f"Mesh: {self.pppm_mesh}\n"
                f"Ewald parameter alpha = {self.pppm_alpha_ewald * self.a_ws:.4f} / a_ws = {self.pppm_alpha_ewald:.6e} {self.units_dict['inverse length']}\n"
                f"Mesh width = {h_a[0]:.4f}, {h_a[1]:.4f}, {h_a[2]:.4f} a_ws\n"
                f"           = {self.pppm_h_array[0]:.4e}, {self.pppm_h_array[1]:.4e}, {self.pppm_h_array[2]:.4e} {self.units_dict['length']}\n"
                f"Mesh size * Ewald_parameter (h * alpha) = {halpha[0]:.4f}, {halpha[1]:.4f}, {halpha[2]:.4f}\n"
                f"                                        ~ 1/{inv_halpha[0]}, 1/{inv_halpha[1]}, 1/{inv_halpha[2]}\n"
                f"PP Force Error = {self.pppm_pp_err:.6e}\n"
                f"PM Force Error = {self.pppm_pm_err:.6e}\n"
                f"Tot Force Error = {self.force_error:.6e}\n"
            )
            msg += pppm_msg

        print(msg)

    def method_setup(self):
        """Setup algorithm's specific parameters."""

        # Check for cutoff radius
        if not self.method == "fmm":
            self.linked_list_on = True  # linked list on

            mask = self.box_lengths > 0.0
            min_length = self.box_lengths[mask].min()
            self.calc_acc_pot = self.update_linked_list
            if not self.rc:
                warn(
                    f"\nThe cut-off radius is not defined. I will use the brute force method.",
                    category=AlgorithmWarning,
                )
                self.rc = min_length / 2.0
                self.linked_list_on = False  # linked list off
                self.calc_acc_pot = self.update_brute

            if self.rc > min_length / 2.0:
                warn(
                    f"\nThe cut-off radius is larger than half of the minimum box length. "
                    f"I will use the brute force method.",
                    # f"L_min/ 2 = {0.5 * min_length:.4e} will be used as rc",
                    category=AlgorithmWarning,
                )

                self.rc = min_length / 2.0
                self.linked_list_on = False  # linked list off
                self.calc_acc_pot = self.update_linked_list

            if self.a_rs != 0.0:
                warn("\nShort-range cut-off enabled. Use this feature with care!", category=AlgorithmWarning)

            # renaming
            if self.method == "p3m":
                self.method == "pppm"

            # Compute pppm parameters
            if self.method == "pppm":
                self.pppm_on = True
                self.pppm_setup()
                self.calc_acc_pot = self.update_pppm
        else:
            self.linked_list_on = False
            self.pppm_on = False
            if self.type == "coulomb":
                self.force_error = self.fmm_precision
                self.calc_acc_pot = self.update_fmm_coulomb
            else:
                self.force_error = self.fmm_precision
                self.calc_acc_pot = self.update_fmm_yukawa

    def pppm_setup(self):
        """Calculate the pppm parameters."""

        # Change lists to numpy arrays for Numba compatibility
        if isinstance(self.pppm_mesh, list):
            self.pppm_mesh = array(self.pppm_mesh, dtype=int64)
        elif not isinstance(self.pppm_mesh, ndarray):
            raise TypeError(f"pppm_mesh is a {type(self.pppm_mesh)}. Please pass a list or numpy array.")

        # Mesh array should be 3 even in 2D
        if not len(self.pppm_mesh) == 3:
            raise AlgorithmWarning(
                f"len(potential.pppm_mesh) = {len(self.pppm_mesh)}.\n"
                f"The PPPM mesh array should be of length 3 even in non 3D simulations."
            )

        if isinstance(self.pppm_aliases, list):
            self.pppm_aliases = array(self.pppm_aliases, dtype=int64)
        elif not isinstance(self.pppm_aliases, ndarray):
            raise TypeError(f"pppm_aliases is a {type(self.pppm_aliases)}. Please pass a list or numpy array.")

        # In case you pass one number and not a list
        if isinstance(self.pppm_cao, int):
            caos = array([1, 1, 1], dtype=int64) * self.pppm_cao
            self.pppm_cao = caos.copy()
        elif isinstance(self.pppm_cao, list):
            self.pppm_cao = array(self.pppm_cao, dtype=int64)
        elif not isinstance(self.pppm_cao, ndarray):
            raise TypeError(f"pppm_cao is a {type(self.pppm_cao)}. Please pass a list or numpy array.")

        if self.pppm_cao.max() > 7:
            raise AttributeError("\nYou have chosen a charge assignment order bigger than 7. Please choose a value <= 7")

        # pppm parameters
        self.pppm_h_array = self.box_lengths / self.pppm_mesh
        # To avoid division by zero
        mask = self.pppm_h_array == 0.0
        self.pppm_h_array[mask] = 1.0
        self.pppm_h_volume = self.pppm_h_array.prod()
        # To avoid unnecessary loops
        self.pppm_aliases[mask] = 0

        # Pack constants together for brevity in input list
        kappa = 1.0 / self.screening_length if self.type == "yukawa" else 0.0
        constants = array([kappa, self.pppm_alpha_ewald, self.fourpie0])

        # Calculate the Optimized Green's Function
        self.pppm_green_function, self.pppm_kx, self.pppm_ky, self.pppm_kz, self.pppm_pm_err = gf_opt(
            self.box_lengths, self.pppm_h_array, self.pppm_mesh, self.pppm_aliases, self.pppm_cao, constants
        )

        # Complete PM Force error calculation
        self.pppm_pm_err *= sqrt(self.total_num_ptcls) * self.a_ws**2 * self.fourpie0
        self.pppm_pm_err /= self.box_volume ** (2.0 / 3.0)

        # Total Force Error
        self.force_error = sqrt(self.pppm_pm_err**2 + self.pppm_pp_err**2)

    def pretty_print(self):
        """Print potential information in a user-friendly way."""

        print("\nPOTENTIAL: ", self.type)
        self.pot_pretty_print(potential=self)
        self.method_pretty_print()

    def setup(self, params, species) -> None:
        """Set up the potential class.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            Simulation's parameters.

        """

        # Enforce consistency
        self.type = self.type.lower()
        self.method = self.method.lower()

        self.copy_params(params)
        self.type_setup(species)
        self.method_setup()

    def type_setup(self, species):
        # Update potential-specific parameters
        # Coulomb potential

        if self.type == "coulomb":
            if self.method == "pp":
                warn("Use the PP method with care for pure Coulomb interactions.", category=AlgorithmWarning)

            from .coulomb import pretty_print_info, update_params

            self.pot_update_params = update_params
            update_params(self)

        elif self.type == "yukawa":
            # Yukawa potential
            from .yukawa import pretty_print_info, update_params

            self.calc_screening_length(species)

            self.pot_update_params = update_params
            update_params(self)

        elif self.type == "egs":
            # exact gradient-corrected screening (EGS) potential
            from .egs import pretty_print_info, update_params

            self.calc_screening_length(species)

            self.pot_update_params = update_params
            update_params(self)

        elif self.type == "lj":
            # Lennard-Jones potential
            from .lennardjones import pretty_print_info, update_params

            self.pot_update_params = update_params
            update_params(self)

        elif self.type == "moliere":
            # Moliere potential
            from .moliere import pretty_print_info, update_params

            self.pot_update_params = update_params
            update_params(self)

        elif self.type == "qsp":
            # QSP potential
            from .qsp import pretty_print_info, update_params

            self.screening_length_type = "qsp"
            self.calc_screening_length(species)
            self.pot_update_params = update_params
            update_params(self, species)

        elif self.type == "hs_yukawa":
            # Hard-Sphere Yukawa
            from .hs_yukawa import update_params

            self.calc_screening_length(species)

            self.pot_update_params = update_params
            update_params(self)

        self.pot_pretty_print = pretty_print_info

    def update_linked_list(self, ptcls):
        """
        Calculate the pp part of the acceleration.

        Parameters
        ----------
        ptcls : :class:`sarkas.particles.Particles`
            Particles data.

        """
        ptcls.potential_energy, ptcls.acc, ptcls.virial = pp_update(
            ptcls.pos,
            ptcls.id,
            ptcls.masses,
            self.box_lengths,
            self.rc,
            self.matrix,
            self.force,
            self.measure,
            ptcls.rdf_hist,
        )

        if self.type != "lj":
            # Mie Energy of charged systems
            # J-M.Caillol, J Chem Phys 101 6080(1994) https: // doi.org / 10.1063 / 1.468422
            dipole = ptcls.charges @ ptcls.pos
            ptcls.potential_energy += 2.0 * pi * (dipole**2).sum() / (3.0 * self.box_volume * self.fourpie0)

    def update_brute(self, ptcls):
        """
        Calculate particles' acceleration and potential brutally.

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        """
        ptcls.potential_energy, ptcls.acc, ptcls.virial = pp_update_0D(
            ptcls.pos,
            ptcls.id,
            ptcls.masses,
            self.box_lengths,
            self.rc,
            self.matrix,
            self.force,
            self.measure,
            ptcls.rdf_hist,
        )
        if self.type != "lj":
            # Mie Energy of charged systems
            # J-M.Caillol, J Chem Phys 101 6080(1994) https: // doi.org / 10.1063 / 1.468422
            dipole = ptcls.charges @ ptcls.pos
            ptcls.potential_energy += 2.0 * pi * (dipole**2).sum() / (3.0 * self.box_volume * self.fourpie0)

    def update_pm(self, ptcls):
        """Calculate the pm part of the potential and acceleration.

        Parameters
        ----------
        ptcls : :class:`sarkas.particles.Particles`
            Particles' data

        """
        U_long, acc_l_r = pm_update(
            ptcls.pos,
            ptcls.charges,
            ptcls.masses,
            self.pppm_mesh,
            self.pppm_h_array,
            self.pppm_h_volume,
            self.box_volume,
            self.pppm_green_function,
            self.pppm_kx,
            self.pppm_ky,
            self.pppm_kz,
            self.pppm_cao,
        )
        # Ewald Self-energy
        U_long += self.QFactor * self.pppm_alpha_ewald / sqrt(pi)
        # Neutrality condition
        U_long += -pi * self.total_net_charge**2.0 / (2.0 * self.box_volume * self.pppm_alpha_ewald**2)

        ptcls.potential_energy += U_long

        ptcls.acc += acc_l_r

    def update_pppm(self, ptcls):
        """Calculate particles' potential and accelerations using pppm method.

        Parameters
        ----------
        ptcls : :class:`sarkas.particles.Particles`
            Particles' data.

        """
        self.update_linked_list(ptcls)
        self.update_pm(ptcls)

    def update_fmm_coulomb(self, ptcls):
        """Calculate particles' potential and accelerations using FMM method.

        Parameters
        ----------
        ptcls : :class:`sarkas.particles.Particles`
            Particles' data

        """

        out_fmm = lfmm3d(eps=self.fmm_precision, sources=ptcls.pos.transpose(), charges=ptcls.charges, pg=2)

        potential_energy = ptcls.charges @ out_fmm.pot.real / self.fourpie0
        acc = -(ptcls.charges * out_fmm.grad.real / ptcls.masses) / self.fourpie0
        ptcls.acc = acc.transpose().copy()
        ptcls.potential_energy = potential_energy

    def update_fmm_yukawa(self, ptcls):
        """Calculate particles' potential and accelerations using FMM method.

        Parameters
        ----------
        ptcls : :class:`sarkas.particles.Particles`
            Particles' data

        """
        out_fmm = hfmm3d(
            eps=self.fmm_precision,
            zk=1j / self.screening_length,
            sources=ptcls.pos.transpose(),
            charges=ptcls.charges,
            pg=2,
        )

        potential_energy = ptcls.charges @ out_fmm.pot.real / self.fourpie0
        acc = -(ptcls.charges * out_fmm.grad.real / ptcls.masses) / self.fourpie0
        ptcls.acc = acc.transpose().copy()
        ptcls.potential_energy = potential_energy
