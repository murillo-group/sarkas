"""
Module handling the potential class.
"""
from warnings import warn

from numpy import pi, tanh, sqrt, array, ndarray

from ..utilities.exceptions import AlgorithmWarning
from ..utilities.maths import inverse_fd_half, fd_integral

from .force_pm import force_optimized_green_function as gf_opt
from .force_pm import update as pm_update
from .force_pp import update as pp_update
from .force_pp import update_0D as pp_update_0D


class Potential:
    """
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
        Coulomb constant :math:`4 \\pi \\epsilon_0`.

    kappa : float
        Inverse screening length.

    linked_list_on : bool
        Flag for choosing the Linked cell list algorithem.

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
        Cutoff radius for the Linked Cell List algorithme.

    screening_length_type : str
        Choice of ways to calculate the screening length. \n
        Choices = `[thomas-fermi, tf, debye, debye-huckel, db, moliere, custom]`.

    screening_length : float
        Value of the screening length.

    total_net_charge : float
        Sum of all the charges.

    type : str
        Type of potential. \n
        Choices = [`"coulomb"`, `"egs"`, `"lennardjones"`, `"moliere"`, `"qsp"`].

    """

    a_rs: float = 0.0
    box_lengths: array = None
    box_volume: float = 0.0
    force_error: float = None
    fourpie0: float = 0.0
    kappa: float = None
    linked_list_on: bool = True
    matrix: ndarray = None
    measure: bool = False
    method: str = "pp"
    pbox_lengths: float = 0.0
    pbox_volume: float = 0.0
    pppm_on: bool = False
    QFactor: float = 0.0
    rc: float = None
    screening_length_type: str = None
    screening_length: float = None
    total_net_charge: float = 0.0
    type: str = "yukawa"

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Potential( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def from_dict(self, input_dict: dict) -> None:
        """
        Update attributes from input dictionary.

        Parameters
        ----------
        input_dict: dict
            Dictionary to be copied.

        """

        self.__dict__.update(input_dict)

    def setup(self, params) -> None:
        """Setup the potential class.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            Simulation's parameters.

        """

        # Enforce consistency
        self.type = self.type.lower()
        self.method = self.method.lower()

        if self.method == "p3m":
            self.method == "pppm"
        # Check for cutoff radius
        if not self.type == "fmm":
            self.linked_list_on = True  # linked list on

            if not self.rc:
                warn(
                    "\nThe cut-off radius is not defined. "
                    "L/2 = {:.4e} will be used as rc".format(0.5 * params.box_lengths.min()),
                    category=AlgorithmWarning,
                )
                self.rc = params.box_lengths.min() / 2.0
                self.linked_list_on = False  # linked list off

            if self.rc > params.box_lengths.min() / 2.0:
                warn(
                    "\nThe cut-off radius is larger than half the box length. "
                    "L/2 = {:.4e} will be used as rc".format(0.5 * params.box_lengths.min()),
                    category=AlgorithmWarning,
                )

                self.rc = params.box_lengths.min() / 2.0
                self.linked_list_on = False  # linked list off

            if self.a_rs != 0.0:
                warn("\nShort-range cut-off enabled. Use this feature with care!", category=AlgorithmWarning)
        # Check for electrons as dynamical species
        if self.type == "qsp":
            mask = params.species_names == "e"
            self.electron_temperature = params.species_temperatures[mask]
            params.ne = float(params.species_num_dens[mask])
            params.electron_temperature = float(params.species_temperatures[mask])
            params.qe = float(params.species_charges[mask])
            params.me = float(params.species_masses[mask])
        elif self.type == "coulomb" and "e" in params.species_names:
            mask = params.species_names == "e"
            self.electron_temperature = params.species_temperatures[mask]
            params.ne = float(params.species_num_dens[mask])
            params.electron_temperature = float(params.species_temperatures[mask])
            params.qe = float(params.species_charges[mask])
            params.me = float(params.species_masses[mask])
        else:
            params.ne = (
                    params.species_charges.transpose() @ params.species_concentrations * params.total_num_density / params.qe
            )

            # Check electron properties
            if hasattr(self, "electron_temperature_eV"):
                self.electron_temperature = params.eV2K * self.electron_temperature_eV
            if hasattr(self, "electron_temperature"):
                params.electron_temperature = self.electron_temperature
            else:
                # if the electron temperature is not defined. The total ion temperature will be used for it.
                # print("\nWARNING: electron temperature not defined. I will use the total ion temperature.")
                params.electron_temperature = params.total_ion_temperature
                self.electron_temperature = params.electron_temperature

        self.calc_electron_properties(params)

        if self.screening_length_type:
            self.screening_length_type = self.screening_length_type.lower()

            if self.screening_length_type in ["thomas-fermi", "tf"]:
                self.screening_length = params.lambda_TF
            elif self.screening_length_type in ["debye", "debye-huckel", "dh"]:
                self.screening_length = params.electron_debye_length
            elif self.screening_length_type in ["custom"]:
                if self.screening_length is None:
                    raise AttributeError("potential.screening_length not defined!")
        else:
            if not self.screening_length and not self.kappa:
                warn("You have not defined the screening_length nor kappa. I will use the Thomas-Fermi length")
                self.screening_length_type = "thomas-fermi"
                self.screening_length = params.lambda_TF

        # Update potential-specific parameters
        # Coulomb potential
        if self.type == "coulomb":
            if self.method == "pp":
                warn("Use the PP method with care for pure Coulomb interactions.", category=AlgorithmWarning)

            from .coulomb import update_params

            update_params(self, params)

        elif self.type == "yukawa":
            # Yukawa potential
            from .yukawa import update_params

            update_params(self, params)

        elif self.type == "egs":
            # exact gradient-corrected screening (EGS) potential
            from .egs import update_params
            update_params(self, params)

        elif self.type == "lj":
            # Lennard-Jones potential
            from .lennardjones import update_params

            update_params(self, params)

        elif self.type == "moliere":
            # Moliere potential
            from .moliere import update_params
            update_params(self, params)

        elif self.type == "qsp":
            # QSP potential
            from .qsp import update_params
            update_params(self, params)

        elif self.type == "hs_yukawa":
            # Hard-Sphere Yukawa
            from .hs_yukawa import update_params
            update_params(self, params)

        # Compute pppm parameters
        if self.method == "pppm":
            self.pppm_on = True
            self.pppm_setup(params)

        # Copy needed parameters
        self.box_lengths = params.box_lengths
        self.pbox_lengths = params.pbox_lengths
        self.box_volume = params.box_volume
        self.pbox_volume = params.pbox_volume
        self.fourpie0 = params.fourpie0
        self.QFactor = params.QFactor
        self.total_net_charge = params.total_net_charge
        self.measure = params.measure

    @staticmethod
    def calc_electron_properties(params):
        """Calculate electronic parameters.
        See Electron Properties webpage in documentation website.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            Simulation's parameters.

        """

        twopi = 2.0 * pi
        spin_degeneracy = 2.0  # g in the notes

        # Inverse temperature for convenience
        beta_e = 1.0 / (params.kB * params.electron_temperature)

        # Plasma frequency
        params.electron_plasma_frequency = sqrt(
            4.0 * pi * params.qe ** 2 * params.ne / (params.fourpie0 * params.me)
        )

        params.electron_debye_length = sqrt(params.fourpie0 / (4.0 * pi * params.qe ** 2 * params.ne * beta_e))

        # de Broglie wavelength
        params.lambda_deB = sqrt(twopi * params.hbar2 * beta_e / params.me)
        lambda3 = params.lambda_deB ** 3

        # Landau length 4pi e^2 beta. The division by fourpie0 is needed for MKS units
        params.landau_length = 4.0 * pi * params.qe ** 2 * beta_e / params.fourpie0

        # chemical potential of electron gas/(kB T), obtained by inverting the density equation.
        params.eta_e = inverse_fd_half(lambda3 * sqrt(pi) * params.ne / 4.0)

        # Thomas-Fermi length obtained from compressibility. See eq.(10) in Ref. [3]_
        lambda_TF_sq = lambda3 / params.landau_length
        lambda_TF_sq /= spin_degeneracy / sqrt(pi) * fd_integral(eta=params.eta_e, p=-0.5)
        params.lambda_TF = sqrt(lambda_TF_sq)

        # Electron WS radius
        params.ae_ws = (3.0 / (4.0 * pi * params.ne)) ** (1.0 / 3.0)
        # Brueckner parameters
        params.rs = params.ae_ws / params.a0
        # Fermi wave number
        params.kF = (3.0 * pi ** 2 * params.ne) ** (1.0 / 3.0)

        # Fermi energy
        params.fermi_energy = params.hbar2 * params.kF ** 2 / (2.0 * params.me)

        # Other electron parameters
        params.electron_degeneracy_parameter = params.kB * params.electron_temperature / params.fermi_energy
        params.relativistic_parameter = params.hbar * params.kF / (params.me * params.c0)

        # Eq. 1 in Murillo Phys Rev E 81 036403 (2010)
        params.electron_coupling = params.qe ** 2 / (
                params.fourpie0 * params.fermi_energy * params.ae_ws * sqrt(
            params.electron_degeneracy_parameter ** 2)
        )

        # Warm Dense Matter Parameter, Eq.3 in Murillo Phys Rev E 81 036403 (2010)
        params.wdm_parameter = 2.0 / (params.electron_degeneracy_parameter + 1.0 / params.electron_degeneracy_parameter)
        params.wdm_parameter *= 2.0 / (params.electron_coupling + 1.0 / params.electron_coupling)

        if params.magnetized:
            b_mag = sqrt((params.magnetic_field ** 2).sum())  # magnitude of B
            if params.units == "cgs":
                params.electron_cyclotron_frequency = (
                        params.qe * b_mag / params.c0 / params.me
                )
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

    def pppm_setup(self, params):
        """Calculate the pppm parameters.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            Simulation's parameters

        """

        # Change lists to numpy arrays for Numba compatibility
        if not isinstance(self.pppm_mesh, ndarray):
            self.pppm_mesh = array(self.pppm_mesh)

        if not isinstance(self.pppm_aliases, ndarray):
            self.pppm_aliases = array(self.pppm_aliases)

        # pppm parameters
        self.pppm_h_array = params.box_lengths / self.pppm_mesh

        # Pack constants together for brevity in input list
        kappa = 1.0/self.screening_length if self.type == "yukawa" else 0.0
        constants = array([kappa, self.pppm_alpha_ewald, params.fourpie0])
        # Calculate the Optimized Green's Function
        self.pppm_green_function, self.pppm_kx, self.pppm_ky, self.pppm_kz, params.pppm_pm_err = gf_opt(
            params.box_lengths, self.pppm_mesh, self.pppm_aliases, self.pppm_cao, constants
        )

        # Complete PM Force error calculation
        params.pppm_pm_err *= sqrt(params.total_num_ptcls) * params.a_ws ** 2 * params.fourpie0
        params.pppm_pm_err /= params.box_volume ** (2.0 / 3.0)

        # Total Force Error
        params.force_error = sqrt(params.pppm_pm_err ** 2 + params.pppm_pp_err ** 2)

        self.force_error = params.force_error

    def update_linked_list(self, ptcls):
        """
        Calculate the pp part of the acceleration.

        Parameters
        ----------
        ptcls : :class:`sarkas.core.Particles`
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
            ptcls.potential_energy += 2.0 * pi * (dipole ** 2).sum() / (3.0 * self.box_volume * self.fourpie0)

    def update_brute(self, ptcls):
        """
        Calculate particles' acceleration and potential brutally.

        Parameters
        ----------
        ptcls: :class:`sarkas.core.Particles`
            Particles data.

        """
        ptcls.potential_energy, ptcls.acc = pp_update_0D(
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
            ptcls.potential_energy += 2.0 * pi * (dipole ** 2).sum() / (3.0 * self.box_volume * self.fourpie0)

    def update_pm(self, ptcls):
        """Calculate the pm part of the potential and acceleration.

        Parameters
        ----------
        ptcls : :class:`sarkas.core.Particles`
            Particles' data

        """
        U_long, acc_l_r = pm_update(
            ptcls.pos,
            ptcls.charges,
            ptcls.masses,
            self.pppm_mesh,
            self.box_lengths,
            self.pppm_green_function,
            self.pppm_kx,
            self.pppm_ky,
            self.pppm_kz,
            self.pppm_cao,
        )
        # Ewald Self-energy
        U_long += self.QFactor * self.pppm_alpha_ewald / sqrt(pi)
        # Neutrality condition
        U_long += -pi * self.total_net_charge ** 2.0 / (2.0 * self.box_volume * self.pppm_alpha_ewald ** 2)

        ptcls.potential_energy += U_long

        ptcls.acc += acc_l_r

    def update_pppm(self, ptcls):
        """Calculate particles' potential and accelerations using pppm method.

        Parameters
        ----------
        ptcls : :class:`sarkas.core.Particles`
            Particles' data.

        """
        self.update_linked_list(ptcls)
        self.update_pm(ptcls)

    # def update_fmm(ptcls, params):
    #     """
    #
    #     Parameters
    #     ----------
    #     ptcls
    #     params
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     if params.potential.type == 'coulomb':
    #         out_fmm = fmm.lfmm3d(eps=1.0e-07, sources=ptcls.pos.transpose(), charges=ptcls.charges, pg=2)
    #     elif params.potential.type == 'yukawa':
    #         out_fmm = fmm.hfmm3d(eps=1.0e-05, zk=1j / params.lambda_TF, sources=ptcls.pos.transpose(),
    #                          charges=ptcls.charges, pg=2)
    #
    #     potential_energy = ptcls.charges @ out_fmm.pot.real * 4.0 * pi / params.fourpie0
    #     ptcls.acc = - (ptcls.charges * out_fmm.grad.real / ptcls.mass).transpose() / params.fourpie0
    #
    #     return potential_energy
