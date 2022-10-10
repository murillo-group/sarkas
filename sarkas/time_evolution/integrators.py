"""
Module of various types of time_evolution
"""

from copy import deepcopy
from numba import float64, int64, jit, void
from numpy import arange, array, cos, cross, log, pi, sin, sqrt, zeros
from scipy.linalg import norm


class Integrator:
    """
    Class used to assign integrator type.

    Attributes
    ----------
    dt : float
        Timestep.

    kB : float
        Boltzmann constant.

    magnetized : bool
        Magnetized simulation flag.

    species_num : numpy.ndarray
        Number of particles of each species.

    species_plasma_frequencies : numpy.ndarray
        Plasma frequency of each species.

    box_lengths : numpy.ndarray
        Length of each box side.

    pbox_lengths : numpy.ndarray
        Initial particle box sides' lengths.

    verbose : bool
        Verbose output flag.

    type : str
        Integrator type.

    """

    dt: float = None
    kB: float = None

    # attributes
    type: str = None
    supported_integrators = {}
    equilibration_type: str = "verlet"
    magnetization_type: str = "magnetic_verlet"
    production_type: str = "verlet"

    species_num = None
    species_plasma_frequencies = None

    # Thermostat attributes
    thermalization: bool = True
    thermostat_type: str = "berendsen"
    thermalization_rate: float = 2.0
    thermalization_timestep: int = 0
    berendsen_tau: float = None
    thermostat_temperatures = None
    thermostat_temperatures_eV = None

    # Magnetic attributes
    magnetized: bool = False
    magnetic_field_uvector = None
    magnetic_field = None
    omega_c = None
    species_cyclotron_frequencies = None
    ccodt = None
    cdt = None
    ssodt = None
    sdt = None
    v_B = None
    v_F = None

    # Langevin attributes
    c1 = None
    c2 = None
    sigma = None
    box_lengths = None
    pbox_lengths = None

    boundary_conditions = None

    supported_boundary_conditions = {}

    verbose: bool = False

    # def __repr__(self):
    #     sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
    #     disp = 'Integrator( \n'
    #     for key, value in sortedDict.items():
    #         disp += "\t{} : {}\n".format(key, value)
    #     disp += ')'
    #     return disp

    def __copy__(self):
        """Make a shallow copy of the object using copy by creating a new instance of the object and copying its __dict__."""
        # Create a new object
        _copy = type(self)()
        # copy the dictionary
        _copy.from_dict(input_dict=self.__dict__)
        return _copy

    def __deepcopy__(self, memodict={}):
        """Make a deepcopy of the object.

        Parameters
        ----------
        memodict: dict
            Dictionary of id's to copies

        Returns
        -------
        _copy: :class:`sarkas.time_evolution.integrators.Integrator`
            A new Integrator class.
        """
        id_self = id(self)  # memorization avoids unnecessary recursion
        _copy = memodict.get(id_self)
        if _copy is None:
            _copy = type(self)()
            # Make a deepcopy of the mutable arrays using numpy copy function
            for k, v in self.__dict__.items():
                if k != "thread_ls":
                    _copy.__dict__[k] = deepcopy(v, memodict)

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

    def copy_params(self, params):
        """
        Copy necessary parameters.

        Parameters
        ----------
        params: :class:`sarkas.core.Parameters`
            Simulation's parameters.

        """
        self.box_lengths = params.box_lengths
        self.pbox_lengths = params.pbox_lengths
        self.dimensions = params.dimensions
        self.kB = params.kB
        self.eV2K = params.eV2K
        self.total_num_ptcls = params.total_num_ptcls
        self.species_num = params.species_num.copy()
        self.species_plasma_frequencies = params.species_plasma_frequencies.copy()
        self.species_masses = params.species_masses.copy()
        self.species_temperatures = params.species_temperatures.copy()
        self.verbose = params.verbose
        # Enforce consistency
        if not self.boundary_conditions:
            self.boundary_conditions = params.boundary_conditions.lower()

        # Check whether you input temperatures in eV or K
        if self.thermalization and self.thermostat_temperatures:
            self.thermostat_temperatures_eV = self.thermostat_temperatures.copy() / self.eV2K
        elif self.thermalization and self.thermostat_temperatures_eV:
            self.thermostat_temperatures = self.thermostat_temperatures_eV.copy() * self.eV2K
        elif self.thermalization and not self.thermostat_temperatures:
            self.thermostate_temperatures = params.species_temperatures.copy()
            self.thermostate_temperatures_eV = params.species_temperatures_eV.copy()

        # Backwards compatibility
        if hasattr(self, "equilibration_steps"):
            params.equilibration_steps = self.equilibration_steps

        if hasattr(self, "magnetization_steps"):
            params.equilibration_steps = self.magnetization_steps

        if hasattr(self, "production_steps"):
            params.production_steps = self.production_steps

        if hasattr(self, "eq_dump_step"):
            params.eq_dump_step = self.eq_dump_step

        if hasattr(self, "mag_dump_step"):
            params.mag_dump_step = self.mag_dump_step

        if hasattr(self, "eq_dump_step"):
            params.prod_dump_step = self.prod_dump_step

        if not self.boundary_conditions:
            self.boundary_conditions = params.boundary_conditions

        if not hasattr(params, "boundary_conditions"):
            params.boundary_conditions = self.boundary_conditions

        if params.magnetized:
            self.magnetized = True
            self.magnetic_field = params.magnetic_field.copy()
            self.species_cyclotron_frequencies = params.species_cyclotron_frequencies.copy()

    def setup(self, params, potential):
        """
        Assign attributes from simulation's parameters and classes.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
            Parameters class.

        potential : :class:`sarkas.potentials.core.Potential`
            Potential class.

        """
        if self.dt is None:
            raise ValueError("integrator.dt is None. Please define Integrator.dt")

        self.copy_params(params)

        if self.magnetized:
            self.magnetic_setup()

        if self.type:
            self.type = self.type.lower()
            self.equilibration_type = self.type
            self.production_type = self.type

        self.boundary_condition_setup()

        if self.thermalization:
            self.thermostat_setup()

        self.pot_acc_setup(potential)

    def pot_acc_setup(self, potential):
        """
        Link the :meth:`.update_accelerations` method depending on the potential algorithm.

        Parameters
        ----------
        potential : :class:`sarkas.potentials.core.Potential`
            Potential class.

        """

        self.potential_type = potential.type
        if potential.method != "fmm":
            if potential.pppm_on:
                self.update_accelerations = potential.update_pppm
            else:
                if potential.linked_list_on:
                    self.update_accelerations = potential.update_linked_list
                else:
                    self.update_accelerations = potential.update_brute
        else:
            self.update_accelerations = (
                potential.update_fmm_coulomb if potential.type == "coulomb" else potential.update_fmm_yukawa
            )

    def boundary_condition_setup(self):

        self.supported_boundary_conditions = {
            "periodic": self.periodic_bc,
            "absorbing": self.absorbing_bc,
            "reflective": self.reflecting_bc,
            "open": self.open_bc,
        }
        msg = (
            f"Unsupported boundary conditions. "
            f"Choose one of the supported boundary conditions\n{self.supported_boundary_conditions.keys()}",
        )
        # Assign integrator.enforce_bc to the correct method
        self.enforce_bc = self.supported_boundary_conditions.get(self.boundary_conditions, ValueError(msg))

    def thermostat_setup(self):
        """
        Assign attributes from simulation's parameters.

        Raises
        ------
        ValueError
            If a thermostat different than Berendsen is chosen.

        """

        self.thermostat_type = self.thermostat_type.lower()

        if self.thermostat_type != "berendsen":
            raise ValueError("Only Berendsen thermostat is supported.")

        if self.berendsen_tau:
            self.thermalization_rate = 1.0 / self.berendsen_tau
        else:
            self.berendsen_tau = 1.0 / self.thermalization_rate

    def type_setup(self, int_type):
        """

        Parameters
        ----------
        int_type: str
            Integrator type to use.

        Raises
        ------
        : ValueError
            If `int_type` is not a supported integrator.

        """

        # if int_type not in self.supported_integrators:
        #     raise ValueError(
        #         "Integrator not supported. " "Please choose one of the supported integrators \n",
        #         self.supported_integrators,
        #     )

        # Assign integrator.update to the correct method

        if int_type == "langevin":

            self.sigma = sqrt(2.0 * self.langevin_gamma * self.kB * self.species_temperatures / self.species_masses)
            self.c1 = 1.0 - 0.5 * self.langevin_gamma * self.dt
            self.c2 = 1.0 / (1.0 + 0.5 * self.langevin_gamma * self.dt)

        elif int_type == "magnetic_verlet":

            # Calculate functions for magnetic integrator
            # This could be used when the generalization to Forest-Ruth and MacLachlan algorithms will be implemented
            # In a magnetic Velocity-Verlet the coefficient is 1/2, see eq.~(78) in :cite:`Chin2008`
            self.magnetic_helpers(0.5)

            if self.magnetic_field_uvector @ array([0.0, 0.0, 1.0]) == 1.0:  # dot product
                int_type = "magnetic_verlet_zdir"

        elif int_type == "magnetic_pos_verlet":
            # Calculate functions for magnetic integrator
            # This could be used when the generalization to Forest-Ruth and MacLachlan algorithms will be implemented
            # In a magnetic Velocity-Verlet the coefficient is 1/2, see eq.~(78) in :cite:`Chin2008`
            self.magnetic_helpers(1.0)

            if self.magnetic_field_uvector @ array([0.0, 0.0, 1.0]) == 1.0:  # dot product
                int_type = "magnetic_pos_verlet_zdir"

        elif int_type == "magnetic_boris":

            # In a leapfrog-type algorithm the coefficient is different for the acceleration and magnetic rotation
            # see eq.~(79) in :cite:`Chin2008`
            self.magnetic_helpers(1.0)

            if self.magnetic_field_uvector @ array([0.0, 0.0, 1.0]) == 1.0:  # dot product
                int_type = "magnetic_boris_zdir"

        elif int_type == "cyclotronic":
            # Calculate functions for magnetic integrator
            # This could be used when the generalization to Forest-Ruth and MacLachlan algorithms will be implemented
            # In a magnetic Velocity-Verlet the coefficient is 1/2, see eq.~(78) in :cite:`Chin2008`
            self.magnetic_helpers(0.5)

            if self.magnetic_field_uvector @ array([0.0, 0.0, 1.0]) == 1.0:  # dot product
                int_type = "cyclotronic_zdir"

        self.supported_integrators = {
            "verlet": self.verlet,
            "langevin": self.langevin,
            "magnetic_verlet": self.magnetic_verlet,
            "magnetic_verlet_zdir": self.magnetic_verlet_zdir,
            "magnetic_pos_verlet": self.magnetic_pos_verlet,
            "magnetic_pos_verlet_zdir": self.magnetic_pos_verlet_zdir,
            "magnetic_boris": self.magnetic_boris,
            "magnetic_boris_zdir": self.magnetic_boris_zdir,
            "cyclotronic": self.cyclotronic,
            "cyclotronic_zdir": self.cyclotronic_zdir,
        }

        msg = f"Integrator not supported. Please choose one of the supported integrators \n{self.supported_integrators.keys()}"

        return self.supported_integrators.get(int_type, ValueError(msg))

    def magnetic_setup(self):
        # Create the unit vector of the magnetic field
        self.magnetic_field_uvector = self.magnetic_field / norm(self.magnetic_field)
        self.omega_c = zeros((self.total_num_ptcls, 3))

        sp_start = 0
        sp_end = 0
        for ic, sp_np in enumerate(self.species_num):
            sp_end += sp_np
            self.omega_c[sp_start:sp_end, :] = self.species_cyclotron_frequencies[ic]
            sp_start += sp_np

        # array to temporary store velocities
        # Luciano: I have the vague doubt that allocating memory for these arrays is faster than calculating them
        # each time step
        self.v_B = zeros((self.total_num_ptcls, 3))
        self.v_F = zeros((self.total_num_ptcls, 3))

    def langevin(self, ptcls):
        """
        Update particles class using the velocity verlet algorithm and Langevin damping.

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.


        """

        beta = ptcls.gaussian(0.0, 1.0, (self.total_num_ptcls, self.dimensions))
        sp_start = 0  # start index for species loop
        sp_end = 0
        for ic, num in enumerate(self.species_num):
            sp_end += num
            ptcls.pos[sp_start:sp_end, : self.dimensions] += (
                self.c1 * self.dt * ptcls.vel[sp_start:sp_end, : self.dimensions]
                + 0.5 * self.dt**2 * ptcls.acc[sp_start:sp_end, : self.dimensions]
                + 0.5 * self.sigma[ic] * self.dt**1.5 * beta
            )
            sp_start += num

        # Enforce boundary condition
        self.enforce_bc(ptcls)

        acc_old = ptcls.acc.copy()
        self.update_accelerations(ptcls)

        sp_start = 0
        sp_end = 0
        for ic, num in enumerate(self.species_num):
            sp_end += num

            ptcls.vel[sp_start:sp_end, : self.dimensions] = (
                self.c1 * self.c2 * ptcls.vel[sp_start:sp_end, : self.dimensions]
                + 0.5
                * self.c2
                * self.dt
                * (ptcls.acc[sp_start:sp_end, : self.dimensions] + acc_old[sp_start:sp_end, : self.dimensions])
                + self.c2 * self.sigma[ic] * sqrt(self.dt) * beta
            )
            sp_start += num

    def verlet(self, ptcls):
        """
        Update particles' class based on velocity verlet algorithm.
        More information can be found here: https://en.wikipedia.org/wiki/Verlet_integration
        or on the Sarkas website.

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        """
        # First half step velocity update
        ptcls.vel += 0.5 * ptcls.acc * self.dt
        # Full step position update
        ptcls.pos += ptcls.vel * self.dt
        # Enforce boundary condition
        self.enforce_bc(ptcls)
        # Compute total potential energy and acceleration for second half step velocity update
        self.update_accelerations(ptcls)
        # Second half step velocity update
        ptcls.vel += 0.5 * ptcls.acc * self.dt

    def magnetic_helpers(self, coefficient):
        """Calculate the trigonometric functions of the magnetic integrators.

        Parameters
        ----------
        coefficient: float
            Timestep coefficient.

        Notes
        -----
        This is useful for the Leapfrog magnetic algorithm and future Forest-Ruth and MacLachlan algorithms.

        """
        theta = self.omega_c * self.dt * coefficient
        self.sdt = sin(theta)
        self.cdt = cos(theta)
        self.ccodt = 1.0 - self.cdt
        self.ssodt = 1.0 - self.sdt / theta

    def magnetic_verlet_zdir(self, ptcls):
        """
        Update particles' class based on velocity verlet method in the case of a
        constant magnetic field along the :math:`z` axis. For more info see eq. (78) of Ref. :cite:`Chin2008`

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        Returns
        -------
        potential_energy : float
             Total potential energy.

        Notes
        -----
        This integrator is faster than `magnetic_verlet` but valid only for a magnetic field in the :math:`z`-direction.
        This is the preferred choice in this case.
        """

        # First half step of velocity update
        # # Magnetic rotation x - velocity
        # (B x v)_x  = -v_y, (B x B x v)_x = -v_x
        self.v_B[:, 0] = ptcls.vel[:, 1] * self.sdt[:, 0] + ptcls.vel[:, 0] * self.cdt[:, 0]
        # Magnetic rotation y - velocity
        # (B x v)_y  = v_x, (B x B x v)_y = -v_y
        self.v_B[:, 1] = -ptcls.vel[:, 0] * self.sdt[:, 0] + ptcls.vel[:, 1] * self.cdt[:, 1]

        # Magnetic + Const force field x - velocity
        # (B x a)_x  = -a_y, (B x B x a)_x = -a_x
        self.v_F[:, 0] = (
            self.ccodt[:, 1] / self.omega_c[:, 1] * ptcls.acc[:, 1]
            + self.sdt[:, 0] / self.omega_c[:, 0] * ptcls.acc[:, 0]
        )
        # Magnetic + Const force field y - velocity
        # (B x a)_y  = a_x, (B x B x a)_y = -a_y
        self.v_F[:, 1] = (
            -self.ccodt[:, 0] / self.omega_c[:, 0] * ptcls.acc[:, 0]
            + self.sdt[:, 1] / self.omega_c[:, 1] * ptcls.acc[:, 1]
        )

        ptcls.vel[:, 0] = self.v_B[:, 0] + self.v_F[:, 0]
        ptcls.vel[:, 1] = self.v_B[:, 1] + self.v_F[:, 1]
        ptcls.vel[:, 2] += 0.5 * self.dt * ptcls.acc[:, 2]

        # Position update
        ptcls.pos += ptcls.vel * self.dt

        # Enforce boundary condition
        self.enforce_bc(ptcls)

        # Compute total potential energy and acceleration for second half step velocity update
        potential_energy = self.update_accelerations(ptcls)

        # # Magnetic rotation x - velocity
        # (B x v)_x  = -v_y, (B x B x v)_x = -v_x
        self.v_B[:, 0] = ptcls.vel[:, 1] * self.sdt[:, 0] + ptcls.vel[:, 0] * self.cdt[:, 0]
        # Magnetic rotation y - velocity
        # (B x v)_y  = v_x, (B x B x v)_y = -v_y
        self.v_B[:, 1] = -ptcls.vel[:, 0] * self.sdt[:, 0] + ptcls.vel[:, 1] * self.cdt[:, 1]

        # Magnetic + Const force field x - velocity
        # (B x a)_x  = -a_y, (B x B x a)_x = -a_x
        self.v_F[:, 0] = (
            self.ccodt[:, 1] / self.omega_c[:, 1] * ptcls.acc[:, 1]
            + self.sdt[:, 0] / self.omega_c[:, 0] * ptcls.acc[:, 0]
        )
        # Magnetic + Const force field y - velocity
        # (B x a)_y  = a_x, (B x B x a)_y = -a_y
        self.v_F[:, 1] = (
            -self.ccodt[:, 0] / self.omega_c[:, 0] * ptcls.acc[:, 0]
            + self.sdt[:, 1] / self.omega_c[:, 1] * ptcls.acc[:, 1]
        )

        ptcls.vel[:, 0] = self.v_B[:, 0] + self.v_F[:, 0]
        ptcls.vel[:, 1] = self.v_B[:, 1] + self.v_F[:, 1]
        ptcls.vel[:, 2] += 0.5 * self.dt * ptcls.acc[:, 2]

        return potential_energy

    def magnetic_verlet(self, ptcls):
        """
        Update particles' class based on velocity verlet method in the case of an arbitrary direction of the
        constant magnetic field. For more info see eq. (78) of Ref. :cite:`Chin2008`

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        Returns
        -------
        potential_energy : float
             Total potential energy.

        Notes
        -----
        :cite:`Chin2008` equations are written for a negative charge. This allows him to write
        :math:`\\dot{\\mathbf v} = \\omega_c \\hat{B} \\times \\mathbf v`. In the case of positive charges we will have
        :math:`\\dot{\\mathbf v} = - \\omega_c \\hat{B} \\times \\mathbf v`.
        Hence the reason of the different signs in the formulas below compared to Chin's.

        Warnings
        --------
        This integrator is valid for a magnetic field in an arbitrary direction. However, while the integrator works for
        an arbitrary direction, methods in :mod:`sarkas.tools.observables` work only for a magnetic field in the
        :math:`z` - direction. Hence, if you choose to use this integrator remember to change your physical observables.

        """
        # Calculate the cross products
        b_cross_v = cross(self.magnetic_field_uvector, ptcls.vel)
        b_cross_b_cross_v = cross(self.magnetic_field_uvector, b_cross_v)
        b_cross_a = cross(self.magnetic_field_uvector, ptcls.acc)
        b_cross_b_cross_a = cross(self.magnetic_field_uvector, b_cross_a)

        # First half step of velocity update
        ptcls.vel += -self.sdt * b_cross_v + self.ccodt * b_cross_b_cross_v

        ptcls.vel += (
            0.5 * ptcls.acc * self.dt
            - self.ccodt / self.omega_c * b_cross_a
            + 0.5 * self.dt * self.ssodt * b_cross_b_cross_a
        )

        # Position update
        ptcls.pos += ptcls.vel * self.dt

        # Enforce boundary condition
        self.enforce_bc(ptcls)

        # Compute total potential energy and acceleration for second half step velocity update
        potential_energy = self.update_accelerations(ptcls)

        # Re-calculate the cross products
        b_cross_v = cross(self.magnetic_field_uvector, ptcls.vel)
        b_cross_b_cross_v = cross(self.magnetic_field_uvector, b_cross_v)
        b_cross_a = cross(self.magnetic_field_uvector, ptcls.acc)
        b_cross_b_cross_a = cross(self.magnetic_field_uvector, b_cross_a)

        # Second half step velocity update
        ptcls.vel += -self.sdt * b_cross_v + self.ccodt * b_cross_b_cross_v

        ptcls.vel += (
            0.5 * ptcls.acc * self.dt
            - self.ccodt / self.omega_c * b_cross_a
            + 0.5 * self.dt * self.ssodt * b_cross_b_cross_a
        )

        return potential_energy

    def magnetic_boris_zdir(self, ptcls):
        """
        Update particles' class using the Boris algorithm in the case of a
        constant magnetic field along the :math:`z` axis. For more info see eqs. (80) - (81) of Ref. :cite:`Chin2008`

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        Returns
        -------
        potential_energy : float
             Total potential energy.

        """
        # First half step of velocity update: Apply exp(dt * V_F / 2)
        ptcls.vel += 0.5 * ptcls.acc * self.dt

        # Rotate: Apply exp( dt * V)
        # B cross v
        self.v_B[:, 0] = -self.sdt[:, 1] * ptcls.vel[:, 1]
        self.v_B[:, 1] = self.sdt[:, 0] * ptcls.vel[:, 0]

        # B cross B cross v
        self.v_B[:, 0] -= self.ccodt[:, 0] * ptcls.vel[:, 0]
        self.v_B[:, 1] -= self.ccodt[:, 1] * ptcls.vel[:, 1]
        # Update velocities
        ptcls.vel[:, :2] += self.v_B[:, :2]

        # Second Acceleration half step: Apply exp(dt * V_F / 2)
        ptcls.vel += 0.5 * ptcls.acc * self.dt

        # Full step position update
        ptcls.pos += ptcls.vel * self.dt

        # Enforce boundary condition
        self.enforce_bc(ptcls)

        # Compute total potential energy and acceleration for second half step velocity update
        potential_energy = self.update_accelerations(ptcls)

        return potential_energy

    def magnetic_boris(self, ptcls):
        """
        Update particles' class using the Boris algorithm in the case of a
        constant magnetic field along the :math:`z` axis. For more info see eqs. (80) - (81) of Ref. :cite:`Chin2008`

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        Returns
        -------
        potential_energy : float
             Total potential energy.

        """

        # First half step of velocity update: Apply exp(eV_F/2)
        ptcls.vel += 0.5 * ptcls.acc * self.dt

        # Rotate: Apply exp( dt * V)
        # B cross v
        b_cross_v = cross(self.magnetic_field_uvector, ptcls.vel)
        # B cross B cross v
        b_cross_b_cross_v = cross(self.magnetic_field_uvector, b_cross_v)
        ptcls.vel += self.sdt * b_cross_v + self.ccodt * b_cross_b_cross_v

        # Second Acceleration half step: Apply exp(dt * V_F / 2)
        ptcls.vel += 0.5 * ptcls.acc * self.dt

        # Full step position update
        ptcls.pos += ptcls.vel * self.dt

        # Periodic boundary condition
        enforce_pbc(ptcls.pos, ptcls.pbc_cntr, self.box_lengths)

        # Compute total potential energy and acceleration for second half step velocity update
        potential_energy = self.update_accelerations(ptcls)

        return potential_energy

    def magnetic_pos_verlet_zdir(self, ptcls):
        """
        Update particles' class based on position verlet method in the case of a
        constant magnetic field along the :math:`z` axis. For more info see eq. (79) of Ref. :cite:`Chin2008`

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        Returns
        -------
        potential_energy : float
             Total potential energy.

        Notes
        -----
        This integrator is faster than `magnetic_verlet` but valid only for a magnetic field in the :math:`z`-direction.
        This is the preferred choice in this case.
        """

        # Position update
        ptcls.pos += 0.5 * ptcls.vel * self.dt

        # Enforce boundary condition
        self.enforce_bc(ptcls)

        # Compute total potential energy and acceleration for second half step velocity update
        potential_energy = self.update_accelerations(ptcls)

        # First half step of velocity update
        # # Magnetic rotation x - velocity
        # (B x v)_x  = -v_y, (B x B x v)_x = -v_x
        self.v_B[:, 0] = ptcls.vel[:, 1] * self.sdt[:, 0] + ptcls.vel[:, 0] * self.cdt[:, 0]
        # Magnetic rotation y - velocity
        # (B x v)_y  = v_x, (B x B x v)_y = -v_y
        self.v_B[:, 1] = -ptcls.vel[:, 0] * self.sdt[:, 0] + ptcls.vel[:, 1] * self.cdt[:, 1]

        # Magnetic + Const force field x - velocity
        # (B x a)_x  = -a_y, (B x B x a)_x = -a_x
        self.v_F[:, 0] = (
            self.ccodt[:, 1] / self.omega_c[:, 1] * ptcls.acc[:, 1]
            + self.sdt[:, 0] / self.omega_c[:, 0] * ptcls.acc[:, 0]
        )
        # Magnetic + Const force field y - velocity
        # (B x a)_y  = a_x, (B x B x a)_y = -a_y
        self.v_F[:, 1] = (
            -self.ccodt[:, 0] / self.omega_c[:, 0] * ptcls.acc[:, 0]
            + self.sdt[:, 1] / self.omega_c[:, 1] * ptcls.acc[:, 1]
        )

        ptcls.vel[:, 0] = self.v_B[:, 0] + self.v_F[:, 0]
        ptcls.vel[:, 1] = self.v_B[:, 1] + self.v_F[:, 1]
        ptcls.vel[:, 2] += self.dt * ptcls.acc[:, 2]

        # Position update
        ptcls.pos += 0.5 * ptcls.vel * self.dt

        # Enforce boundary condition
        self.enforce_bc(ptcls)

        return potential_energy

    def magnetic_pos_verlet(self, ptcls):
        """
        Update particles' class based on position verlet method in the case of an arbitrary direction of the
        constant magnetic field. For more info see eq. (79) of Ref. :cite:`Chin2008`

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        Returns
        -------
        potential_energy : float
             Total potential energy.

        Notes
        -----
        :cite:`Chin2008` equations are written for a negative charge. This allows him to write
        :math:`\\dot{\\mathbf v} = \\omega_c \\hat{B} \\times \\mathbf v`. In the case of positive charges we will have
        :math:`\\dot{\\mathbf v} = - \\omega_c \\hat{B} \\times \\mathbf v`.
        Hence the reason of the different signs in the formulas below compared to Chin's.

        Warnings
        --------
        This integrator is valid for a magnetic field in an arbitrary direction. However, while the integrator works for
        an arbitrary direction, methods in :mod:`sarkas.tools.observables` work only for a magnetic field in the
        :math:`z` - direction. Hence, if you choose to use this integrator remember to change your physical observables.

        """
        # Half position update
        ptcls.pos += ptcls.vel * self.dt

        # Enforce boundary condition
        self.enforce_bc(ptcls)

        # Compute total potential energy and acceleration for second half step velocity update
        potential_energy = self.update_accelerations(ptcls)

        # Calculate the cross products
        b_cross_v = cross(self.magnetic_field_uvector, ptcls.vel)
        b_cross_b_cross_v = cross(self.magnetic_field_uvector, b_cross_v)
        b_cross_a = cross(self.magnetic_field_uvector, ptcls.acc)
        b_cross_b_cross_a = cross(self.magnetic_field_uvector, b_cross_a)

        # First half step of velocity update
        ptcls.vel += -self.sdt * b_cross_v + self.ccodt * b_cross_b_cross_v

        ptcls.vel += (
            ptcls.acc * self.dt - self.ccodt / self.omega_c * b_cross_a + self.dt * self.ssodt * b_cross_b_cross_a
        )

        # Second half position update
        ptcls.pos += ptcls.vel * self.dt

        # Enforce boundary condition
        self.enforce_bc(ptcls)

        return potential_energy

    def cyclotronic_zdir(self, ptcls):
        """
        Update particles' class using the cyclotronic algorithm in the case of a
        constant magnetic field along the :math:`z` axis.
        For more info see eqs. (16) - (17) of Ref. :cite:`Patacchini2009`

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        Returns
        -------
        potential_energy : float
             Total potential energy.

        """
        # Drift half step
        # Rotate Positions
        ptcls.pos[:, 0] += (
            ptcls.vel[:, 0] * self.sdt[:, 0] / self.omega_c[:, 0]
            + ptcls.vel[:, 1] * self.ccodt[:, 1] / self.omega_c[:, 1]
        )
        ptcls.pos[:, 1] += (
            ptcls.vel[:, 1] * self.sdt[:, 1] / self.omega_c[:, 1]
            - ptcls.vel[:, 0] * self.ccodt[:, 0] / self.omega_c[:, 0]
        )
        ptcls.pos[:, 2] += 0.5 * ptcls.vel[:, 2] * self.dt
        # Enforce boundary condition
        self.enforce_bc(ptcls)
        # Create rotated velocities
        self.v_B[:, 0] = self.cdt[:, 0] * ptcls.vel[:, 0] + self.sdt[:, 1] * ptcls.vel[:, 1]
        self.v_B[:, 1] = self.cdt[:, 1] * ptcls.vel[:, 1] - self.sdt[:, 0] * ptcls.vel[:, 0]
        ptcls.vel[:, :2] = self.v_B[:, :2].copy()
        # Compute total potential energy and accelerations
        potential_energy = self.update_accelerations(ptcls)

        # Kick full step
        ptcls.vel += ptcls.acc * self.dt

        # Drift half step
        # Rotate Positions
        ptcls.pos[:, 0] += (
            ptcls.vel[:, 0] * self.sdt[:, 0] / self.omega_c[:, 0]
            + ptcls.vel[:, 1] * self.ccodt[:, 1] / self.omega_c[:, 1]
        )
        ptcls.pos[:, 1] += (
            ptcls.vel[:, 1] * self.sdt[:, 1] / self.omega_c[:, 1]
            - ptcls.vel[:, 0] * self.ccodt[:, 0] / self.omega_c[:, 0]
        )
        ptcls.pos[:, 2] += 0.5 * ptcls.vel[:, 2] * self.dt
        # Enforce boundary condition
        self.enforce_bc(ptcls)
        # Create rotated velocities
        self.v_B[:, 0] = self.cdt[:, 0] * ptcls.vel[:, 0] + self.sdt[:, 1] * ptcls.vel[:, 1]
        self.v_B[:, 1] = self.cdt[:, 1] * ptcls.vel[:, 1] - self.sdt[:, 0] * ptcls.vel[:, 0]
        # Update final velocities
        ptcls.vel[:, :2] = self.v_B[:, :2].copy()

        return potential_energy

    def cyclotronic(self, ptcls):
        """
        Update particles' class using the cyclotronic algorithm in the case of a
        constant magnetic field along the :math:`z` axis.
        For more info see eqs. (16) - (17) of Ref. :cite:`Patacchini2009`

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        Returns
        -------
        potential_energy : float
             Total potential energy.

        """
        # Drift half step

        # Calculate the cross products
        b_cross_v = cross(self.magnetic_field_uvector, ptcls.vel)
        b_cross_b_cross_v = cross(self.magnetic_field_uvector, b_cross_v)
        # Rotate Positions
        ptcls.pos += (
            0.5 * ptcls.vel * self.dt
            - self.ccodt * b_cross_v / self.omega_c
            + 0.5 * self.dt * self.ssodt * b_cross_b_cross_v
        )
        # Enforce boundary condition
        self.enforce_bc(ptcls)
        # First half step of velocity update
        ptcls.vel += -self.sdt * b_cross_v + self.ccodt * b_cross_b_cross_v
        # Compute total potential energy and accelerations
        potential_energy = self.update_accelerations(ptcls)

        # Kick full step
        ptcls.vel += ptcls.acc * self.dt

        # Drift half step
        # Calculate the cross products
        b_cross_v = cross(self.magnetic_field_uvector, ptcls.vel)
        b_cross_b_cross_v = cross(self.magnetic_field_uvector, b_cross_v)
        # Rotate Positions
        ptcls.pos += (
            0.5 * ptcls.vel * self.dt
            - self.ccodt * b_cross_v / self.omega_c
            + 0.5 * self.dt * self.ssodt * b_cross_b_cross_v
        )
        # Enforce boundary condition
        self.enforce_bc(ptcls)
        # Second half step of velocity update
        ptcls.vel += -self.sdt * b_cross_v + self.ccodt * b_cross_b_cross_v

        return potential_energy

    def thermostate(self, ptcls):
        """
        Update particles' velocities according to the chosen thermostat

        Parameters
        ----------
        ptcls : :class:`sarkas.particles.Particles`
            Particles' data.

        """
        _, T = ptcls.kinetic_temperature()
        berendsen(ptcls.vel, self.species_temperatures, T, self.species_num, self.thermalization_rate)

    def periodic_bc(self, ptcls):
        """
        Applies periodic boundary conditions by calling enforce_pbc

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        """

        enforce_pbc(ptcls.pos, ptcls.pbc_cntr, self.box_lengths)

    def absorbing_bc(self, ptcls):
        """
        Applies absorbing boundary conditions by calling enforce_abc

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        """

        enforce_abc(ptcls.pos, ptcls.vel, ptcls.acc, ptcls.charges, self.box_lengths)

    def open_bc(self, ptcls):
        """
        Applies open boundary conditions. Basically it does nothing. pass

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        """

        pass

    def reflecting_bc(self, ptcls):
        """
        Applies reflective boundary conditions by calling enforce_rbc

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        """

        enforce_rbc(ptcls.pos, ptcls.vel, self.box_lengths, self.dt)

    def pretty_print(self):
        """Print integrator and thermostat information in a user-friendly way."""

        if self.thermalization:
            print("\nTHERMOSTAT: ")
            print(f"Type: {self.thermostat_type}")
            print(f"First thermostating timestep, i.e. thermalization_timestep = {self.thermalization_timestep}")
            print(f"Berendsen parameter tau: {self.berendsen_tau:.3f} [timesteps]")
            print(f"Berendsen relaxation rate: {self.thermalization_rate:.3f} [1/timesteps] ")
            print("Thermostating temperatures: ")
            for i, (t, t_ev) in enumerate(zip(self.thermostate_temperatures, self.thermostate_temperatures_eV)):
                print(f"Species ID {i}: T_eq = {t:.6e} [K] = {t_ev:.6e} [eV]")

        print("\nINTEGRATOR: ")
        print(f"Equilibration Integrator Type: {self.equilibration_type}")
        if self.magnetized:
            print(f"Magnetization Integrator Type: {self.magnetization_type}")
        print(f"Production Integrator Type: {self.production_type}")

        wp_tot = norm(self.species_plasma_frequencies)
        wp_dt = wp_tot * self.dt

        print(f"Time step = {self.dt:.6e} [s]")
        print(f"Total plasma frequency = {wp_tot:.6e} [rad/s]")
        print(f"w_p dt = {wp_dt:.4f} ~ 1/{int(1.0 / wp_dt)}")

        if self.potential_type == "qsp":
            wp_e = self.species_plasma_frequencies[0]
            wp_ions = norm(self.species_plasma_frequencies[1:])
            print(f"e plasma frequency = {wp_e:.6e} [rad/s]")
            print(f"total ion plasma frequency = {wp_ions:.6e} [rad/s]")
            print(f"w_pe dt = {self.dt * wp_e:.4f} ~ 1/{int(1.0 / (self.dt * wp_e))}")
            print(f"w_pi dt = {self.dt * wp_ions:.4f} ~ 1/{int(1.0 / (self.dt * wp_ions))}")

        elif self.potential_type == "lj":
            print(f"The plasma frequency is defined as w_p = sqrt( epsilon / (sigma^2 * mass) )")

        if self.magnetized:
            high_wc_dt = abs(self.species_cyclotron_frequencies).max() * self.dt
            low_wc_dt = abs(self.species_cyclotron_frequencies).min() * self.dt

            if high_wc_dt > low_wc_dt:
                print(f"Highest w_c dt = {high_wc_dt:2.4f} = {high_wc_dt / pi:.4f} pi")
                print(f"Smallest w_c dt = {low_wc_dt:2.4f} = {low_wc_dt / pi:.4f} pi")
            else:
                print(f"w_c dt = {high_wc_dt:2.4f} = {high_wc_dt / pi:.4f} pi")

        if self.equilibration_type == "langevin" or self.production_type == "langevin":
            print(f"langevin_gamma = {self.langevin_gamma:.4e}")
            print(f"langevin_gamma * dt = {self.langevin_gamma * self.dt:.4e}")
            print(f"langevin_gamma / wp = {self.langevin_gamma / wp_tot:.4e}")
            N = -log(0.001) / (self.langevin_gamma * self.dt)
            print(f"exp( - gamma N dt) = 0.001 ==> N = {N:.4f}")


@jit(void(float64[:, :], float64[:], float64[:], int64[:], float64), nopython=True)
def berendsen(vel, T_desired, T, species_np, tau):
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

    tau : float
        Scale factor.

    """

    # if it < therm_timestep:
    #     fact = sqrt(T_desired / T)
    # else:
    #     fact = sqrt(1.0 + (T_desired / T - 1.0) * tau)  # eq.(11)

    # branchless programming
    fact = sqrt(1.0 + (T_desired / T - 1.0) * tau)
    species_start = 0
    species_end = 0

    for i, num in enumerate(species_np):
        species_end += num
        vel[species_start:species_end, :] *= fact[i]
        species_start += num


@jit(void(float64[:, :], float64[:, :], float64[:]), nopython=True)
def enforce_pbc(pos, cntr, box_vector) -> None:
    """
    Numba'd function to enforce periodic boundary conditions.

    Parameters
    ----------
    pos : numpy.ndarray
        Particles' positions.

    cntr : numpy.ndarray
        Counter for the number of times each particle get folded back into the main simulation box

    box_vector : numpy.ndarray
        Box Dimensions.

    """

    # Loop over all particles
    for p in arange(pos.shape[0]):
        for d in arange(pos.shape[1]):

            # If particle is outside of box in positive direction, wrap to negative side
            if pos[p, d] > box_vector[d]:
                pos[p, d] -= box_vector[d]
                cntr[p, d] += 1
            # If particle is outside of box in negative direction, wrap to positive side
            if pos[p, d] < 0.0:
                pos[p, d] += box_vector[d]
                cntr[p, d] -= 1


@jit(void(float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:]), nopython=True)
def enforce_abc(pos, vel, acc, charges, box_vector) -> None:
    """
    Numba'd function to enforce absorbing boundary conditions.

    Parameters
    ----------
    pos: numpy.ndarray
        Particles' positions.

    vel : numpy.ndarray
        Particles' velocities.

    acc : numpy.ndarray
        Particles' accelerations.

    charges : numpy.ndarray
        Charge of each particle. Shape = (``total_num_ptcls``).

    box_vector: numpy.ndarray
        Box Dimensions.

    """

    # Loop over all particles
    for p in arange(pos.shape[0]):
        for d in arange(pos.shape[1]):

            # If particle is outside of box in positive direction, remove charge, velocity and acceleration
            if pos[p, d] >= box_vector[d]:
                pos[p, d] = box_vector[d]
                vel[p, :] = zeros(3)
                acc[p, :] = zeros(3)
                charges[p] = 0.0
            # If particle is outside of box in negative direction, remove charge, velocity and acceleration
            if pos[p, d] <= 0.0:
                pos[p, d] = 0.0
                vel[p, :] = zeros(3)
                acc[p, :] = zeros(3)
                charges[p] = 0.0


@jit(void(float64[:, :], float64[:, :], float64[:], float64), nopython=True)
def enforce_rbc(pos, vel, box_vector, dt) -> None:
    """
    Numba'd function to enforce reflecting boundary conditions.

    Parameters
    ----------
    pos: numpy.ndarray
        Particles' positions.

    vel : numpy.ndarray
        Particles' velocities.

    box_vector: numpy.ndarray
        Box Dimensions.

    dt : float
        Timestep.

    """

    # Loop over all particles
    for p in arange(pos.shape[0]):
        for d in arange(pos.shape[1]):

            # If particle is outside of box in positive direction, wrap to negative side
            if pos[p, d] > box_vector[d] or pos[p, d] < 0.0:
                # Revert velocity
                vel[p, d] *= -1.0
                # Restore previous position assuming verlet algorithm
                pos[p, d] += vel[p, d] * dt


# @jit(void(float64[:, :], int64[:], float64[:]), nopython=True)
# def remove_drift(vel, nums, masses) -> None:
#     """
#     Numba'd function to enforce conservation of total linear momentum.
#     It updates :attr:`sarkas.particles.Particles.vel`.
#
#     Parameters
#     ----------
#     vel: numpy.ndarray
#         Particles' velocities.
#
#     nums: numpy.ndarray
#         Number of particles of each species.
#
#     masses: numpy.ndarray
#         Mass of each species.
#
#     """
#
#     P = zeros((len(nums), vel.shape[1]))
#     species_start = 0
#     for ic in range(len(nums)):
#         species_end = species_start + nums[ic]
#         P[ic, :] = vel[species_start:species_end, :].sum(axis=0) * masses[ic]
#         species_start = species_end
#
#     if P.sum(axis=0).any() > 1e-40:
#         # Remove tot momentum
#         species_start = 0
#         for ic in range(len(nums)):
#             species_end = species_start + nums[ic]
#             vel[species_start:species_end, :] -= P[ic, :] / (float(nums[ic]) * masses[ic])
#             species_start = species_end
