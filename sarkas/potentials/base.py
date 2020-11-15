
import numpy as np
from sarkas.potentials.force_pm import force_optimized_green_function as gf_opt
from sarkas.potentials import force_pm, force_pp
import fdint


class Potential:
    """
    Parameters specific to potential choice.

    Attributes
    ----------
    matrix : array
        Potential's parameters.

    method : str
        Algorithm to use for force calculations.
        "PP" = Linked Cell List (default).
        "P3M" = Particle-Particle Particle-Mesh.

    rc : float
        Cutoff radius.

    type : str
        Interaction potential: LJ, Yukawa, EGS, Coulomb, QSP, Moliere.

    pppm_aliases : array, shape(3)
        Number of aliases in each direction.

    pppm_cao : int
        Charge assignment order.

    on : bool
        Flag.

    pppm_mesh : array, shape(3), int
        Number of mesh point in each direction.

    Mx : int
        Number of mesh point along the :math:`x` axis.

    My : int
        Number of mesh point along the :math:`y` axis.

    Mz : int
        Number of mesh point along the :math:`z` axis.

    mx_max : int
        Number of aliases along the reciprocal :math:`x` direction.

    my_max : int
        Number of aliases along the reciprocal :math:`y` direction.

    mz_max : int
        Number of aliases along the reciprocal :math:`z` direction.

    G_ew : float
        Ewald parameter.

    G_k : array
        Optimized Green's function.

    hx : float
        Mesh spacing in :math:`x` direction.

    hy : float
        Mesh spacing in :math:`y` direction.

    hz : float
        Mesh spacing in :math:`z` direction.

    PP_err : float
        Force error due to short range cutoff.

    PM_err : float
        Force error due to long range cutoff.

    F_err : float
        Total force error.

    kx_v : array
        Array of :math:`k_x` values.

    ky_v : array
        Array of :math:`k_y` values.

    kz_v : array
        Array of :math:`k_z` values.

    """

    def __init__(self):
        self.type = "yukawa"
        self.method = "PP"
        self.matrix = None
        self.force_error = None
        self.measure = False
        self.box_lengths = 0.0
        self.box_volume = 0.0
        self.fourpie0 = 0.0
        self.QFactor = 0.0
        self.total_net_charge = 0.0
        self.pppm_on = False

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = 'Potential( \n'
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ')'
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
        params: sarkas.base.Parameters
            Simulation's parameters.

        """
        # Check for cutoff radius
        if not self.type.lower() == 'fmm':
            self.linked_list_on = True  # linked list on
            if not hasattr(self, "rc"):
                print("\nWARNING: The cut-off radius is not defined. L/2 = {:1.4e} will be used as rc".format(
                    0.5 * params.box_lengths.min()))
                self.rc = params.box_lengths.min() / 2.
                self.linked_list_on = False  # linked list off

            if self.rc > params.box_lengths.min() / 2.:
                print("\nWARNING: The cut-off radius is > L/2. L/2 = ", params.box_lengths.min() / 2,
                      "will be used as rc")
                self.rc = params.box_lengths.min() / 2.
                self.linked_list_on = False  # linked list off

        if self.type.lower() == 'qsp':
            mask = params.species_names == 'e'
            self.electron_temperature = params.species_temperatures[mask]
            params.ne = float(params.species_num_dens[mask])
            params.electron_temperature = float(params.species_temperatures[mask])
            params.qe = float(params.species_charges[mask])
            params.me = float(params.species_masses[mask])
        else:
            params.ne = params.species_charges.transpose() @ params.species_concentrations * params.total_num_density / params.qe

            # Check electron properties
            if hasattr(self, "electron_temperature_eV"):
                self.electron_temperature = params.eV2K * self.electron_temperature_eV
            if hasattr(self, "electron_temperature"):
                params.electron_temperature = self.electron_temperature
            else:
                params.electron_temperature = params.total_ion_temperature
                # if the electron temperature is not defined. The total ion temperature will be used for it.
        self.calc_electron_properties(params)

        if hasattr(self, "kappa"):
            # Thomas-Fermi Length
            params.lambda_TF = params.a_ws / self.kappa

        # Update potential-specific parameters
        # Coulomb potential
        if self.type.lower() == "coulomb":
            from sarkas.potentials import coulomb
            coulomb.update_params(self, params)
        # Yukawa potential
        if self.type.lower() == "yukawa":
            from sarkas.potentials import yukawa
            yukawa.update_params(self, params)
        # exact gradient-corrected screening (EGS) potential
        if self.type.lower() == "egs":
            from sarkas.potentials import egs
            egs.update_params(self, params)
        # Lennard-Jones potential
        if self.type.lower() == "lj":
            from sarkas.potentials import lennardjones as lj
            lj.update_params(self, params)
        # Moliere potential
        if self.type.lower() == "moliere":
            from sarkas.potentials import moliere
            moliere.update_params(self, params)
        # QSP potential
        if self.type.lower() == "qsp":
            from sarkas.potentials import qsp
            qsp.update_params(self, params)

        # Compute pppm parameters
        if self.method == 'P3M' or self.method.lower() == 'pppm':
            self.pppm_on = True
            self.pppm_setup(params)

        # Copy needed parameters
        self.box_lengths = np.copy(params.box_lengths)
        self.box_volume = params.box_volume
        self.fourpie0 = params.fourpie0
        self.QFactor = params.QFactor
        self.total_net_charge = params.total_net_charge
        self.measure = params.measure

    @staticmethod
    def calc_electron_properties(params):
        """Calculate electronic parameters.

        Parameters
        ----------
        params: sarkas.base.Parameters
            Simulation's parameters.

        """

        twopi = 2.0 * np.pi
        # Calculate electron gas properties
        fdint_fdk_vec = np.vectorize(fdint.fdk)
        fdint_ifd1h_vec = np.vectorize(fdint.ifd1h)
        beta_e = 1. / (params.kB * params.electron_temperature)
        lambda_DB = np.sqrt(twopi * params.hbar2 * beta_e / params.me)
        lambda3 = lambda_DB ** 3
        # chemical potential of electron gas/(kB T). See eq.(4) in Ref.[3]_
        params.eta_e = fdint_ifd1h_vec(lambda3 * np.sqrt(np.pi) * params.ne / 4.0)
        # Thomas-Fermi length obtained from compressibility. See eq.(10) in Ref. [3]_
        params.lambda_TF = np.sqrt(params.fourpie0 * np.sqrt(np.pi) * lambda3 / (
                8.0 * np.pi * params.qe ** 2 * beta_e * fdint_fdk_vec(k=-0.5, phi=params.eta_e)))

        params.ae_ws = (3.0/(4.0 * np.pi * params.ne))**(1./3.) # Electron WS radius
        params.rs = params.ae_ws / params.a0
        kF = (3.0 * np.pi ** 2 * params.ne) ** (1. / 3.)
        params.fermi_energy = params.hbar2 * kF ** 2 / (2.0 * params.me)
        params.electron_degeneracy_parameter = params.kB * params.electron_temperature / params.fermi_energy
        params.relativistic_parameter = params.hbar * kF / (params.me * params.c0)
        # Eq. 1 in Murillo Phys Rev E 81 036403 (2010)
        params.electron_coupling = params.qe**2/(
                params.fourpie0 * params.fermi_energy * params.ae_ws* np.sqrt(params.electron_degeneracy_parameter**2))
        # Warm Dense Matter Parameter, Eq.3 in Murillo Phys Rev E 81 036403 (2010)
        params.wdm_parameter = 2.0/(params.electron_degeneracy_parameter + 1.0/params.electron_degeneracy_parameter)
        params.wdm_parameter *= 2.0/(params.electron_coupling + 1.0/params.electron_coupling)

    def update_linked_list(self, ptcls):
        """
        Calculate the pp part of the acceleration.

        Parameters
        ----------
        ptcls: sarkas.base.Particles
            Particles data.

        """
        ptcls.potential_energy, ptcls.acc = force_pp.update(ptcls.pos, ptcls.id, ptcls.masses, self.box_lengths,
                                           self.rc, self.matrix, self.force,
                                           self.measure, ptcls.rdf_hist)

        if not (self.type == "LJ"):
            # Mie Energy of charged systems
            # J-M.Caillol, J Chem Phys 101 6080(1994) https: // doi.org / 10.1063 / 1.468422
            dipole = ptcls.charges @ ptcls.pos
            ptcls.potential_energy += 2.0 * np.pi * np.sum(dipole ** 2) / (3.0 * self.box_volume * self.fourpie0)

    def update_brute(self, ptcls):
        """
        Calculate particles' acceleration and potential brutally.

        Parameters
        ----------
        ptcls: sarkas.base.Particles
            Particles data.

        """
        ptcls.potential_energy, ptcls.acc = force_pp.update_0D(ptcls.pos, ptcls.id, ptcls.masses, self.box_lengths,
                                               self.rc, self.matrix, self.force,
                                               self.measure, ptcls.rdf_hist)
        if not (self.type == "LJ"):
            # Mie Energy of charged systems
            # J-M.Caillol, J Chem Phys 101 6080(1994) https: // doi.org / 10.1063 / 1.468422
            dipole = ptcls.charges @ ptcls.pos
            ptcls.potential_energy += 2.0 * np.pi * np.sum(dipole ** 2) / (3.0 * self.box_volume * self.fourpie0)

    def update_pm(self, ptcls):
        """Calculate the pm part of the potential and acceleration.

        Parameters
        ----------
        ptcls : sarkas.base.Particles
            Particles' data

        """
        U_long, acc_l_r = force_pm.update(ptcls.pos, ptcls.charges, ptcls.masses,
                                          self.pppm_mesh, self.box_lengths, self.pppm_green_function,
                                          self.pppm_kx,
                                          self.pppm_ky,
                                          self.pppm_kz, self.pppm_cao)
        # Ewald Self-energy
        U_long += self.QFactor * self.pppm_alpha_ewald / np.sqrt(np.pi)
        # Neutrality condition
        U_long += - np.pi * self.total_net_charge ** 2.0 / (2.0 * self.box_volume * self.pppm_alpha_ewald ** 2)

        ptcls.potential_energy += U_long

        ptcls.acc += acc_l_r

    def update_pppm(self, ptcls):
        """Calculate particles' potential and accelerations using pppm method.

        Parameters
        ----------
        ptcls : sarkas.base.Particles
            Particles' data

        """
        self.update_linked_list(ptcls)
        self.update_pm(ptcls)

    def pppm_setup(self, params) -> None:
        """Calculate the P3M parameters.

        Parameters
        ----------
        params : sarkas.base.Parameters
            Simulation's parameters

        """
        # P3M parameters
        self.pppm_h_array = params.box_lengths / self.pppm_mesh
        if not isinstance(self.pppm_mesh, np.ndarray):
            self.pppm_mesh = np.array(self.pppm_mesh)

        self.matrix[-1, :, :] = self.pppm_alpha_ewald
        # Calculate the Optimized Green's Function
        kappa = 1. / params.lambda_TF if self.type == "Yukawa" else 0.0
        constants = np.array([kappa, self.pppm_alpha_ewald, params.fourpie0])
        self.pppm_green_function, self.pppm_kx, self.pppm_ky, self.pppm_kz, params.pppm_pm_err = gf_opt(
            params.box_lengths, self.pppm_mesh, self.pppm_aliases, self.pppm_cao, constants)
        # Complete PM Force error calculation
        params.pppm_pm_err *= np.sqrt(params.total_num_ptcls) * params.a_ws ** 2 * params.fourpie0
        params.pppm_pm_err /= params.box_volume ** (2. / 3.)

        # Total Force Error
        params.force_error = np.sqrt(params.pppm_pm_err ** 2 + params.pppm_pp_err ** 2)

        self.force_error = params.force_error

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
    #     if params.potential.type == 'Coulomb':
    #         out_fmm = fmm.lfmm3d(eps=1.0e-07, sources=np.transpose(ptcls.pos), charges=ptcls.charges, pg=2)
    #     elif params.potential.type == 'Yukawa':
    #         out_fmm = fmm.hfmm3d(eps=1.0e-05, zk=1j / params.lambda_TF, sources=np.transpose(ptcls.pos),
    #                          charges=ptcls.charges, pg=2)
    #
    #     potential_energy = ptcls.charges @ out_fmm.pot.real * 4.0 * np.pi / params.fourpie0
    #     ptcls.acc = - np.transpose(ptcls.charges * out_fmm.grad.real / ptcls.mass) / params.fourpie0
    #
    #     return potential_energy





