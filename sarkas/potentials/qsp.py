"""
Module for handling the Quantum Statistical Potential.

Note
----
Notice that in Ref. [Hansen1981]_ the DeBroglie wavelength is defined as

.. math::
   \lambda_{ee} = \dfrac{\hbar}{\sqrt{2 \pi \mu_{ee} k_{B} T} },

while in statistical physics textbooks and Ref. [Glosli2008]_ is defined as

.. math::
   \lambda_{ee} = \dfrac{h}{\sqrt{2 \pi \mu_{ee} k_{B} T} },

References
----------
.. [Hansen1981] `J.P. Hansen and I.R. McDonald, Phys Rev A 23 2041 (1981) <https://doi.org/10.1103/PhysRevA.23.2041>`_
.. [Glosli2008] `J.N. Glosli et al. Phys Rev E 78 025401(R) (2008) <https://doi.org/10.1103/PhysRevE.78.025401>`_


"""
import numpy as np
from numba import njit
import yaml
import math as mt
from sarkas.algorithm.force_pm import force_optimized_green_function as gf_opt


def setup(params, read_input=True):
    """
    Update the ``params`` class with QSP Potential parameters. 
    The QSP Potential is given by eq.(5) in Ref. [Glosli2008]_ .

    Parameters
    ----------
    read_input: bool
        Flag to read inputs from YAML input file.

    params : object
        Simulation's parameters

    """
    # Do a bunch of checks
    # P3M algorithm only
    assert params.potential.method == "P3M", 'QSP interaction can only be calculated using P3M algorithm.'

    # Check for neutrality
    assert params.tot_net_charge == 0, 'Total net charge is not zero.'

    e_list = ['e', 'electrons', 'electron']
    assert params.species[0].name in e_list, 'The 1st species are not electrons. Please redefine the 1st species as electrons.'

    # Default attributes
    params.potential.QSP_type = 'Deutsch'
    params.potential.QSP_Pauli = True
    if read_input:
        # open the input file to read Yukawa parameters
        with open(params.input_file, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)
            for lkey in dics:
                if lkey == "Potential":
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if key == "QSP_type":
                                params.potential.QSP_type = value

                            if key == "QSP_Pauli":
                                params.potential.QSP_Pauli = value

                            if key == "rc":  # cutoff
                                params.potential.rc = float(value)

    update_params(params)


def update_params(params):
    """
    Create potential dependent simulation's parameters.

    Parameters
    ----------
    params : object
        Simulation's parameters

    """
    """
    Dev Notes
    for more info and a description of the potential's parameters.
    QSP_matrix[0,:,:] = qi*qj/4*pi*eps0
    QSP_matrix[1,:,:] = 2pi/deBroglie
    QSP_matrix[2,:,:] = e-e Pauli term factor
    QSP_matrix[3,:,:] = e-e Pauli term exponent term
    QSP_matrix[4,:,:] = Ewald parameter
    """

    if not params.BC.open_axes:
        params.potential.LL_on = True  # linked list on
        if not hasattr(params.potential, "rc"):
            print("\nWARNING: The cut-off radius is not defined. L/2 = ", params.Lv.min() / 2, "will be used as rc")
            params.potential.rc = params.Lv.min() / 2.
            params.potential.LL_on = False  # linked list off

        if params.potential.method == "PP" and params.potential.rc > params.Lv.min() / 2.:
            print("\nWARNING: The cut-off radius is > L/2. L/2 = ", params.Lv.min() / 2, "will be used as rc")
            params.potential.rc = params.Lv.min() / 2.
            params.potential.LL_on = False  # linked list off

    two_pi = 2.0 * np.pi
    four_pi = 2.0 * two_pi
    params.ne = params.species[0].num_density
    params.ae = (3.0 / (four_pi * params.ne)) ** (1.0 / 3.0)  # e WS
    params.rs = params.ae / params.a0  # e coupling parameter
    params.Te = params.species[0].temperature

    # Redefine ion temperatures and ion total number density
    params.Ti = 0.
    params.ni = 0.
    for isp in range(1, params.num_species):
        params.Ti += params.species[isp].concentration * params.species[isp].temperature
        params.ni += params.species[isp].num_density

    # Calculate the total and ion Wigner-Seitz Radius from the total density
    params.aws = (3.0 / (four_pi * params.total_num_density)) ** (1. / 3.)
    params.ai = (3.0 / (four_pi * params.ni)) ** (1.0 / 3.0)  # Ion WS

    beta_e = 1.0 / (params.kB * params.Te)
    beta_i = 1.0 / (params.kB * params.Ti)

    QSP_matrix = np.zeros((5, params.num_species, params.num_species))
    for i, sp1 in enumerate(params.species):
        m1 = sp1.mass
        q1 = sp1.charge

        for j, sp2 in enumerate(params.species):
            m2 = sp2.mass
            q2 = sp2.charge
            reduced = (m1 * m2) / (m1 + m2)
            if i == 0:
                Lambda_dB = np.sqrt(two_pi * beta_e * params.hbar2 / reduced)
                if j == i:  # e-e
                    QSP_matrix[2, i, j] = np.log(2.0) / beta_e
                    QSP_matrix[3, i, j] = four_pi / (np.log(2.0) * Lambda_dB ** 2)
            else:
                Lambda_dB = np.sqrt(two_pi * beta_i * params.hbar2 / reduced)

            QSP_matrix[0, i, j] = q1 * q2 / params.fourpie0
            QSP_matrix[1, i, j] = two_pi / Lambda_dB

    params.QFactor /= params.fourpie0
    if not params.potential.QSP_Pauli:
        QSP_matrix[2, :, :] = 0.0

    params.potential.matrix = QSP_matrix
    params.potential.Gamma_eff = abs(params.potential.matrix[0, 0, 1]) / (params.ai * params.kB * params.Ti)

    # Calculate the (total) plasma frequency
    wp_tot_sq = 0.0
    for i, sp in enumerate(params.species):
        wp2 = four_pi * sp.charge ** 2 * sp.num_density / (sp.mass * params.fourpie0)
        sp.wp = np.sqrt(wp2)
        wp_tot_sq += wp2

    params.wp = np.sqrt(wp_tot_sq)

    if params.potential.QSP_type == "Deutsch":
        params.force = Deutsch_force_P3M
        # Calculate the PP Force error from the e-e diffraction term only.
        params.pppm.PP_err = np.sqrt(two_pi * params.potential.matrix[1, 0, 0])
        params.pppm.PP_err *= np.exp(- params.potential.rc * params.potential.matrix[1, 0, 0])

    elif params.potential.QSP_type == "Kelbg":
        params.force = Kelbg_force_P3M
        # TODO: Calculate the PP Force error from the e-e diffraction term only.
        params.pppm.PP_err = np.sqrt(two_pi * params.potential.matrix[1, 0, 0])
        params.pppm.PP_err *= np.exp(- params.potential.rc * params.potential.matrix[1, 0, 0])

    # P3M parameters
    params.pppm.hx = params.Lx / params.pppm.Mx
    params.pppm.hy = params.Ly / params.pppm.My
    params.pppm.hz = params.Lz / params.pppm.Mz
    params.potential.matrix[4, :, :] = params.pppm.G_ew
    # Calculate the Optimized Green's Function
    constants = np.array([0.0, params.pppm.G_ew, params.fourpie0])
    params.pppm.G_k, params.pppm.kx_v, params.pppm.ky_v, params.pppm.kz_v, params.pppm.PM_err = gf_opt(
        params.pppm.MGrid, params.pppm.aliases, params.Lv, params.pppm.cao, constants)

    # Complete PM and PP Force error calculation
    params.pppm.PM_err *= np.sqrt(params.total_num_ptcls) * params.aws ** 2 * params.fourpie0 / params.box_volume ** (2. / 3.)
    params.pppm.PP_err *= params.aws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)
    # Calculate the total force error
    params.pppm.F_err = np.sqrt(params.pppm.PM_err ** 2 + params.pppm.PP_err ** 2)

    return


@njit
def Deutsch_force_P3M(r, pot_matrix):
    """ 
    Calculate Deutsch QSP Force between two particles.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : array
        It contains potential dependent variables.
        pot_matrix[0,:,:] = qi*qj/4*pi*eps0
        pot_matrix[1,:,:] = 2pi/deBroglie
        pot_matrix[2,:,:] = e-e Pauli term factor
        pot_matrix[3,:,:] = e-e Pauli term exponent term
        pot_matrix[4,:,:] = Ewald parameter

    Returns
    -------
    U : float
        Potential.
                
    force : float
        Force between two particles.

    """

    A = pot_matrix[0]
    C = pot_matrix[1]
    D = pot_matrix[2]
    F = pot_matrix[3]
    alpha = pot_matrix[4]

    a2 = alpha * alpha
    r2 = r * r

    # Ewald short-range potential and force terms
    U_ewald = A * mt.erfc(alpha * r) / r
    f_ewald = U_ewald / r2  # 1/r derivative
    f_ewald += A * (2.0 * alpha / np.sqrt(np.pi) / r2) * np.exp(- a2 * r2)  # erfc derivative

    # Diffraction potential and force term
    U_diff = -A * np.exp(-C * r) / r
    f_diff = U_diff / r2  # 1/r derivative
    f_diff += - A * C * np.exp(-C * r) / r2  # exp derivative

    # Pauli potential and force terms
    U_pauli = D * np.exp(-F * r2)
    f_pauli = 2.0 * D * F * np.exp(-F * r2)

    U = U_ewald + U_diff + U_pauli
    force = f_ewald + f_diff + f_pauli

    return U, force


@njit
def Kelbg_force_P3M(r, pot_matrix):
    """ 
    Calculates the QSP Force between two particles when the P3M algorithm is chosen.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : array
        It contains potential dependent parameters.
        pot_matrix[0] = qi*qj/4*pi*eps0
        pot_matrix[1] = 2pi/deBroglie
        pot_matrix[2] = e-e Pauli term factor
        pot_matrix[3] = e-e Pauli term exponent term
        pot_matrix[4] = Ewald parameter

    Returns
    -------
    U : float
        Potential.
                
    force : float
        Force between two particles.
    
    """

    A = pot_matrix[0]
    C = pot_matrix[1]
    D = pot_matrix[2]
    F = pot_matrix[3]
    alpha = pot_matrix[4]
    C2 = C * C
    a2 = alpha * alpha
    r2 = r * r

    # Ewald short-range potential and force terms
    U_ewald = A * mt.erfc(alpha * r) / r
    f_ewald = U_ewald / r2  # 1/r derivative
    f_ewald += A * (2.0 * alpha / np.sqrt(np.pi) / r2) * np.exp(- a2 * r2)  # erfc derivative

    # Diffraction potential and force term
    U_diff = -A * np.exp(-C2 * r2 / np.pi) / r
    U_diff += A * C * mt.erfc(C * r / np.sqrt(np.pi))

    f_diff = -A * (2.0 * C2 * r2 + np.pi) * np.exp(- C2 * r2 / np.pi) / (np.pi * r * r2)  # exp(r)/r derivative
    f_diff += 2.0 * A * C2 * np.exp(- C2 * r2 / np.pi) / r / np.pi  # erfc derivative

    # Pauli Term
    U_pauli = D * np.exp(-F * r2)
    f_pauli = 2.0 * D * F * np.exp(-F * r2)

    U = U_ewald + U_diff + U_pauli
    force = f_ewald + f_diff + f_pauli

    return U, force
