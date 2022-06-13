"""
Module for handling Quantum Statistical Potentials.

Potential
*********

Quantum Statistical Potentials are defined by three terms

.. math::
    U(r) = U_{\\rm pauli}(r) + U_{\\rm coul} + U_{\\rm diff} (r)

where

.. math::
    U_{\\rm pauli}(r) = k_BT \\ln (2)  e^{ - 4\\pi r^2/ \\Lambda_{ab}^2 }

is due to the Pauli exclusion principle,

.. math::
    U_{\\rm coul}(r) = \\frac{q_iq_j}{4\\pi \\epsilon_0} \\frac{1}{r}

is the usual Coulomb interaction, and :math:`U_{\\rm diff}(r)` is a diffraction term.

There are two possibilities for the diffraction term. The most common is the Deutsch Potential

.. math::
    U_{\\rm deutsch}(r) = \\frac{q_aq_b}{4\\pi \\epsilon_0} \\frac{e^{- 2 \\pi r/\\Lambda_{ab}} }{r}.

The second most common form is the Kelbg potential

.. math::
    U_{\\rm kelbg}(r) = - \\frac{q_aq_b}{4\\pi \\epsilon_0} \\frac{1}{r} \\left [  e^{- 2 \\pi r^2/\\Lambda_{ab}^2 }
    - \\sqrt{2} \\pi \\dfrac{r}{\\Lambda_{ab}} \\textrm{erfc} \\left ( \\sqrt{ 2\\pi}  r/ \\Lambda_{ab} \\right )
    \\right ].

In the above equations the screening length :math:`\\Lambda_{ab}` is the thermal de Broglie wavelength
between the two charges defined as

.. math::
   \\Lambda_{ab} = \\sqrt{\\frac{2\\pi \\hbar^2}{\\mu_{ab} k_BT}}, \\quad  \\mu_{ab} = \\frac{m_a m_b}{m_a + m_b}


Note that in Ref. :cite:`Hansen1981` the DeBroglie wavelength is defined as

.. math::
   \\Lambda_{ee} = \\sqrt{ \\dfrac{\\hbar^2}{2 \\pi \\mu_{ee} k_{B} T}},

while in statistical physics textbooks is defined as

.. math::
   \\Lambda_{ee} = \\sqrt{ \\dfrac{2 \\pi \\hbar^2}{\\mu_{ee} k_{B} T}} .

The latter will be used in Sarkas. The difference is in the factor of :math:`2\\pi`, i.e. the difference between
a wave number and wave length.

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.pot_matrix` are:

.. code-block:: python

    pot_matrix[0] = qi*qj/4*pi*eps0
    pot_matrix[1] = 2pi/deBroglie
    pot_matrix[2] = e-e Pauli term factor
    pot_matrix[3] = e-e Pauli term exponent term
    pot_matrix[4] = Ewald parameter
    pot_matrix[5] = Short-range cutoff

"""

from math import erfc
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import exp, log, pi, sqrt, zeros
from warnings import warn

from ..utilities.exceptions import AlgorithmWarning
from ..utilities.maths import force_error_analytic_lcl, TWOPI


def update_params(potential, species):
    """
    Create potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    species : list, [:class:`sarkas.plasma.Species`,]
        Species' data.


    """
    # Do a bunch of checks
    # pppm algorithm only
    if potential.method != "pppm":
        raise ValueError("QSP interaction can only be calculated using pppm algorithm.")

    # Check for neutrality
    if potential.total_net_charge != 0:
        warn("Total net charge is not zero.", category=AlgorithmWarning)

    # Default attributes
    if not hasattr(potential, "qsp_type"):
        potential.qsp_type = "deutsch"
    if not hasattr(potential, "qsp_pauli"):
        potential.qsp_pauli = True

    # Enforce consistency
    potential.qsp_type = potential.qsp_type.lower()

    four_pi = 2.0 * TWOPI
    log_2 = log(2.0)

    # Redefine ion temperatures and ion total number density
    total_ion_temperature = 0.0
    total_ion_number_density = 0.0
    for is1, sp1 in enumerate(species[1:]):
        total_ion_temperature += sp1.concentration * sp1.temperature
        total_ion_number_density += sp1.number_density

    # Calculate the total and ion Wigner-Seitz Radius from the total density
    ai = (3.0 / (four_pi * total_ion_number_density)) ** (1.0 / 3.0)  # Ion WS

    deBroglie_const = TWOPI * potential.hbar**2 / potential.kB

    potential.matrix = zeros((6, potential.num_species, potential.num_species))
    for i, sp1 in enumerate(species):
        m1 = sp1.mass
        q1 = sp1.charge

        for j, sp2 in enumerate(species):
            m2 = sp2.mass
            q2 = sp2.charge

            reduced = (m1 * m2) / (m1 + m2)

            if sp1.name == "e" or sp2.name == "e":
                # Use electron temperature in e-e and e-i interactions
                lambda_deB = sqrt(deBroglie_const / (reduced * species[0].temperature))
            else:
                # Use ion temperature in i-i interactions only
                lambda_deB = sqrt(deBroglie_const / (reduced * total_ion_temperature))

            if sp1.name == sp2.name:  # e-e
                potential.matrix[2, i, j] = log_2 * potential.kB * sp1.temperature
                potential.matrix[3, i, j] = four_pi / (log_2 * lambda_deB**2)

            potential.matrix[0, i, j] = q1 * q2 / potential.fourpie0
            potential.matrix[1, i, j] = TWOPI / lambda_deB

    if not potential.qsp_pauli:
        potential.matrix[2, :, :] = 0.0

    potential.matrix[4, :, :] = potential.pppm_alpha_ewald
    potential.matrix[5, :, :] = potential.a_rs

    if potential.qsp_type == "deutsch":
        potential.force = deutsch_force

        # Calculate the LCL Force error from the e-e diffraction term only since it is the largest.
        potential.pppm_pp_err = force_error_analytic_lcl(
            potential.type, potential.rc, potential.matrix, sqrt(3.0 * potential.a_ws / (4.0 * pi))
        )
    elif potential.qsp_type == "kelbg":
        potential.force = kelbg_force
        # TODO: Calculate the PP Force error from the e-e diffraction term only.
        # the following is a placeholder
        potential.pppm_pp_err = force_error_analytic_lcl(
            potential.type, potential.rc, potential.matrix, sqrt(3.0 * potential.a_ws / (4.0 * pi))
        )


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def deutsch_force(r_in, pot_matrix):
    """
    Calculate Deutsch QSP Force between two particles.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (6, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

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
    rs = pot_matrix[5]

    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    a2 = alpha * alpha
    r2 = r * r

    # Ewald short-range potential and force terms
    U_ewald = A * erfc(alpha * r) / r
    f_ewald = U_ewald / r  # 1/r derivative
    f_ewald += A * (2.0 * alpha / sqrt(pi)) * exp(-a2 * r2) / r  # erfc derivative

    # Diffraction potential and force term
    U_diff = -A * exp(-C * r) / r
    f_diff = U_diff / r  # 1/r derivative
    f_diff += -A * C * exp(-C * r) / r  # exp derivative

    # Pauli potential and force terms
    U_pauli = D * exp(-F * r2)
    f_pauli = 2.0 * r * D * F * exp(-F * r2)

    U = U_ewald + U_diff + U_pauli
    force = f_ewald + f_diff + f_pauli

    return U, force


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def kelbg_force(r_in, pot_matrix):
    """
    Calculates the QSP Force between two particles when the pppm algorithm is chosen.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (6, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

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
    rs = pot_matrix[5]

    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    C2 = C * C
    a2 = alpha * alpha
    r2 = r * r

    # Ewald short-range potential and force terms
    U_ewald = A * erfc(alpha * r) / r
    f_ewald = U_ewald / r2  # 1/r derivative
    f_ewald += A * (2.0 * alpha / sqrt(pi) / r2) * exp(-a2 * r2)  # erfc derivative

    # Diffraction potential and force term
    U_diff = -A * exp(-C2 * r2 / pi) / r
    U_diff += A * C * erfc(C * r / sqrt(pi))

    f_diff = -A * (2.0 * C2 * r2 + pi) * exp(-C2 * r2 / pi) / (pi * r * r2)  # exp(r)/r derivative
    f_diff += 2.0 * A * C2 * exp(-C2 * r2 / pi) / r / pi  # erfc derivative

    # Pauli Term
    U_pauli = D * exp(-F * r2)
    f_pauli = 2.0 * D * F * exp(-F * r2)

    U = U_ewald + U_diff + U_pauli
    force = f_ewald + f_diff + f_pauli

    return U, force


def pretty_print_info(potential):
    """
    Print potential specific parameters in a user-friendly way.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """

    ii_scr_len = 1.0 / potential.matrix[1, 1, 1]
    ei_scr_len = 1.0 / potential.matrix[1, 0, 1]
    ee_scr_len = 1.0 / potential.matrix[1, 0, 0]
    e_deBroglie_lambda = sqrt(2.0) * pi / potential.matrix[1, 0, 0]
    i_deBroglie_lambda = sqrt(2.0) * pi / potential.matrix[1, 1, 1]
    a_ws = potential.a_ws

    print(f"QSP type: {potential.qsp_type}")
    print(f"Pauli term: {potential.qsp_pauli}")
    print(f"e de Broglie wavelength = {e_deBroglie_lambda / a_ws:.4f} a_ws = {e_deBroglie_lambda:.6e} ", end="")
    print("[cm]" if potential.units == "cgs" else "[m]")
    print(f"ion de Broglie wavelength  = {i_deBroglie_lambda / a_ws:.4f} a_ws = {i_deBroglie_lambda:.6e} ", end="")
    print("[cm]" if potential.units == "cgs" else "[m]")
    print(f"e-e screening length = {ee_scr_len / a_ws:.4f} a_ws = {ee_scr_len:.6e} ", end="")
    print("[cm]" if potential.units == "cgs" else "[m]")
    print(f"i-i screening length = {ii_scr_len / a_ws:.4f} a_ws = {ii_scr_len:.6e} ", end="")
    print("[cm]" if potential.units == "cgs" else "[m]")
    print(f"e-i screening length = {ei_scr_len / a_ws:.4f} a_ws = {ei_scr_len:.6e} ", end="")
    print("[cm]" if potential.units == "cgs" else "[m]")
    print(f"e-i coupling constant = {potential.coupling_constant:.4f}")
