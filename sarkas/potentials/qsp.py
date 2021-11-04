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

from warnings import warn
import numpy as np
from numba import njit
import math as mt

from sarkas.utilities.exceptions import AlgorithmWarning
from sarkas.utilities.maths import force_error_analytic_pp, TWOPI


def update_params(potential, params):
    """
    Create potential dependent simulation's parameters.

    Parameters
    ----------
    potential : sarkas.potentials.core.Potential
        Class handling potential form.

    params : object
        Simulation's parameters


    """
    # Do a bunch of checks
    # pppm algorithm only
    if potential.method != "pppm":
        raise ValueError("QSP interaction can only be calculated using pppm algorithm.")

    # Check for neutrality
    if params.total_net_charge != 0:
        warn("Total net charge is not zero.", category=AlgorithmWarning)

    # Default attributes
    if not hasattr(potential, "qsp_type"):
        potential.qsp_type = "deutsch"
    if not hasattr(potential, "qsp_pauli"):
        potential.qsp_pauli = True

    # Enforce consistency
    potential.qsp_type = potential.qsp_type.lower()

    four_pi = 2.0 * TWOPI
    log2 = np.log(2.0)

    # Redefine ion temperatures and ion total number density
    mask = params.species_names != "e"
    params.total_ion_temperature = params.species_concentrations[mask].transpose() * params.species_temperatures[mask]
    params.ni = np.sum(params.species_num_dens[mask])

    # Calculate the total and ion Wigner-Seitz Radius from the total density
    params.ai = (3.0 / (four_pi * params.ni)) ** (1.0 / 3.0)  # Ion WS

    deBroglie_const = TWOPI * params.hbar2 / params.kB

    potential.matrix = np.zeros((6, params.num_species, params.num_species))
    for i, name1 in enumerate(params.species_names):
        m1 = params.species_masses[i]
        q1 = params.species_charges[i]

        for j, name2 in enumerate(params.species_names):
            m2 = params.species_masses[j]
            q2 = params.species_charges[j]

            reduced = (m1 * m2) / (m1 + m2)

            if name1 == "e" or name2 == "e":
                # Use electron temperature in e-e and e-i interactions
                lambda_deB = np.sqrt(deBroglie_const / (reduced * params.electron_temperature))
            else:
                # Use ion temperature in i-i interactions only
                lambda_deB = np.sqrt(deBroglie_const / (reduced * params.total_ion_temperature))

            if name2 == name1:  # e-e
                potential.matrix[2, i, j] = log2 * params.kB * params.electron_temperature
                potential.matrix[3, i, j] = four_pi / (log2 * lambda_deB ** 2)

            potential.matrix[0, i, j] = q1 * q2 / params.fourpie0
            potential.matrix[1, i, j] = TWOPI / lambda_deB

    if not potential.qsp_pauli:
        potential.matrix[2, :, :] = 0.0

    potential.matrix[4, :, :] = potential.pppm_alpha_ewald
    potential.matrix[5, :, :] = potential.a_rs

    if potential.qsp_type == "deutsch":
        potential.force = deutsch_force
        # Calculate the PP Force error from the e-e diffraction term only.
        params.pppm_pp_err = force_error_analytic_pp(
            potential.type, potential.rc, potential.matrix, np.sqrt(3.0 * params.a_ws / (4.0 * np.pi))
        )
    elif potential.qsp_type == "kelbg":
        potential.force = kelbg_force
        # TODO: Calculate the PP Force error from the e-e diffraction term only.
        # the following is a placeholder
        params.pppm_pp_err = force_error_analytic_pp(
            potential.type, potential.rc, potential.matix, np.sqrt(3.0 * params.a_ws / (4.0 * np.pi))
        )


@njit
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
    U_ewald = A * mt.erfc(alpha * r) / r
    f_ewald = U_ewald / r  # 1/r derivative
    f_ewald += A * (2.0 * alpha / np.sqrt(np.pi)) * np.exp(-a2 * r2) / r  # erfc derivative

    # Diffraction potential and force term
    U_diff = -A * np.exp(-C * r) / r
    f_diff = U_diff / r  # 1/r derivative
    f_diff += -A * C * np.exp(-C * r) / r  # exp derivative

    # Pauli potential and force terms
    U_pauli = D * np.exp(-F * r2)
    f_pauli = 2.0 * r * D * F * np.exp(-F * r2)

    U = U_ewald + U_diff + U_pauli
    force = f_ewald + f_diff + f_pauli

    return U, force


@njit
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
    U_ewald = A * mt.erfc(alpha * r) / r
    f_ewald = U_ewald / r2  # 1/r derivative
    f_ewald += A * (2.0 * alpha / np.sqrt(np.pi) / r2) * np.exp(-a2 * r2)  # erfc derivative

    # Diffraction potential and force term
    U_diff = -A * np.exp(-C2 * r2 / np.pi) / r
    U_diff += A * C * mt.erfc(C * r / np.sqrt(np.pi))

    f_diff = -A * (2.0 * C2 * r2 + np.pi) * np.exp(-C2 * r2 / np.pi) / (np.pi * r * r2)  # exp(r)/r derivative
    f_diff += 2.0 * A * C2 * np.exp(-C2 * r2 / np.pi) / r / np.pi  # erfc derivative

    # Pauli Term
    U_pauli = D * np.exp(-F * r2)
    f_pauli = 2.0 * D * F * np.exp(-F * r2)

    U = U_ewald + U_diff + U_pauli
    force = f_ewald + f_diff + f_pauli

    return U, force
