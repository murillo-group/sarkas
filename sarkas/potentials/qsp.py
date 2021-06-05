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
import math as mt


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
    """
    Dev Notes
    for more info and a description of the potential's parameters.
    QSP_matrix[0,:,:] = qi*qj/4*pi*eps0
    QSP_matrix[1,:,:] = 2pi/deBroglie
    QSP_matrix[2,:,:] = e-e Pauli term factor
    QSP_matrix[3,:,:] = e-e Pauli term exponent term
    QSP_matrix[4,:,:] = Ewald parameter
    """
    # Do a bunch of checks
    # P3M algorithm only
    assert potential.method == "P3M", 'QSP interaction can only be calculated using P3M algorithm.'

    # Check for neutrality
    assert params.total_net_charge == 0, 'Total net charge is not zero.'

    # Default attributes
    if not hasattr(potential, 'qsp_type'):
        potential.qsp_type = 'Deutsch'
    if not hasattr(potential, 'qsp_pauli'):
        potential.qsp_pauli = True

    two_pi = 2.0 * np.pi
    four_pi = 2.0 * two_pi
    log2 = np.log(2.0)

    # Redefine ion temperatures and ion total number density
    mask = params.species_names != 'e'
    params.total_ion_temperature = params.species_concentrations[mask].transpose() * params.species_temperatures[mask]
    params.ni = np.sum(params.species_num_dens[mask])

    # Calculate the total and ion Wigner-Seitz Radius from the total density
    params.ai = (3.0 / (four_pi * params.ni)) ** (1.0 / 3.0)  # Ion WS

    deBroglie_const = two_pi * params.hbar2 / params.kB

    QSP_matrix = np.zeros((5, params.num_species, params.num_species))
    for i, name1 in enumerate(params.species_names):
        m1 = params.species_masses[i]
        q1 = params.species_charges[i]

        for j, name2 in enumerate(params.species_names):
            m2 = params.species_masses[j]
            q2 = params.species_charges[j]

            reduced = (m1 * m2) / (m1 + m2)

            if name1 == 'e' or name2 == 'e':
                # Use electron temperature in e-e and e-i interactions
                lambda_deB = np.sqrt(deBroglie_const / (reduced * params.electron_temperature))
            else:
                # Use ion temperature in i-i interactions only
                lambda_deB = np.sqrt(deBroglie_const / (reduced * params.total_ion_temperature))

            if name2 == name1:  # e-e
                QSP_matrix[2, i, j] = log2 * params.kB * params.electron_temperature
                QSP_matrix[3, i, j] = four_pi / (log2 * lambda_deB ** 2)

            QSP_matrix[0, i, j] = q1 * q2 / params.fourpie0
            QSP_matrix[1, i, j] = two_pi / lambda_deB

    if not potential.qsp_pauli:
        QSP_matrix[2, :, :] = 0.0

    QSP_matrix[4, :, :] = potential.pppm_alpha_ewald
    potential.matrix = QSP_matrix

    if potential.qsp_type.lower() == "deutsch":
        potential.force = deutsch_force
        # Calculate the PP Force error from the e-e diffraction term only.
        params.pppm_pp_err = np.sqrt(two_pi * potential.matrix[1, 0, 0])
        params.pppm_pp_err *= np.exp(- potential.rc * potential.matrix[1, 0, 0])
        params.pppm_pp_err *= params.a_ws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)

    elif potential.qsp_type.lower() == "kelbg":
        potential.force = kelbg_force
        # TODO: Calculate the PP Force error from the e-e diffraction term only.
        params.pppm_pp_err = np.sqrt(two_pi * potential.matrix[1, 0, 0])
        params.pppm_pp_err *= np.exp(- potential.rc * potential.matrix[1, 0, 0])


@njit
def deutsch_force(r, pot_matrix):
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
    f_ewald = U_ewald / r  # 1/r derivative
    f_ewald += A * (2.0 * alpha / np.sqrt(np.pi) ) * np.exp(- a2 * r2)/r  # erfc derivative

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
def kelbg_force(r, pot_matrix):
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
