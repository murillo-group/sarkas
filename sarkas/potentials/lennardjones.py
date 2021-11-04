"""
Module for handling Lennard-Jones interaction.

Potential
*********

The generalized Lennard-Jones potential is defined as

.. math::
    U_{\\mu\\nu}(r) = k \\epsilon_{\\mu\\nu} \\left [ \\left ( \\frac{\\sigma_{\\mu\\nu}}{r}\\right )^m -
    \\left ( \\frac{\\sigma_{\\mu\\nu}}{r}\\right )^n \\right ],

where

.. math::
    k = \\frac{n}{m-n} \\left ( \\frac{n}{m} \\right )^{\\frac{m}{n-m}}.

In the case of multispecies liquids we use the `Lorentz-Berthelot <https://en.wikipedia.org/wiki/Combining_rules>`_
mixing rules

.. math::
    \\epsilon_{12} = \\sqrt{\epsilon_{11} \\epsilon_{22}}, \\quad \\sigma_{12} = \\frac{\\sigma_{11} + \\sigma_{22}}{2}.

Force Error
***********

The force error for the LJ potential is given by

.. math::
    \\Delta F = \\frac{k\\epsilon}{ \\sqrt{2\\pi n}} \\left [ \\frac{m^2 \\sigma^{2m}}{2m - 1} \\frac{1}{r_c^{2m -1}}
    + \\frac{n^2 \\sigma^{2n}}{2n - 1} \\frac{1}{r_c^{2n -1}} \\
    -\\frac{2 m n \\sigma^{m + n}}{m + n - 1} \\frac{1}{r_c^{m + n -1}} \\
    \\right ]^{1/2}

which we approximate with the first term only

.. math::
    \\Delta F \\approx \\frac{k\\epsilon} {\\sqrt{2\\pi n} }
    \\left [ \\frac{m^2 \\sigma^{2m}}{2m - 1} \\frac{1}{r_c^{2m -1}} \\right ]^{1/2}

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.pot_matrix` are:

.. code-block::

    pot_matrix[0] = epsilon_12 * lj_constant
    pot_matrix[1] = sigmas
    pot_matrix[2] = highest power
    pot_matrix[3] = lowest power
    pot_matrix[4] = short-range cutoff

"""
import numpy as np
import numba as nb
from sarkas.utilities.maths import force_error_analytic_pp


def update_params(potential, params):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : sarkas.potentials.core.Potential
        Class handling potential form.

    params : sarkas.core.Parameters
        Simulation's parameters.

    """
    potential.matrix = np.zeros((5, params.num_species, params.num_species))
    # See Lima Physica A 391 4281 (2012) for the following definitions
    if not hasattr(potential, "powers"):
        potential.powers = np.array([12, 6])

    exponent = potential.powers[0] / (potential.powers[1] - potential.powers[0])
    lj_constant = potential.powers[1] / (potential.powers[0] - potential.powers[1])
    lj_constant *= (potential.powers[1] / potential.powers[0]) ** exponent

    # Use the Lorentz-Berthelot mixing rules.
    # Lorentz: sigma_12 = 0.5 * (sigma_1 + sigma_2)
    # Berthelot: epsilon_12 = sqrt( eps_1 eps2)
    sigma2 = 0.0
    epsilon_tot = 0.0
    # Recall that species_charges contains sqrt(epsilon)
    for i, q1 in enumerate(params.species_charges):
        for j, q2 in enumerate(params.species_charges):
            potential.matrix[0, i, j] = lj_constant * q1 * q2
            potential.matrix[1, i, j] = 0.5 * (params.species_lj_sigmas[i] + params.species_lj_sigmas[j])
            potential.matrix[2, i, j] = potential.powers[0]
            potential.matrix[3, i, j] = potential.powers[1]
            sigma2 += params.species_lj_sigmas[i]
            epsilon_tot += q1 * q2

    potential.matrix[4, :, :] = potential.a_rs

    potential.force = lj_force

    # The rescaling constant is sqrt ( na^4 ) = sqrt( 3 a/(4pi) )
    params.force_error = force_error_analytic_pp(
        potential.type, potential.rc, potential.matrix, np.sqrt(3.0 * params.a_ws / (4.0 * np.pi))
    )


@nb.njit
def lj_force(r_in, pot_matrix):
    """
    Calculates the PP force between particles using Lennard-Jones Potential.

    Parameters
    ----------
    r_in : float
        Particles' distance.

    pot_matrix : numpy.ndarray
        LJ potential parameters. \n
        Shape = (5, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    U : float
        Potential.

    force : float
        Force.

    """

    rs = pot_matrix[4]
    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    epsilon = pot_matrix[0]
    sigma = pot_matrix[1]
    s_over_r = sigma / r
    s_over_r_high = s_over_r ** pot_matrix[2]
    s_over_r_low = s_over_r ** pot_matrix[3]

    U = epsilon * (s_over_r_high - s_over_r_low)
    force = epsilon * (pot_matrix[2] * s_over_r_high - pot_matrix[3] * s_over_r_low) / r

    return U, force
