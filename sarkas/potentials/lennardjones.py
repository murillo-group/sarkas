"""
Module for handling Lennard-Jones interaction
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

    Notes
    -----
    The force error for the LJ potential is given by

    .. math::

        \Delta F = \frac{k\epsilon \sqrt{2\pi n} }\left [ \frac{m^2 \sigma^{2m}}{2m - 1} \frac{1}{r_c^{2m -1}} \right . \\
        + \frac{n^2 \sigma^{2n}}{2n - 1} \frac{1}{r_c^{2n -1}} \\
        \left . -\frac{2 m n \sigma^{m + n}}{m + n - 1} \frac{1}{r_c^{m + n -1}} \\
        \right ]^{1/2}

    which we approximate with the first term only

    .. math::

        \Delta F \approx \frac{k\epsilon \sqrt{2\pi n} }\left [ \frac{m^2 \sigma^{2m}}{2m - 1} \frac{1}{r_c^{2m -1}} \right ]^{1/2}
        \right ]^{1/2}

    """
    potential.matrix = np.zeros((5, params.num_species, params.num_species))
    # See Lima Physica A 391 4281 (2012) for the following definitions
    if not hasattr(potential, 'powers'):
        potential.powers = np.array([12, 6])

    exponent = potential.powers[0] / (potential.powers[1] - potential.powers[0])
    lj_constant = potential.powers[1]/(potential.powers[0] - potential.powers[1])
    lj_constant *= (potential.powers[1]/potential.powers[0]) ** exponent

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

    potential.matrix[4, :, :] = potential.rs

    potential.force = lj_force

    # The rescaling constant is sqrt ( na^4 ) = sqrt( 3 a/(4pi) )
    params.force_error = force_error_analytic_pp(
        potential.type,
        potential.rc,
        potential.matrix,
        np.sqrt(3.0 * params.a_ws / (4.0 * np.pi))
    )

@nb.njit
def lj_force(r, pot_matrix):
    """
    Calculates the PP force between particles using Lennard-Jones Potential.

    Parameters
    ----------
    pot_matrix : array
        LJ potential parameters.

    r : float
        Particles' distance.


    Returns
    -------
    U : float
        Potential.

    force : float
        Force.
    """
    """
    Notes
    -----
    pot_matrix[0] = epsilon_12 * lj_constant
    pot_matrix[1] = sigmas
    pot_matrix[2] = highest power
    pot_matrix[3] = lowest power
    pot_matrix[4] = short-range cutoff

    """

    rs = pot_matrix[4]
    if r < rs:
        r = rs

    epsilon = pot_matrix[0]
    sigma = pot_matrix[1]
    s_over_r = sigma / r
    s_over_r_high = s_over_r ** pot_matrix[2]
    s_over_r_low = s_over_r ** pot_matrix[3]

    U = epsilon * (s_over_r_high - s_over_r_low)
    force = epsilon * (pot_matrix[2] * s_over_r_high - pot_matrix[3] * s_over_r_low) / r

    return U, force
