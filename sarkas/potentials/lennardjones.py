r"""
Module for handling Lennard-Jones interaction.

Potential
*********

The generalized Lennard-Jones potential is defined as

.. math::
    U_{\mu\nu}(r) = k \epsilon_{\mu\nu} \left [ \left ( \frac{\sigma_{\mu\nu}}{r}\right )^m -
    \left ( \frac{\sigma_{\mu\nu}}{r}\right )^n \right ],

where

.. math::
    k = \frac{n}{m-n} \left ( \frac{n}{m} \right )^{\frac{m}{n-m}}.

In the case of multispecies liquids we use the `Lorentz-Berthelot <https://en.wikipedia.org/wiki/Combining_rules>`_
mixing rules

.. math::
    \epsilon_{12} = \sqrt{\epsilon_{11} \epsilon_{22}}, \quad \sigma_{12} = \frac{\sigma_{11} + \sigma_{22}}{2}.

Force Error
***********

The force error for the LJ potential is given by

.. math::
    \Delta F = \frac{k\epsilon}{ \sqrt{2\pi n}} \left [ \frac{m^2 \sigma^{2m}}{2m - 1} \frac{1}{r_c^{2m -1}}
    + \frac{n^2 \sigma^{2n}}{2n - 1} \frac{1}{r_c^{2n -1}} \
    -\frac{2 m n \sigma^{m + n}}{m + n - 1} \frac{1}{r_c^{m + n -1}} \
    \right ]^{1/2}

which we approximate with the first term only

.. math::
    \Delta F \approx \frac{k\epsilon} {\sqrt{2\pi n} }
    \left [ \frac{m^2 \sigma^{2m}}{2m - 1} \frac{1}{r_c^{2m -1}} \right ]^{1/2}

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
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import array, pi, sqrt, zeros

from ..utilities.maths import force_error_analytic_lcl


def update_params(potential):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.
    """
    potential.matrix = zeros((5, potential.num_species, potential.num_species))
    # See Lima Physica A 391 4281 (2012) for the following definitions
    if not hasattr(potential, "powers"):
        potential.powers = array([12, 6])

    exponent = potential.powers[0] / (potential.powers[1] - potential.powers[0])
    lj_constant = potential.powers[1] / (potential.powers[0] - potential.powers[1])
    lj_constant *= (potential.powers[1] / potential.powers[0]) ** exponent

    # Use the Lorentz-Berthelot mixing rules.
    # Lorentz: sigma_12 = 0.5 * (sigma_1 + sigma_2)
    # Berthelot: epsilon_12 = sqrt( eps_1 eps2)
    potential.epsilon_tot = 0.0
    # Recall that species_charges contains sqrt(epsilon)
    for i, q1 in enumerate(potential.species_charges):
        for j, q2 in enumerate(potential.species_charges):
            potential.matrix[0, i, j] = lj_constant * q1 * q2
            potential.matrix[1, i, j] = 0.5 * (potential.species_lj_sigmas[i] + potential.species_lj_sigmas[j])
            potential.matrix[2, i, j] = potential.powers[0]
            potential.matrix[3, i, j] = potential.powers[1]

            potential.epsilon_tot += q1 * q2

    potential.sigma_avg = potential.species_lj_sigmas.mean()
    potential.matrix[4, :, :] = potential.a_rs

    potential.force = lj_force

    # The rescaling constant is sqrt ( na^4 ) = sqrt( 3 a/(4pi) )
    potential.force_error = force_error_analytic_lcl(
        potential.type, potential.rc, potential.matrix, sqrt(3.0 * potential.a_ws / (4.0 * pi))
    )


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def lj_force(r_in, pot_matrix):
    """
    Numba'd function to calculate the PP force between particles using Lennard-Jones Potential.

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

    Examples
    --------
    >>> pot_const = 4.0 * 1.656e-21 # 4*epsilon in [J] (mks units)
    >>> sigma = 3.4e-10   # [m] (mks units)
    >>> high_pow, low_pow = 12., 6.
    >>> short_cutoff = 0.0001 * sigma
    >>> pot_mat = array([pot_const, sigma, high_pow, low_pow, short_cutoff])
    >>> r = 15.0 * sigma  # particles' distance in [m]
    >>> lj_force(r, pot_mat)
    (-5.815308131440668e-28, -6.841538377536503e-19)

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


def pretty_print_info(potential):
    """
    Print potential specific parameters in a user-friendly way.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """

    print(f"epsilon_tot = {potential.epsilon_tot/potential.eV2J:.6e} [eV] = {potential.epsilon_tot:6e} ", end="")
    print("[erg]" if potential.units == "cgs" else "[J]")
    print(f"sigma_avg = {potential.sigma_avg/potential.a_ws:.6e} a_ws =  {potential.sigma_avg:6e} ", end="")
    print("[cm]" if potential.units == "cgs" else "[m]")
    rho = potential.sigma_avg**3 * potential.total_num_density
    tau = potential.kB * potential.T_desired / potential.epsilon_tot
    print(f"reduced density = {rho:.6e}")
    print(f"reduced temperature = {tau:.6e}")
