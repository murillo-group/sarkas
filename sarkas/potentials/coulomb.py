r"""
Module for handling Coulomb interaction.

Potential
*********

The Coulomb potential between two particles :math:`a,b` is

.. math::
   U_{ab}(r) = \frac{q_{a}q_b}{4 \pi \epsilon_0 r}.

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.pot_matrix` are:

.. code-block::

    pot_matrix[0] : qi qj/(4pi esp0) Force factor between two particles.
    pot_matrix[1] : Ewald parameter in the case of pppm Algorithm. Same value for all species.
    pot_matrix[2] : Short-range cutoff. Same value for all species.

"""

from math import erfc
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import exp, inf, pi, sqrt, zeros

from ..utilities.maths import force_error_analytic_pp


def update_params(potential):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Potential's information

    """

    potential.matrix = zeros((3, potential.num_species, potential.num_species))
    potential.screening_length = inf
    potential.screening_length_type = "coulomb"
    for i, q1 in enumerate(potential.species_charges):
        for j, q2 in enumerate(potential.species_charges):
            potential.matrix[0, i, j] = q1 * q2 / potential.fourpie0

    if potential.method == "pp":
        potential.matrix[2, :, :] = potential.a_rs
        potential.force = coulomb_force
        potential.force_error = 0.0  # TODO: Implement force error in PP case
    elif potential.method == "pppm":
        potential.matrix[1, :, :] = potential.pppm_alpha_ewald
        potential.matrix[2, :, :] = potential.a_rs
        # Calculate the (total) plasma frequency
        potential.force = coulomb_force_pppm

        rescaling_constant = sqrt(potential.total_num_ptcls) * potential.a_ws**2 / sqrt(potential.pbox_volume)
        potential.pppm_pp_err = force_error_analytic_pp(
            potential.type, potential.rc, potential.screening_length, potential.pppm_alpha_ewald, rescaling_constant
        )
        # # PP force error calculation. Note that the equation was derived for a single component plasma.
        # alpha_times_rcut = -((potential.pppm_alpha_ewald * potential.rc) ** 2)
        # potential.pppm_pp_err = 2.0 * exp(alpha_times_rcut) / sqrt(potential.rc)
        # potential.pppm_pp_err *= sqrt(potential.total_num_ptcls) * potential.a_ws ** 2 / sqrt(potential.pbox_volume)


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def coulomb_force_pppm(r_in, pot_matrix):
    """
    Numba'd function to calculate the potential and force between two particles when the pppm algorithm is chosen.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables.\n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`) .

    Returns
    -------
    U : float
        Potential value.

    fr : float
        Force between two particles.


    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([ 1.0, 0.5, 0.0])
    >>> coulomb_force_pppm(r, pot_matrix)
    (0.07864960352514257, 0.14310167611771996)

    """

    # Short-range cutoff to deal with divergence of the Coulomb potential
    rs = pot_matrix[2]
    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    alpha = pot_matrix[1]  # Ewald parameter alpha
    alpha_r = alpha * r
    r2 = r * r
    U = pot_matrix[0] * erfc(alpha_r) / r
    f1 = erfc(alpha_r) / r2
    f2 = (2.0 * alpha / sqrt(pi) / r) * exp(-(alpha_r**2))
    fr = pot_matrix[0] * (f1 + f2)

    return U, fr


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def coulomb_force(r_in, pot_matrix):
    """
    Numba'd function to calculate the bare coulomb potential and force between two particles.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`) .

    Returns
    -------
    U : float
        Potential value.

    fr : float
        Force between two particles.

    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([ 1.0, 0.0, 0.0])
    >>> coulomb_force(r, pot_matrix)
    (0.5, 0.25)

    """

    # Short-range cutoff to deal with divergence of the Coulomb potential
    rs = pot_matrix[2]
    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    U = pot_matrix[0] / r
    fr = U / r

    return U, fr


def pretty_print_info(potential):
    """
    Print potential specific parameters in a user-friendly way.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """

    if potential.method != "fmm":
        fmm_msg = f"Short-range cutoff radius: a_rs = {potential.a_rs:.6e} {potential.units_dict['length']}"
    msg = f"Effective coupling constant: Gamma_eff = {potential.coupling_constant:.2f}"

    print(msg + fmm_msg)
