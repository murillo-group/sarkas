r"""
Module for handling Coulomb interaction.

Potential
*********

The Coulomb potential between two particles :math:`a,b` is

.. math::
   U_{ab}(r) = \frac{q_{a}q_b}{4 \pi \epsilon_0 r}.

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.matrix` are:

.. code-block::

    matrix[0] : qi qj/(4pi esp0) Force factor between two particles.
    matrix[1] : Ewald parameter in the case of pppm Algorithm. Same value for all species.
    matrix[2] : Short-range cutoff. Same value for all species.

"""

from math import erfc
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import exp, inf, pi, sqrt, zeros

from ..utilities.maths import force_error_analytic_pp


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
    u_r : float
        Potential value.

    f_r : float
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
    u_r = pot_matrix[0] * erfc(alpha_r) / r
    f1 = erfc(alpha_r) / r2
    f2 = (2.0 * alpha / sqrt(pi) / r) * exp(-(alpha_r**2))
    f_r = pot_matrix[0] * (f1 + f2)

    return u_r, f_r


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
        Shape = (:attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`, 3) .

    Returns
    -------
    u_r : float
        Potential value.

    f_r : float
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

    u_r = pot_matrix[0] / r
    f_r = u_r / r

    return u_r, f_r


def potential_derivatives(r, pot_matrix):
    """Calculate the first and second derivatives of the potential.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables.

    Returns
    -------
    u_r : float, numpy.ndarray
        Potential value.

    dv_dr : float, numpy.ndarray
        First derivative of the potential.

    d2v_dr2 : float, numpy.ndarray
        Second derivative of the potential.

    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([ 1.0, 0.0, 0.0])
    >>> potential_derivatives(r, pot_matrix)
    (0.5, 0.25)

    """

    u_r = pot_matrix[0] / r
    dv_dr = -u_r / r

    d2v_dr2 = 2.0 * dv_dr / r

    return u_r, dv_dr, d2v_dr2


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
    msg = f"Effective coupling constant: Gamma_eff = {potential.coupling_constant:.2f}\n"

    print(msg + fmm_msg)


def update_params(potential):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Potential's information

    """

    potential.screening_length = inf
    potential.screening_length_type = "coulomb"

    potential.matrix = zeros((potential.num_species, potential.num_species, 3))

    potential.potential_derivatives = potential_derivatives

    for i, q1 in enumerate(potential.species_charges):
        for j, q2 in enumerate(potential.species_charges):
            potential.matrix[i, j, 0] = q1 * q2 / potential.fourpie0

    if potential.method == "pp":
        potential.matrix[:, :, 2] = potential.a_rs
        potential.force = coulomb_force
        potential.force_error = 0.0  # TODO: Implement force error in PP case
    elif potential.method == "pppm":
        potential.matrix[:, :, 1] = potential.pppm_alpha_ewald
        potential.matrix[:, :, 2] = potential.a_rs
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
