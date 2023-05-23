r"""
Module for handling Moliere Potential.

Potential
*********

The Moliere Potential is defined as

.. math::
    U(r) = \frac{q_i q_j}{4 \pi \epsilon_0} \frac{1}{r} \sum_{\alpha} C_{\alpha} e^{- b_{\alpha} r}.

For more details see :cite:`Wilson1977`. Note that the parameters :math:`b` are not normalized by the Bohr radius.
They should be passed with the correct units [m] if mks or [cm] if cgs.

Force Error
***********

The force error is calculated from the Yukawa's formula with the smallest screening length.

.. math::

    \Delta F = \frac{q^2}{4 \pi \epsilon_0} \sqrt{2 \pi n b_{\textrm min} }e^{-b_{\textrm min} r_c},

This overestimates it, but it doesn't matter.

Potential Attributes
********************
The elements of the :attr:`sarkas.potentials.core.Potential.pot_matrix` are:

.. code-block:: python

    pot_matrix[0] = q_iq_je^2/(4 pi eps_0) Force factor between two particles.
    pot_matrix[1] = C_1
    pot_matrix[2] = C_2
    pot_matrix[3] = C_3
    pot_matrix[4] = b_1
    pot_matrix[5] = b_2
    pot_matrix[6] = b_3

"""
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import array, exp, pi, sqrt, zeros

from ..utilities.maths import force_error_analytic_lcl


def update_params(potential):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """
    potential.fit_params = array(potential.fit_params)
    params_len = len(potential.fit_params)

    potential.matrix = zeros((params_len + 1, potential.num_species, potential.num_species))

    for i, q1 in enumerate(potential.species_charges):
        for j, q2 in enumerate(potential.species_charges):
            potential.matrix[0, i, j] = q1 * q2 / potential.fourpie0
            potential.matrix[1 : params_len + 1, i, j] = potential.fit_params

    potential.force = fit_force
    # Use Yukawa force error formula with the smallest screening length.
    # This overestimates the Force error, but it doesn't matter.
    # The rescaling constant is sqrt ( na^4 ) = sqrt( 3 a/(4pi) )

    _, f_rc = fit_force(potential.rc, potential.matrix[:, 0, 0])
    _, f_2a = fit_force(2.0 * potential.a_ws, potential.matrix[:, 0, 0])
    potential.force_error = f_rc / f_2a
    # force_error_analytic_lcl(
    #    potential.type, potential.rc, potential.matrix[:, :, :], sqrt(3.0 * potential.a_ws / (4.0 * pi))
    # )


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def fit_force(r, pot_matrix):
    """
    Numba'd function to calculate the PP force between particles using the Moliere Potential.

    Parameters
    ----------
    r : float
        Particles' distance.

    pot_matrix : numpy.ndarray
        Moliere potential parameters. \n
        Shape = (7, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    U : float
        Potential.

    force : float
        Force between two particles.

    Examples
    --------
    >>> from scipy.constants import epsilon_0, pi, elementary_charge
    >>> from numpy import array, zeros
    >>> charge = 4.0 * elementary_charge  # = 4e [C] mks units
    >>> coul_const = 1.0/ (4.0 * pi * epsilon_0)
    >>> screening_charges = array([0.5, -0.5, 1.0])
    >>> screening_lengths = array([5.99988000e-11, 1.47732309e-11, 1.47732309e-11])  # [m]
    >>> params_len = len(screening_lengths)
    >>> pot_mat = zeros(2 * params_len + 1)
    >>> pot_mat[0] = coul_const * charge**2
    >>> pot_mat[1: params_len + 1] = screening_charges.copy()
    >>> pot_mat[params_len + 1:] = 1./screening_lengths
    >>> r = 6.629755e-10  # [m] particles distance
    >>> moliere_force(r, pot_mat)
    (4.423663010052846e-23, 6.672438139145769e-14)

    """

    # Unpack the parameters
    q2_e0 = pot_matrix[0]
    kappa = pot_matrix[1]
    a = pot_matrix[2]
    b = pot_matrix[3]
    c = pot_matrix[4]
    d = pot_matrix[5]

    yukawa = exp(-kappa * r) / r
    denom = a + b * exp(c * r - c * d)

    U = yukawa / denom

    # d/dr 1/r
    f1 = U / r
    # d/dr exp(-kappa r)
    f2 = kappa * U
    # d/dr denom
    f3 = U / denom * (b * c * exp(c * r - c * d))

    force = f1 + f2 + f3
    force *= q2_e0
    U *= q2_e0

    return U, force


def pretty_print_info(potential):
    """
    Print potential specific parameters in a user-friendly way.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """
    msg = (
        f"screening type : {potential.screening_length_type}\n"
        f"screening length = {potential.screening_length:.6e} {potential.units_dict['length']}\n"
        f"kappa = {potential.a_ws / potential.screening_length:.4f}\n"
        f"Fit params:\n"
        f"kappa = {potential.matrix[1, 0,0]*potential.a_ws:.3f} / a_ws = {potential.matrix[1, 0,0]:.6e} {potential.units_dict['inverse length']}\n"
        f"a = {potential.matrix[2,0,0]}\n"
        f"b = {potential.matrix[3,0,0]}\n"
        f"c = {potential.matrix[4,0,0] * potential.a_ws:.3f} /a_ws => {potential.matrix[4, 0,0]:.6e} {potential.units_dict['inverse length']}\n"
        f"d = {potential.matrix[5,0,0] / potential.a_ws:.3f} a_ws => {potential.matrix[4, 0,0]:.6e} {potential.units_dict['length']}\n"
        f"Gamma_eff = {potential.coupling_constant:.2f}"
    )
    print(msg)
