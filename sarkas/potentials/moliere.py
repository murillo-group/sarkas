"""
Module for handling Moliere Potential.

Potential
*********

The Moliere Potential is defined as

.. math::
    U(r) = \\frac{q_i q_j}{4 \\pi \\epsilon_0} \\frac{1}{r} \\sum_{\\alpha} C_{\\alpha} e^{- b_{\\alpha} r}.

For more details see :cite:`Wilson1977`. Note that the parameters :math:`b` are not normalized by the Bohr radius.
They should be passed with the correct units [m] if mks or [cm] if cgs.

Force Error
***********

The force error is calculated from the Yukawa's formula with the smallest screening length.

.. math::

    \\Delta F = \\frac{q^2}{4 \\pi \\epsilon_0} \\sqrt{2 \\pi n b_{\\textrm min} }e^{-b_{\\textrm min} r_c},

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
    potential.screening_lengths = np.array(potential.screening_lengths)
    potential.screening_charges = np.array(potential.screening_charges)
    params_len = len(potential.screening_lengths)

    potential.matrix = np.zeros((2 * params_len + 1, params.num_species, params.num_species))

    for i, q1 in enumerate(params.species_charges):
        for j, q2 in enumerate(params.species_charges):

            potential.matrix[0, i, j] = q1 * q2 / params.fourpie0
            potential.matrix[1 : params_len + 1, i, j] = potential.screening_charges
            potential.matrix[params_len + 1 :, i, j] = potential.screening_lengths

    potential.force = moliere_force
    # Use Yukawa force error formula with the smallest screening length.
    # This overestimates the Force error, but it doesn't matter.
    # The rescaling constant is sqrt ( na^4 ) = sqrt( 3 a/(4pi) )
    params.force_error = force_error_analytic_pp(
        potential.type, potential.rc, potential.matrix[params_len + 1 :, :, :], np.sqrt(3.0 * params.a_ws / (4.0 * np.pi))
    )


@nb.njit
def moliere_force(r, pot_matrix):
    """
    Calculates the PP force between particles using the Moliere Potential.

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

    """

    U = 0.0
    force = 0.0

    for i in range(int(len(pot_matrix[:-1]) / 2)):
        factor1 = r * pot_matrix[i + 4]
        factor2 = pot_matrix[i + 1] / r
        U += factor2 * np.exp(-factor1)
        force += np.exp(-factor1) * factor2 * (1.0 / r + pot_matrix[i + 1])

    force *= pot_matrix[0]
    U *= pot_matrix[0]

    return U, force
