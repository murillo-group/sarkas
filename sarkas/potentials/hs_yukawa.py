"""
Module for handling Hard-Sphere Yukawa potential.

Potential
*********

The Hard-Sphere Yukawa potential between two charges :math:`q_i` and :math:`q_j` at distant :math:`r` is defined as

.. math::
    U_{ab}(r) = \\left ( \\frac{\\sigma}{r} \\right )^{n}  + \\frac{q_a q_b}{4 \\pi \\epsilon_0} \\frac{e^{- \\kappa r} }{r}.

where :math:`\\kappa = 1/\\lambda` is the screening parameter.

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.pot_matrix` are:

.. code-block:: python

    pot_matrix[0] = q_iq_j^2/(4 pi eps0)
    pot_matrix[1] = 1/lambda
    pot_matrix[2] = Ewald screening parameter
    pot_matrix[3] = short range cutoff
"""
from numba import njit
from numpy import exp, pi, sqrt
from numpy import zeros as np_zeros
from warnings import warn

from ..utilities.maths import force_error_analytic_lcl


@njit
def hs_yukawa_force(r, pot_matrix):
    """
    Calculates Potential and Force between two particles.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    U : float
        Potential.

    force : float
        Force between two particles.

    """
    U_y = pot_matrix[0] * exp(-pot_matrix[1] * r) / r
    U_hs = (pot_matrix[2] / r) ** (50)
    U = U_y + U_hs
    force = U_y * (1.0 / r + pot_matrix[1]) + (50.0) * U_hs / r

    return U, force


@njit
def force_deriv(r, pot_matrix):
    """Calculate the second derivative of the potential.

    Parameters
    ----------

    r : float
        Distance between particles

    pot_matrix : numpy.ndarray
        Values of the potential constants. \n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    f_dev : float
        Second derivative of potential.

    """
    kappa_r = pot_matrix[1] * r
    U2 = pot_matrix[0] * exp(-kappa_r) / r**3
    f_dev = U2 * (2.0 * (1.0 + kappa_r) + kappa_r**2)
    return f_dev


def update_params(potential, species):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.
    """
    # Potential specific parameters
    potential.packing_fraction = pi / 6.0 * potential.total_num_density * potential.hs_diameter**3

    if hasattr(potential, "kappa") and potential.screening_length is not None:
        warn(
            "You have defined both kappa and the screening_length. \n"
            "I will use kappa to calculate the screening_length from lambda = sigma/kappa"
        )
        potential.screening_length = potential.hs_diameter / potential.kappa

    elif hasattr(potential, "kappa"):
        potential.screening_length = potential.hs_diameter / potential.kappa
    elif potential.screening_length:
        potential.kappa = potential.hs_diameter / potential.screening_length

    # Interaction Matrix
    potential.matrix = np_zeros((3, potential.num_species, potential.num_species))
    potential.matrix[1, :, :] = 1.0 / potential.screening_length
    for i, sp1 in enumerate(species):
        for j, sp2 in enumerate(species):
            potential.matrix[0, i, j] = sp1.charge * sp2.charge / potential.fourpie0

    potential.matrix[2, :, :] = potential.hs_diameter

    if potential.method == "pp":
        # The rescaling constant is sqrt ( n sigma^4 ) = sqrt(  6 eta *sigma/pi )
        potential.force = hs_yukawa_force
        potential.force_error = force_error_analytic_lcl(
            "yukawa", potential.rc, potential.matrix, sqrt(6.0 * potential.packing_fraction * potential.hs_diameter / pi)
        )
        # # Force error calculated from eq.(43) in Ref.[1]_
        # potential.force_error = sqrt( TWOPI / potential.electron_TF_wavelength) * exp(- potential.rc / potential.electron_TF_wavelength)
        # # Renormalize
        # potential.force_error *= potential.a_ws ** 2 * sqrt(potential.total_num_ptcls / potential.pbox_volume)
    elif potential.method == "pppm":
        raise ValueError("PPPM algorithm not supported.")


def pretty_print_info(potential):
    """
    Print potential specific parameters in a user-friendly way.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """

    b = potential.hs_diameter / potential.a_ws
    print(f"hard sphere diameter = {b:.4f} a_ws = {potential.hs_diameter:.4e} ", end="")
    print("[cm]" if potential.units == "cgs" else "[m]")
    print(f"screening length = {potential.screening_length} ", end="")
    print("[cm]" if potential.units == "cgs" else "[m]")
    print(f"kappa = sigma/lambda = {potential.kappa:.4f}")
    print(f"reduced density = n sigma^3 = {potential.hs_diameter ** 3 * potential.total_num_density:.4f}")
    print(f"packing fraction = {potential.packing_fraction:.4f}")
    print(f"Gamma_eff = {potential.coupling_constant / b:.4f}")
