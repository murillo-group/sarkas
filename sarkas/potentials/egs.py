"""
Module for handling Exact Gradient corrected Screened (EGS) Potential
"""
import numpy as np
from numba import njit
import fdint


def update_params(potential, params):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : sarkas.potentials.core.Potential
        Class handling potential form.

    params: sarkas.core.Parameters
        Simulation's parameters.

    """

    twopi = 2.0 * np.pi

    # lambda factor : 1 = von Weizsaecker, 1/9 = Thomas-Fermi
    if not hasattr(potential, 'lmbda'):
        potential.lmbda = 1.0 / 9.0

    fdint_dfdk_vec = np.vectorize(fdint.dfdk)
    # eq. (14) of Ref. [1]_
    params.nu = 3.0/np.pi ** 1.5 * params.landau_length / params.lambda_deB
    params.nu *= potential.lmbda * fdint_dfdk_vec(k=-0.5, phi=params.eta_e)

    # Degeneracy Parameter
    theta = params.electron_degeneracy_parameter
    if 0.1 <= theta <= 12:
        # Regime of validity of the following approximation Perrot et al. Phys Rev A 302619 (1984)
        # eq. (33) of Ref. [1]_
        Ntheta = 1.0 + 2.8343 * theta ** 2 - 0.2151 * theta ** 3 + 5.2759 * theta ** 4
        # eq. (34) of Ref. [1]_
        Dtheta = 1.0 + 3.9431 * theta ** 2 + 7.9138 * theta ** 4
        # eq. (32) of Ref. [1]_
        h = Ntheta / Dtheta * np.tanh(1.0 / theta)
        # grad h(x)
        gradh = (-(Ntheta / Dtheta) / np.cosh(1 / theta) ** 2 / (theta ** 2)  # derivative of tanh(1/x)
                 - np.tanh(1.0 / theta) * (
                         Ntheta * (7.8862 * theta + 31.6552 * theta ** 3) / Dtheta ** 2  # derivative of 1/Dtheta
                         + (5.6686 * theta - 0.6453 * theta ** 2 + 21.1036 * theta ** 3) / Dtheta) # derivative of Ntheta
                 )
        # eq.(31) of Ref. [1]_
        b = 1.0 - 2.0 / ( 8.0 * (params.kF * params.lambda_TF)**2) * (h - 2.0 * theta * gradh)
    else:
        b = 1.0

    params.b = b

    # Monotonic decay
    if params.nu <= 1:
        # eq. (29) of Ref. [1]_
        params.lambda_p = params.lambda_TF * np.sqrt(params.nu / (2.0 * b + 2.0 * np.sqrt(b ** 2 - params.nu)))
        params.lambda_m = params.lambda_TF * np.sqrt(params.nu / (2.0 * b - 2.0 * np.sqrt(b ** 2 - params.nu)))
        params.alpha = b / np.sqrt(b - params.nu)

    # Oscillatory behavior
    if params.nu > 1:
        # eq. (29) of Ref. [1]_
        params.gamma_m = params.lambda_TF * np.sqrt(params.nu / (np.sqrt(params.nu) - b))
        params.gamma_p = params.lambda_TF * np.sqrt(params.nu / (np.sqrt(params.nu) + b))
        params.alphap = b / np.sqrt(params.nu - b)

    potential.matrix = np.zeros((6, params.num_species, params.num_species))

    potential.matrix[1, :, :] = params.nu

    for i, q1 in enumerate(params.species_charges):

        for j, q2 in enumerate(params.species_charges):

            if params.nu <= 1:
                potential.matrix[0, i, j] = q1 * q2 / (2.0 * params.fourpie0)
                potential.matrix[2, i, j] = (1.0 + params.alpha)
                potential.matrix[3, i, j] = (1.0 - params.alpha)
                potential.matrix[4, i, j] = 1.0 / params.lambda_m
                potential.matrix[5, i, j] = 1.0 / params.lambda_p

            if params.nu > 1:
                potential.matrix[0, i, j] = q1 * q2 / params.fourpie0
                potential.matrix[2, i, j] = 1.0
                potential.matrix[3, i, j] = params.alphap
                potential.matrix[4, i, j] = 1.0 / params.gamma_m
                potential.matrix[5, i, j] = 1.0 / params.gamma_p

    assert potential.method == "PP", "P3M Algorithm not implemented yet. Good Bye!"

    potential.force = egs_force
    params.force_error = np.sqrt(twopi / params.lambda_TF) * np.exp(-potential.rc / params.lambda_TF)
    # Renormalize
    params.force_error *= params.a_ws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)


@njit
def egs_force(r, pot_matrix):
    """ 
    Calculates Potential and force between particles using the EGS Potential.
    
    Parameters
    ----------
    r : float
        Particles' distance.

    pot_matrix : array
        EGS potential parameters. 

    Return
    ------

    U : float
        Potential.

    fr : float
        Force.

    """
    # nu = pot_matrix[1]
    if pot_matrix[1] <= 1.0:
        # pot_matrix[0] = Charge factor = q^2/4pi eps0 if mks q^2 if cgs
        # pot_matrix[2] = 1 + alpha
        # pot_matrix[3] = 1 - alpha
        # pot_matrix[4] = 1.0 / lambda_minus
        # pot_matrix[5] = 1.0 / lambda_plus

        temp1 = pot_matrix[2] * np.exp(-r * pot_matrix[4])
        temp2 = pot_matrix[3] * np.exp(-r * pot_matrix[5])
        # Potential
        U = (temp1 + temp2) * pot_matrix[0] / r
        # Force
        fr = U / r + pot_matrix[0] * (temp1 * pot_matrix[4] + temp2 * pot_matrix[5]) / r

    else:
        # pot_matrix[0] = Charge factor
        # pot_matrix[2] = 1.0
        # pot_matrix[3] = alpha prime
        # pot_matrix[4] = 1.0 / gamma_minus
        # pot_matrix[5] = 1.0 / gamma_plus
        cos = np.cos(r * pot_matrix[4])
        sin = np.sin(r * pot_matrix[4])
        exp = pot_matrix[0] * np.exp(-r * pot_matrix[5])
        U = (cos + pot_matrix[3] * sin) * exp / r
        fr = U / r  # derivative of 1/r
        fr += U * pot_matrix[5]  # derivative of exp
        fr += pot_matrix[4] * (sin - pot_matrix[3] * cos) * exp / r

    return U, fr
