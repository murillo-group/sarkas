"""
Module for handling Yukawa interaction
"""
import numpy as np
from numba import njit
import yaml  # IO


@njit
def B(x, k, i, t):
    if k == 0:
        a = 1.0 if t[i] <= x < t[i+1] else 0.0
    else:
        if t[i+k] == t[i]:
            c1 = 0.0
        else:
            c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
        if t[i+k+1] == t[i+1]:
            c2 = 0.0
        else:
            c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)

        a = c1 + c2
    return a


@njit
def bspline(x, t, c, k):
   n = len(t) - k - 1
   assert (n >= k+1) and (len(c) >= n)
   bs = np.sum(c[i] * B(x, k, i, t) for i in range(n))
   return bs


@njit
def tab_force_PP(r, pot_matrix):
    """
    Calculates Potential and Force between two particles.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : array
        It contains potential dependent variables.

    Returns
    -------
    U : float
        Potential.

    force : float
        Force between two particles.

    """
    U = pot_matrix[0] * r
    force = U * (1 / r + pot_matrix[1]) / r

    return U, force


def setup(params, read_input=True):
    """
    Updates ``params`` class with Yukawa's parameters.

    Parameters
    ----------
    read_input: bool
        Flag to read inputs from YAML input file.

    params: object
        Simulation's parameters.

    """

    """
    Dev Notes
    ---------
    Yukawa_matrix[0,i,j] : qi qj/(4pi esp0) Force factor between two particles.
    Yukawa_matrix[1,:,:] : kappa = 1.0/lambda_TF or given as input. Same value for all species.
    Yukawa_matrix[2,i,j] : Ewald parameter in the case of P3M Algorithm. Same value for all species
    """

    # open the input file to read Yukawa parameters
    if read_input:
        with open(params.input_file, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)
            for lkey in dics:
                if lkey == "Potential":
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if key == "tabulated_file":  # screening
                                params.potential.tabulated_file = value

                            if key == "rc":  # cutoff
                                params.potential.rc = float(value)

                            if key == "interpolation_order":
                                params.potential.tab_interp_kind = value

    update_params(params)


def update_params(params):
    """
    Create potential dependent simulation's parameters.

    Parameters
    ----------
    params: object
        Simulation's parameters.

    References
    ----------
    .. [Stanton2015] `Stanton and Murillo Phys Rev E 91 033104 (2015) <https://doi.org/10.1103/PhysRevE.91.033104>`_
    .. [Haxhimali2014] `T. Haxhimali et al. Phys Rev E 90 023104 (2014) <https://doi.org/10.1103/PhysRevE.90.023104>`_
    """

    params.potential.matrix = np.array([params.potential.tab_interp_ord,num])

    # Calculate the (total) plasma freque
    # Interpolate


    if params.potential.method == "PP":
        params.force = Yukawa_force_PP
        # Force error calculated from eq.(43) in Ref.[1]_
        params.potential.F_err = np.sqrt(twopi / params.lambda_TF) * np.exp(-params.potential.rc / params.lambda_TF)
        # Renormalize
        params.potential.F_err *= params.a_ws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)
