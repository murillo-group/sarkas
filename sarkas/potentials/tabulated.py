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
    if not params.BC.open_axes:
        params.potential.LL_on = True  # linked list on
        if not hasattr(params.potential, "rc"):
            print("\nWARNING: The cut-off radius is not defined. L/2 = ", params.box_lengths.min() / 2, "will be used as rc")
            params.potential.rc = params.box_lengths.min() / 2.
            params.potential.LL_on = False  # linked list off

        if params.potential.method == "PP" and params.potential.rc > params.box_lengths.min() / 2.:
            print("\nWARNING: The cut-off radius is > L/2. L/2 = ", params.box_lengths.min() / 2, "will be used as rc")
            params.potential.rc = params.box_lengths.min() / 2.
            params.potential.LL_on = False  # linked list off

    twopi = 2.0 * np.pi
    beta_i = 1.0 / (params.kB * params.total_ion_temperature)

    # Calculate the Potential Matrix
    Z53 = 0.0
    Z_avg = 0.0

    for i in range(params.num_species):
        if hasattr(params.species[i], "Z"):
            Zi = params.species[i].Z
        else:
            Zi = 1.0

        Z53 += Zi ** (5. / 3.) * params.species[i].concentration
        Z_avg += Zi * params.species[i].concentration

        for j in range(params.num_species):
            if hasattr(params.species[j], "Z"):
                Zj = params.species[j].Z
            else:
                Zj = 1.0

    # Effective Coupling Parameter in case of multi-species
    # see eq.(3) in Ref.[Haxhimali2014]_
    params.potential.Gamma_eff = Z53 * Z_avg ** (1. / 3.) * params.qe ** 2 * beta_i / (params.fourpie0 * params.aws)
    params.QFactor /= params.fourpie0

    params.potential.matrix = np.array([params.potential.tab_interp_ord])

    # Calculate the (total) plasma frequency
    wp_tot_sq = 0.0
    for i, sp in enumerate(params.species):
        wp2 = 4.0 * np.pi * sp.charge ** 2 * sp.number_density / (sp.mass * params.fourpie0)
        sp.wp = np.sqrt(wp2)
        wp_tot_sq += wp2

    params.total_plasma_frequency = np.sqrt(wp_tot_sq)

    indx, r, u_r, f_r = np.loadtxt(params.potential.tabulated_file, skiprows=7, unpack=True)

    if params.units == 'cgs':
        r *= 1e-8
        u_r *= params.eV2J * params.J2erg
        f_r *= 1e-8 / 1e-12 ** 2
    else:
        r *= 1e-10
        u_r *= params.eV2J
        f_r *= 1e-10 / 1e-12 ** 2

    # Interpolate


    if params.potential.method == "PP":
        params.force = Yukawa_force_PP
        # Force error calculated from eq.(43) in Ref.[1]_
        params.potential.F_err = np.sqrt(twopi / params.lambda_TF) * np.exp(-params.potential.rc / params.lambda_TF)
        # Renormalize
        params.potential.F_err *= params.aws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)
