"""
Module for handling Moliere Potential
"""
import numpy as np
import numba as nb
import sys
import yaml


def setup(params, read_input=True):
    """
    Updates ``params`` class with Moliere's parameters as given by Ref. [Wilson1977]_ .

    Parameters
    ----------
    read_input: bool
        Flag to read inputs from YAML input file.

    params : Params class
        Simulation's parameters.

    References
    ----------
    .. [Wilson1977] `W.D. Wilson et al., Phys Rev B 15, 2458 (1977) <https://doi.org/10.1103/PhysRevB.15.2458>`_
    """
    if read_input:
        with open(params.input_file, 'r') as stream:
            dics = yaml.load(stream, Loader=yaml.FullLoader)

            for lkey in dics:
                if lkey == "Potential":
                    for keyword in dics[lkey]:
                        for key, value in keyword.items():
                            if key == "C":
                                params.potential.C_params = np.array(value)

                            if key == "rc":  # cutoff
                                params.potential.rc = float(value)

                            if key == "b":
                                params.potential.b_params = np.array(value)

    update_params(params)


def update_params(params):
    """
    Create potential dependent simulation's parameters as given by Ref. [Wilson1977]_ .

    Parameters
    ----------
    params : Params class
        Simulation's parameters.

    """
    twopi = 2.0 * np.pi
    beta_i = 1.0 / (params.kB * params.Ti)
    if not params.BC.open_axes:
        params.potential.LL_on = True  # linked list on
        if not hasattr(params.potential, "rc"):
            print("\nWARNING: The cut-off radius is not defined. L/2 = ", params.Lv.min() / 2, "will be used as rc")
            params.potential.rc = params.Lv.min() / 2.
            params.potential.LL_on = False  # linked list off

        if params.potential.method == "PP" and params.potential.rc > params.Lv.min() / 2.:
            print("\nWARNING: The cut-off radius is > L/2. L/2 = ", params.Lv.min() / 2, "will be used as rc")
            params.potential.rc = params.Lv.min() / 2.
            params.potential.LL_on = False  # linked list off

    params_len = int(2 * len(params.potential.C_params))

    Moliere_matrix = np.zeros((params_len + 1, params.num_species, params.num_species))

    Z53 = 0.0
    Z_avg = 0.0
    for i, sp1 in enumerate(params.species):
        if hasattr(sp1, "Z"):
            Zi = sp1.Z
        else:
            Zi = 1.0

        Z53 += Zi ** (5. / 3.) * sp1.concentration
        Z_avg += Zi * sp1.concentration

        for j, sp2 in enumerate(params.species):
            if hasattr(sp2, "Z"):
                Zj = sp2.Z
            else:
                Zj = 1.0

            Moliere_matrix[0:len(params.potential.C_params), i, j] = params.potential.C_params
            Moliere_matrix[len(params.potential.C_params):params_len, i, j] = params.potential.b_params
            Moliere_matrix[params_len, i, j] = (Zi * Zj) * params.qe * params.qe / params.fourpie0

    # Effective Coupling Parameter in case of multi-species
    # see eq.(3) in Haxhimali et al. Phys Rev E 90 023104 (2014)
    params.potential.Gamma_eff = Z53 * Z_avg ** (1. / 3.) * params.qe ** 2 * beta_i / (params.fourpie0 * params.aws)
    params.QFactor = params.QFactor / params.fourpie0
    params.potential.matrix = Moliere_matrix

    # Calculate the (total) plasma frequency
    wp_tot_sq = 0.0
    for i, sp in enumerate(params.species):
        wp2 = 4.0 * np.pi * sp.charge ** 2 * sp.number_density / (sp.mass * params.fourpie0)
        sp.wp = np.sqrt(wp2)
        wp_tot_sq += wp2

    params.wp = np.sqrt(wp_tot_sq)

    if params.potential.method == "PP":
        params.force = Moliere_force_PP

        # Force error calculated from eq.(43) in Ref.[1]_
        params.PP_err = np.sqrt(twopi / params.potential.b_params.min()) \
                        * np.exp(-params.potential.rc / params.potential.b_params.min())
        # Renormalize
        params.PP_err = params.PP_err * params.aws ** 2 * np.sqrt(params.total_num_ptcls / params.box_volume)


@nb.njit
def Moliere_force_PP(r, pot_matrix):
    """ 
    Calculates the PP force between particles using the Moliere Potential.
    
    Parameters
    ----------
    r : float
        Particles' distance.

    pot_matrix : array
        Moliere potential parameters. 


    Returns
    -------
    phi : float
        Potential

    fr : float
        Force
    """
    """
    Notes
    -----
    See Wilson et al. PRB 15 2458 (1977) for parameters' details
    pot_matrix[0] = C_1
    pot_matrix[1] = C_2
    pot_matrix[2] = C_3
    pot_matrix[3] = b_1
    pot_matrix[4] = b_2
    pot_matrix[5] = b_3
<<<<<<< HEAD
    pot_matrix[6] = Z_1Z_2e^2/ (4 np.pi eps_0)
=======
    pot_matrix[6] = Z_1Z_2e^2/(4 np.pi eps_0)
>>>>>>> master
    """

    U = 0.0
    force = 0.0

    for i in range(int(len(pot_matrix[:-1]) / 2)):
        factor1 = r * pot_matrix[i + 3]
        factor2 = pot_matrix[i] / r
        U += factor2 * np.exp(-factor1)
        force += np.exp(-factor1) * factor2 * (1.0 / r + pot_matrix[i]) / r

    force = force * pot_matrix[-1]
    U = U * pot_matrix[-1]

    return U, force
