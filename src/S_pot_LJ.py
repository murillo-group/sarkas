"""
Module for handling Lennard-Jones interaction
"""

import numpy as np
import numba as nb
import sys
import yaml

def setup(params, filename):
    """
    Updates ``params`` class with LJ potential paramters.

    Parameters
    ----------
    params : class
        Simulation's parameters. See ``S_params.py`` for more info.

    filename : string
        Input filename.

    """
    LJ_matrix = np.zeros((2, params.num_species, params.num_species))

    with open(filename, 'r') as stream:
        dics = yaml.load(stream, Loader=yaml.FullLoader)

        for lkey in dics:
            if (lkey == "Potential"):
                for keyword in dics[lkey]:
                    for key, value in keyword.items():
                        if (key == "epsilon"):
                            lj_m1 = np.array(value)

                        if (key == "sigma"):
                            lj_m2 = np.array(value)


    for j in range(params.num_species):
        for i in range(params.num_species):
            idx = i*params.num_species + j
            LJ_matrix[0, i, j] =  lj_m1[idx]
            LJ_matrix[1, i, j] =  lj_m2[idx]

    params.Potential.matrix = LJ_matrix
    params.force = LJ_force_PP

    # Calculate the (total) plasma frequency
    wp_tot_sq = 0.0
    sigma2 = 0.0
    epsilon_tot = 0.0
    for i in range(params.num_species):
        params.species[i].epsilon = lj_m1[i]
        params.species[j].sigma = lj_m2[i]

        wp2 = 48.0*params.species[i].epsilon/params.species[i].sigma**2
        params.species[i].wp = np.sqrt(wp2)
        sigma2 += params.species[i].sigma
        epsilon_tot += params.species[i].epsilon
        wp_tot_sq += wp2

    params.wp = np.sqrt(wp_tot_sq)

    params.PP_err = np.sqrt(np.pi*sigma2**(12)/(13.0*params.Potential.rc**(13) )  )
    params.PP_err *= np.sqrt(params.N/params.box_volume)*params.aws**2

    return


@nb.njit
def LJ_force_PP(r, pot_matrix_ij):
    """ 
    Calculates the PP force between particles using Lennard-Jones Potential.
    
    Parameters
    ----------
    r : float
        Particles' distance.

    pot_matrix : array
        LJ potential parameters. 


    Returns
    -------
    U : float
        Potential.

    force : float
        Force.
    """
    """
    Notes
    -----
    pot_matrix[0] = epsilon
    pot_matrix[1] = sigma
    """
    
    epsilon = pot_matrix_ij[0]
    sigma = pot_matrix_ij[1]
    s_over_r = sigma/r
    s_over_r_6 = s_over_r**6
    s_over_r_12 = s_over_r**12

    U = 4.0*epsilon*(s_over_r_12 - s_over_r_6)
    force = 48.0*epsilon*(s_over_r_12 - 0.5*s_over_r_6)/r**2

    return U, force
