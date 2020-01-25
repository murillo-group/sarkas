"""
S_pot_LJ.py

Module for handling Lennard-Jones interaction
"""

import numpy as np
import numba as nb
import sys
import yaml

def LJ_setup(params, filename):
    """
    Setup simulation's parameters for Moliere-like potential

    Parameters
    ----------
    params : class
            Simulation's parameters. See S_params.py for more info.

    filename : string
                Input filename

    Returns
    -------
    none

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
    return


@nb.njit
def LJ_force_PP(r, pot_matrix_ij):
    """ 
    Calculate the PP force between particles using Lennard-Jones Potential.
    
    Parameters
    ----------
    r : float
        particles' distance

    pot_matrix : array
                Moliere potential parameters according to Wilson et al. PRB 15, 2458 (1977) 


    Returns
    -------
    U : float
          potential

    force : float
         force
    
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
