"""
Module for handling Moliere Potential as given by Ref. [1]_

References
----------
.. [1] `W.D. Wilson et al., Phys Rev B 15, 2458 (1977) <https://doi.org/10.1103/PhysRevB.15.2458>`_
"""
import numpy as np
import numba as nb
import sys
import yaml

def Moliere_setup(params, filename):
    """
    Updates ``params`` class with Moliere potential paramters.

    Parameters
    ----------
    params : class
        Simulation's parameters. See ``S_params.py`` for more info.

    filename : string
        Input filename.

    """

    # constants and conversion factors    
    if (params.Control.units == "cgs"):
        fourpie0 = 1.0
    elif (params.Control.units == "mks"):
        fourpie0 = 4.0*np.pi*params.epsilon_0
    
    twopi = 2.0*np.pi
    beta_i = 1.0/(params.kB*params.Ti)

    with open(filename, 'r') as stream:
        dics = yaml.load(stream, Loader=yaml.FullLoader)

        for lkey in dics:
            if (lkey == "Potential"):
                for keyword in dics[lkey]:
                    for key, value in keyword.items():
                        if (key == "C"):
                            C_params = np.array(value)
                            
                        if (key == "b"):
                            b_params = np.array(value)

    # Calculate the (total) plasma frequency
    if (params.Control.units == "cgs"):
        wp_tot_sq = 0.0
        for i in range(params.num_species):
            wp2 = 4.0*np.pi*params.species[i].charge**2*params.species[i].num_density/params.species[i].mass
            params.species[i].wp = np.sqrt(wp2)
            wp_tot_sq += wp2

        params.wp = np.sqrt(wp_tot_sq)

    elif (params.Control.units == "mks"):
        wp_tot_sq = 0.0
        for i in range(params.num_species):
            wp2 = params.species[i].charge**2*params.species[i].num_density/(params.species[i].mass*params.epsilon_0)
            params.species[i].wp = np.sqrt(wp2)
            wp_tot_sq += wp2

        params.wp = np.sqrt(wp_tot_sq)

    if (params.P3M.on):
        Moliere_matrix = np.zeros( (7, params.num_species, params.num_species) )
    else:
        Moliere_matrix = np.zeros( (8, params.num_species, params.num_species) ) 
    
    Z53 = 0.0
    Z_avg = 0.0
    for i in range(params.num_species):
        if hasattr (params.species[i], "Z"):
            Zi = params.species[i].Z
        else:
            Zi = 1.0

        Z53 += (Zi)**(5./3.)*params.species[i].concentration
        Z_avg += Zi*params.species[i].concentration

        for j in range(params.num_species):
            if hasattr (params.species[j],"Z"):
                Zj = params.species[j].Z
            else:
                Zj = 1.0
        
            Moliere_matrix[0:3, i, j] =  C_params
            Moliere_matrix[3:6, i, j] =  b_params
            Moliere_matrix[6, i, j] = (Zi*Zj)*params.qe*params.qe/fourpie0
    
    # Effective Coupling Parameter in case of multi-species
    # see eq.(3) in Haxhimali et al. Phys Rev E 90 023104 (2014)
    params.Potential.Gamma_eff = Z53*Z_avg**(1./3.)*params.qe**2*beta_i/(fourpie0*params.aws)
    params.QFactor = params.QFactor/fourpie0
    params.Potential.matrix = Moliere_matrix

    if (params.Potential.method == "PP"):
        params.force = Moliere_force_PP

    if (params.Potential.method == "P3M"):
        print("\nP3M Algorithm not implemented yet. Good Bye!")
        sys.exit()

    return
    
@nb.njit
def Moliere_force_PP(r,pot_matrix):
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
    """

    U = 0.0
    force = 0.0

    for i in range(3):

        factor1 = r*pot_matrix[i + 3]
        factor2 = pot_matrix[i]/r
        U += factor2*np.exp(-factor1)
        force += np.exp(-factor1)*(factor2)*(1.0/r + pot_matrix[i])

    force = force*pot_matrix[6]
    U = U*pot_matrix[6]
    
    return U, force
    
