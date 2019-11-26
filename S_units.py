'''
S_units.py

setting units and physical constants
'''
import numpy as np
import sys

import S_constants as const  # empty. 


# read input data from yukawa_MD_p3m.in
def setup(params):
    units = params.Control.units
    if(units == "Yukawa"):
        Yukawa_units()

    elif(units == "cgs"):
        cgs_units()

    elif(units == "mks"):
        mks_units()

    else:
        print("No such units are available")
        sys.exit()

    return

def cgs_units():

    const.elementary_charge = 4.80320425e-10    # statcoulomb
    const.elec_mass = 9.10938356e-28
    const.proton_mass = 1.672621898e-24
    const.kb = 1.38064852e-16
    const.epsilon_0 = 1.
    const.hbar = 1.0545718e-27

    return

def Yukawa_units():

    const.elementary_charge = 1
    const.elec_mass = 1
    const.proton_mass = 1
    const.kb = 1
    const.epsilon_0 = 1
    const.hbar= 1

    return
def mks_units():

    const.elementary_charge = 1.602176634e-19   # coulomb
    const.elec_mass = 9.10938356e-31
    const.proton_mass = 1.672621898e-27
    const.kb = 1.38064852e-23
    const.epsilon_0 = 8.8541878178176e-12
    const.hbar= 1.0545718e-34

    return
