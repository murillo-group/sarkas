'''
S_force.py

Calculate a short range force based on a given potential.
'''
# Python modules
import numpy as np
import numba as nb
import math as mt
import sys

# MD modules
import S_global_names as glb


@nb.jit
def Yukawa_force_PP(r, U_s_r):
    kappa = glb.kappa/glb.ai
    U_s_r = U_s_r + np.exp(-kappa*r)/r 
    f1 = 1./r**2*np.exp(-kappa*r)
    f2 = kappa/r*np.exp(-kappa*r)
    fr = f1+f2

    return U_s_r, fr

@nb.jit
def Yukawa_force_P3M(U_s_r, r):
    #Gautham's thesis Eq. 3.22
    kappa = glb.kappa/glb.ai
    G = glb.G
    U_s_r = U_s_r + (0.5/r)*(np.exp(kappa*r)*mt.erfc(G*r + 0.5*kappa/G) + np.exp(-kappa*r)*mt.erfc(G*r - 0.5*kappa/G))
    f1 = (0.5/r**2)*np.exp(kappa*r)*mt.erfc(G*r + 0.5*kappa/G)*(1-kappa*r)
    f2 = (0.5/r**2)*np.exp(-kappa*r)*mt.erfc(G*r - 0.5*kappa/G)*(1+kappa*r)
    f3 = (G/np.sqrt(np.pi)/r)*(np.exp(-(G*r + 0.5*kappa/G)**2)*np.exp(kappa*r) + np.exp(-(G*r - 0.5*kappa/G)**2)*np.exp(-kappa*r))
    fr = f1+f2+f3

    return U_s_r, fr

@nb.jit
def Coulomb_force(U, r):
    U = U - (1/r)
    fr = 1/(r**2)

    return U


@nb.jit
def EGS_force(U_s_r, r):
    pass
