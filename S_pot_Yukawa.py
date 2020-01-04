""" S_Yukawa.py

Optimized green's Function, potential and force calculation for Yukawa potential.
"""

import numpy as np
import numba as nb
import math as mt
import sys
import scipy.constants as const

@nb.njit
def Yukawa_force_PP(r, pot_matrix_ij):
    """ Calculates the Yukawa Force between two particles when 
        the PP algorithm is chosen

    Parameters
    ----------
    r : float
        distance between two particles

    pot_matrix_ij : array_like
                    it contains potential dependent variables
                    pot_matrix_ij[0,:,:] = 1/lambda_TF
                    pot_matrix_ij[1,:,:] = Gamma_ij
                    pot_matrix_ij[2,:,:] = q1*q2/(4*pi*eps0)

    Returns
    -------
    U : float
        Potential
                
    force : float
            Force between two particles
    
    Notes
    -----    
    Author:
    Date Created: 12/1/19
    Date Updated: 
    Updates: 
    """
    
    factor1 = r*pot_matrix_ij[0]
    factor2 = pot_matrix_ij[2]/r
    U = np.exp(-factor1)*(factor2)
    force = U*(1/r + pot_matrix_ij[0])

    return U, force

@nb.njit
def Yukawa_force_P3M(r, pot_matrix_ij):
    """ Calculates the Yukawa Force between two particles when 
        the P3M algorithm is chosen

    Parameters
    ----------
    U_s_r : real
            short range (PP) part of the potential

    r : real
        distance between two particles

    Returns
    -------
    U_s_r : float
            Potential value
                
    fr : float
         Force between two particles calculated using eq.(22) in 
         Dharuman et al. J Chem Phys 146, 024112 (2017)
    
    """
    kappa = pot_matrix_ij[0]

    G = pot_matrix_ij[3]   # Ewald parameter alpha 

    U_s_r = pot_matrix_ij[2]*(0.5/r)*(np.exp(kappa*r)*mt.erfc(G*r + 0.5*kappa/G) + np.exp(-kappa*r)*mt.erfc(G*r - 0.5*kappa/G))
    f1 = (0.5/r**2)*np.exp(kappa*r)*mt.erfc(G*r + 0.5*kappa/G)*(1.0 - kappa*r)
    f2 = (0.5/r**2)*np.exp(-kappa*r)*mt.erfc(G*r - 0.5*kappa/G)*(1.0 + kappa*r)
    f3 = (G/np.sqrt(np.pi)/r)*(np.exp(-(G*r + 0.5*kappa/G)**2)*np.exp(kappa*r) + np.exp(-(G*r - 0.5*kappa/G)**2)*np.exp(-kappa*r))
    fr = pot_matrix_ij[2]*( f1 + f2 + f3 )

    return U_s_r, fr


@nb.njit
def gf_opt(MGrid, aliases, BoxLv, p, N, pot_matrix,rcut):
    """ Calculates the Optimized Green Function given by eq.(22) in
        Stern et al. J Chem Phys 128, 214006 (2008)

    Parameters
    ----------
    params : class

    Returns
    -------

    G_k : array_like
          optimal Green Function

    kx_v : array_like
           array of reciprocal space vectors along the x-axis

    ky_v : array_like
           array of reciprocal space vectors along the y-axis

    kz_v : array_like
           array of reciprocal space vectors along the z-axis

    PM_err : float
             Error in the force calculation due to the optimized Green's function
             eq.(28) in Stern et al. J Chem Phys 128 214106 (2008)

    PP_err : float
             Error in the force calculation due to the distance cutoff.
             eq.(30) in Dharuman et al. J Chem Phys 146 024112 (2017)

    DeltaF_tot : float
                 Total force error. eq.(42) from Dharuman et al. J Chem Phys 146 024112 (2017)
    
    """
    kappa = pot_matrix[0,0,0] #params.Potential.matrix[0,0,0]
    Gew = pot_matrix[3,0,0] #params.Potential.matrix[3,0,0]
    #p = params.P3M.cao
    rcut2 = rcut*rcut
    mx_max = aliases[0] #params.P3M.mx_max
    my_max = aliases[1] # params.P3M.my_max
    mz_max = aliases[2] #params.P3M.mz_max
    Mx = MGrid[0] #params.P3M.Mx
    My = MGrid[1] #params.P3M.My
    Mz = MGrid[2] #params.P3M.Mz
    #hx = params.P3M.hx
    #hy = params.P3M.hy
    #hz = params.P3M.hz
    Lx = BoxLv[0] #params.Lx
    Ly = BoxLv[1] #params.Ly
    Lz = BoxLv[2] #params.Lz
    hx = Lx/float(Mx)
    hy = Ly/float(My)
    hz = Lz/float(Mz)

    kappa_sq = kappa*kappa
    Gew_sq = Gew*Gew
    
    epsilon = 1.0/(4.0*np.pi*const.epsilon_0)

    G_k = np.zeros((Mz,My,Mx))
    
    if np.mod(Mz,2) == 0:
        nz_mid = Mz/2
    else:
        nz_mid = (Mz-1)/2
    
    if np.mod(My,2) == 0:
        ny_mid = My/2
    else:
        ny_mid = (My-1)/2
    
    if np.mod(Mx,2) == 0:
        nx_mid = Mx/2
    else:
        nx_mid = (Mx-1)/2
        
    nx_v = np.arange(Mx).reshape((1,Mx))
    ny_v = np.arange(My).reshape((My,1))
    nz_v = np.arange(Mz).reshape((Mz,1,1))
    
    kx_v = 2.0*np.pi*(nx_v - nx_mid)/Lx
    ky_v = 2.0*np.pi*(ny_v - ny_mid)/Ly
    kz_v = 2.0*np.pi*(nz_v - nz_mid)/Lz
    
    A_pm = 0.0
    
    for nz in range(Mz):
        nz_sh = nz-nz_mid
        kz = 2.0*np.pi*nz_sh/Lz
        
        for ny in range(My):
            ny_sh = ny-ny_mid
            ky = 2.0*np.pi*ny_sh/Ly
            
            for nx in range(Mx):
                nx_sh = nx-nx_mid
                kx = 2.0*np.pi*nx_sh/Lx
                           
                k_sq = kx*kx + ky*ky + kz*kz
                
                if k_sq != 0.0:
                
                    U_k_sq = 0.0
                    U_G_k = 0.0

                    # Sum over the aliases
                    for mz in range(-mz_max,mz_max+1):
                        for my in range(-my_max,my_max+1):
                            for mx in range(-mx_max,mx_max+1):
                                
                                #if ((nx_sh != 0) or (mx != 0)) and ((ny_sh != 0) or (my != 0)) and ((nz_sh != 0) or (mz != 0)):
                                
                                kx_M = 2.0*np.pi*(nx_sh + mx*Mx)/Lx
                                ky_M = 2.0*np.pi*(ny_sh + my*My)/Ly
                                kz_M = 2.0*np.pi*(nz_sh + mz*Mz)/Lz
                            
                                k_M_sq = kx_M**2 + ky_M**2 + kz_M**2
                                
                                if kx_M != 0.0:
                                    U_kx_M = np.sin(0.5*kx_M*hx)/(0.5*kx_M*hx)
                                else:
                                    U_kx_M = 1.0

                                if ky_M != 0.0:
                                    U_ky_M = np.sin(0.5*ky_M*hy)/(0.5*ky_M*hy)
                                else: 
                                    U_ky_M = 1.0
                                    
                                if kz_M != 0.0:
                                    U_kz_M = np.sin(0.5*kz_M*hz)/(0.5*kz_M*hz)
                                else:
                                    U_kz_M = 1.0
                                
                                U_k_M = (U_kx_M*U_ky_M*U_kz_M)**p
                                U_k_M_sq = U_k_M*U_k_M
                                
                                G_k_M = epsilon*np.exp(-0.25*(kappa_sq + k_M_sq)/Gew_sq)/(kappa_sq + k_M_sq)
                                
                                k_dot_k_M = kx*kx_M + ky*ky_M + kz*kz_M

                                #print( (-0.25*(kappa_sq + k_M_sq)/Gew_sq))

                                U_G_k += (U_k_M_sq * G_k_M * k_dot_k_M)
                                U_k_sq += U_k_M_sq
                                
                    # eq.(31) of Dharuman et al. J Chem Phys 146, 024112 (2017)    
                                                 
                    G_k[nz,ny,nx] = U_G_k/((U_k_sq**2)*k_sq)
                    Gk_hat = epsilon*np.exp(-0.25*(kappa_sq + k_sq)/Gew_sq) / (kappa_sq + k_sq)       

                    # eq.(28) of Stern et al. J Chem Phys 128, 214006 (2008)
                    # eq.(32) of Dharuman et al. J Chem Phys 146, 024112 (2017)
                    A_pm = A_pm + Gk_hat*Gk_hat*k_sq - U_G_k**2/((U_k_sq**2)*k_sq)

    PM_err = A_pm
    PP_err = 2.0*np.exp(-0.25*kappa_sq/Gew_sq)*np.exp(-Gew_sq*rcut2)/np.sqrt(rcut)
    

    DeltaF_tot = np.sqrt(PM_err**2 + PP_err**2)*np.sqrt(N/(Lx*Ly*Lz))*pot_matrix[2,0,0]/epsilon

    return G_k, kx_v, ky_v, kz_v, PM_err, PP_err, DeltaF_tot
