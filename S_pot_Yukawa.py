""" S_Yukawa.py

Optimized green's Function, potential and force calculation for Yukawa potential.
"""

import numpy as np
import numba as nb
import sys

# MD modules
import S_global_names as glb


@nb.njit
def Yukawa_force_PP(r, pot_matrix_ij):
    """ Calculates the Yukawa Force between two particles when 
        the PP algorithm is chosen

    Parameters
    ----------
    r : real
        distance between two particles

    pot_matrix_ij : array_like
                    it contains potential dependent variables
                    pot_matrix_ij[0] = 1/lambda_TF (if Yukawa units, it is kappa)
                    pot_matrix_ij[1] = Gamma
                    pot_matrix_ij[2] = potential factor

    Returns
    -------
    U : real
        Potential value
                
    force : real
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

@nb.jit
def Yukawa_force_P3M(U_s_r, r):
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
    U_s_r : real
            Potential value
                
    fr : real
         Force between two particles calculated using eq.(22) in 
         Dharuman et al. J Chem Phys 146, 024112 (2017)
    
    """
    # Scale the screening parameter by the WS radius
    kappa = glb.kappa/glb.ai
    G = glb.G   # Ewald parameter alpha 
    U_s_r = U_s_r + (0.5/r)*(np.exp(kappa*r)*mt.erfc(G*r + 0.5*kappa/G) + np.exp(-kappa*r)*mt.erfc(G*r - 0.5*kappa/G))
    f1 = (0.5/r**2)*np.exp(kappa*r)*mt.erfc(G*r + 0.5*kappa/G)*(1-kappa*r)
    f2 = (0.5/r**2)*np.exp(-kappa*r)*mt.erfc(G*r - 0.5*kappa/G)*(1+kappa*r)
    f3 = (G/np.sqrt(np.pi)/r)*(np.exp(-(G*r + 0.5*kappa/G)**2)*np.exp(kappa*r) + np.exp(-(G*r - 0.5*kappa/G)**2)*np.exp(-kappa*r))
    fr = f1+f2+f3

    return U_s_r, fr


#Optimized Green's Function
@nb.jit
def gf_opt(params):
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

    A_pm : real
           Second term in eq.(28) in Stern et al. J Chem Phys 128, 214006 (2008)
           representing the mean-square error for reciprocal space differentiation
           A_pm notation comes from Dharuman et al. J Chem Phys 146, 024112 (2017)
           which btw has a mistake since the first G(k) should be squared

    """
    kappa = params.kappa
    Gew = glb.G_ew
    p = glb.p
    mx_max = glb.mx_max
    my_max = glb.my_max
    mz_max = glb.mz_max
    Mx = glb.Mx
    My = glb.My
    Mz = glb.Mz
    hx = glb.hx
    hy = glb.hy
    hz = glb.hz
    Lx = glb.Lx
    Ly = glb.Ly
    Lz = glb.Lz
   
    kappa_sq = kappa**2
    Gew_sq = Gew**2    
    
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
    
    kx_v = 2*np.pi*(nx_v - nx_mid)/Lx
    ky_v = 2*np.pi*(ny_v - ny_mid)/Ly
    kz_v = 2*np.pi*(nz_v - nz_mid)/Lz
    
    A_pm = 0.0
    
    for nz in range(Mz):
        nz_sh = nz-nz_mid
        kz = 2*np.pi*nz_sh/Lz
        
        for ny in range(My):
            ny_sh = ny-ny_mid
            ky = 2*np.pi*ny_sh/Ly
            
            for nx in range(Mx):
                nx_sh = nx-nx_mid
                kx = 2*np.pi*nx_sh/Lx
                           
                k_sq = kx**2 + ky**2 + kz**2
                
                if k_sq != 0.0:
                
                    U_k_sq = 0.0
                    U_G_k = 0.0

                    # Sum over the aliases
                    for mz in range(-mz_max,mz_max+1):
                        for my in range(-my_max,my_max+1):
                            for mx in range(-mx_max,mx_max+1):
                                
                                #if ((nx_sh != 0) or (mx != 0)) and ((ny_sh != 0) or (my != 0)) and ((nz_sh != 0) or (mz != 0)):
                                
                                kx_M = 2*np.pi*(nx_sh + mx*Mx)/Lx
                                ky_M = 2*np.pi*(ny_sh + my*My)/Ly
                                kz_M = 2*np.pi*(nz_sh + mz*Mz)/Lz
                            
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
                                U_k_M_sq = U_k_M**2
                            
                                G_k_M = 4*np.pi*np.exp(-0.25*(kappa_sq + k_M_sq)/Gew_sq) / (kappa_sq + k_M_sq)
                            
                                k_dot_k_M = kx*kx_M + ky*ky_M + kz*kz_M
                            
                                U_G_k += (U_k_M_sq * G_k_M * k_dot_k_M)
                                U_k_sq += U_k_M_sq
                            
                    # eq.(31) of Dharuman et al. J Chem Phys 146, 024112 (2017)                                                 
                    G_k[nz,ny,nx] = U_G_k/((U_k_sq**2)*k_sq)
                    
                    # eq.(28) of Stern et al. J Chem Phys 128, 214006 (2008)
                    # eq.(32) of Dharuman et al. J Chem Phys 146, 024112 (2017)
                    A_pm = A_pm + U_G_k**2/((U_k_sq**2)*k_sq)
                       
    return G_k, kx_v, ky_v, kz_v, A_pm
