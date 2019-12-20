import numpy as np
import numba as nb
import sys
import fdint

import S_global_names as glb
import S_constants as const

@nb.njit
def EGS_force_PP(r,pot_matrix):
    """ Calculate the PP force between particles using the EGS Potential
    
    Parameters
    ----------
    r : float
        particles' distance

    pot_matrix : array
                 EGS potential parameters. See details below and 
                 Stanton et al. PRE 91, 033104 (2015) for more info

    Return
    ------

    phi : float
          potential

    fr : float
         force

    """
    nu = pot_matrix[1]
    if (nu <= 1):
        #pot_matrix[2] = Charge factor
        #pot_matrix[3] = 1 + alpha
        #pot_matrix[4] = 1 - alpha
        #pot_matrix[5] = lambda_minus
        #pot_matrix[6] = lambda_plus

        temp1 = pot_matrix[3]*np.exp(-r/pot_matrix[5])
        temp2 = pot_matrix[4]*np.exp(-r/pot_matrix[6])
        phi = (temp1 + temp2)*pot_matrix[2]/r
        fr = phi/r + pot_matrix[2]*(temp1/pot_matrix[5] + temp2/pot_matrix[6]) 
    else: 
        #pot_matrix[2] = Charge factor
        #pot_matrix[3] = 1.0
        #pot_matrix[4] = alpha prime
        #pot_matrix[5] = gamma_minus
        #pot_matrix[6] = gamma_plus
        cos = np.cos(r/pot_matrix[5])
        sin = np.sin(r/pot_matrix[5])
        exp = pot_matrix[2]*np.exp(-r/pot_matrix[6])
        phi = (pot_matrix[3]*cos + pot_matrix[4]*sin)*exp/r
        fr1 = phi/r   # derivative of 1/r
        fr3 = phi/pot_matrix[6] # derivative of exp
        fr2 = exp/(r*pot_matrix[5])*(sin - pot_matrix[4]*cos)
        fr = fr1 + fr2 + fr3

    return phi, fr


def EGS_force_P3M(pos,acc_s_r):
    pass


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
