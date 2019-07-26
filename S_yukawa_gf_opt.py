'''
S_yukawa_gf_opt.sy

Optimized Green's Function
'''

import numpy as np
import numba as nb
import S_global_names as glb

@nb.jit
def gf_opt():
    kappa = glb.kappa
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
                            
                    # Gautham's Thesis, eq. 3.31                                                   
                    G_k[nz,ny,nx] = U_G_k/((U_k_sq**2)*k_sq)
                    
                    A_pm = A_pm + U_G_k**2/((U_k_sq**2)*k_sq)
                       
    return G_k, kx_v, ky_v, kz_v, A_pm
