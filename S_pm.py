import numpy as np
import numba as nb
import pyfftw
import S_global_names as glb

@nb.jit
def charge_assgn_6_numba(rho_r,pos,Z,N,Mx,My,Mz,hx,hy,hz):
    
    n = 6
    wx = np.zeros(n)
    wy = np.zeros(n)
    wz = np.zeros(n)

    rho_r.fill(0.0)

    for ipart in range(N):

        ix = int(np.floor(pos[ipart,0]/hx))
        x = pos[ipart,0] - (ix + 0.5)*hx
        x = x/hx

        iy = int(np.floor(pos[ipart,1]/hy))
        y = pos[ipart,1] - (iy + 0.5)*hy
        y = y/hy

        iz = int(np.floor(pos[ipart,2]/hz))
        z = pos[ipart,2] - (iz + 0.5)*hz
        z = z/hz

        wx[0] = (1 - 10*x + 40*x**2 - 80*x**3 + 80*x**4 - 32*x**5)/3840
        wx[1] = (237 - 750*x + 840*x**2 - 240*x**3 - 240*x**4 + 160*x**5)/3840
        wx[2] = (841 - 770*x - 440*x**2 + 560*x**3 + 80*x**4 - 160*x**5)/1920
        wx[3] = (841 + 770*x - 440*x**2 - 560*x**3 + 80*x**4 + 160*x**5)/1920
        wx[4] = (237 + 750*x + 840*x**2 + 240*x**3 - 240*x**4 - 160*x**5)/3840
        wx[5] = (1 + 10*x + 40*x**2 + 80*x**3 + 80*x**4 + 32*x**5)/3840
      
        wy[0] = (1 - 10*y + 40*y**2 - 80*y**3 + 80*y**4 - 32*y**5)/3840
        wy[1] = (237 - 750*y + 840*y**2 - 240*y**3 - 240*y**4 + 160*y**5)/3840
        wy[2] = (841 - 770*y - 440*y**2 + 560*y**3 + 80*y**4 - 160*y**5)/1920
        wy[3] = (841 + 770*y - 440*y**2 - 560*y**3 + 80*y**4 + 160*y**5)/1920
        wy[4] = (237 + 750*y + 840*y**2 + 240*y**3 - 240*y**4 - 160*y**5)/3840
        wy[5] = (1 + 10*y + 40*y**2 + 80*y**3 + 80*y**4 + 32*y**5)/3840
        
        wz[0] = (1 - 10*z + 40*z**2 - 80*z**3 + 80*z**4 - 32*z**5)/3840
        wz[1] = (237 - 750*z + 840*z**2 - 240*z**3 - 240*z**4 + 160*z**5)/3840
        wz[2] = (841 - 770*z - 440*z**2 + 560*z**3 + 80*z**4 - 160*z**5)/1920
        wz[3] = (841 + 770*z - 440*z**2 - 560*z**3 + 80*z**4 + 160*z**5)/1920
        wz[4] = (237 + 750*z + 840*z**2 + 240*z**3 - 240*z**4 - 160*z**5)/3840
        wz[5] = (1 + 10*z + 40*z**2 + 80*z**3 + 80*z**4 + 32*z**5)/3840

        #print(w)

        izn = iz # min. index along z-axis
    
        for g in range(n):
    
            if izn < 0:
                r_g = izn + Mz
            elif izn > (Mz-1):
                r_g = izn - Mz
            else:
                r_g = izn

            iyn = iy # min. index along y-axis

            for i in range(n):
    
                if iyn < 0:
                    r_i = iyn + My
                elif iyn > (My-1):
                    r_i = iyn - My
                else:
                    r_i = iyn
    
                ixn = ix # min. index along x-axis
    
                for j in range(n):
    
                    if ixn < 0:
                        r_j = ixn + Mx
                    elif ixn > (Mx-1):
                        r_j = ixn - Mx
                    else:
                        r_j = ixn
        
                    #print([r_g,r_i,r_j], [wz[g],wy[i],wx[j]])
    
                    rho_r[r_g,r_i,r_j] = rho_r[r_g,r_i,r_j] + Z[ipart]*wz[g]*wy[i]*wx[j]
    
                    ixn += 1
        
                iyn += 1
        
            izn += 1

    return rho_r


@nb.jit
def Efield_fourier(phi_k,kx_v,ky_v,kz_v):
    
    E_kx = -1j*kx_v*phi_k
    E_ky = -1j*ky_v*phi_k
    E_kz = -1j*kz_v*phi_k
    
    return E_kx, E_ky, E_kz
    

@nb.jit
def field_assgn_6_numba(acc,E_x_p,E_y_p,E_z_p,E_x_r,E_y_r,E_z_r,pos,Z,N,Mx,My,Mz,hx,hy,hz):
    
    E_x_p.fill(0.)
    E_y_p.fill(0.)
    E_z_p.fill(0.)
    
    acc.fill(0.)
    
    n = 6
    wx = np.zeros(n)
    wy = np.zeros(n)
    wz = np.zeros(n)

    for ipart in range(N):

        ix = int(np.floor(pos[ipart,0]/hx))
        x = pos[ipart,0] - (ix + 0.5)*hx
        x = x/hx

        iy = int(np.floor(pos[ipart,1]/hy))
        y = pos[ipart,1] - (iy + 0.5)*hy
        y = y/hy

        iz = int(np.floor(pos[ipart,2]/hz))
        z = pos[ipart,2] - (iz + 0.5)*hz
        z = z/hz

        wx[0] = (1 - 10*x + 40*x**2 - 80*x**3 + 80*x**4 - 32*x**5)/3840
        wx[1] = (237 - 750*x + 840*x**2 - 240*x**3 - 240*x**4 + 160*x**5)/3840
        wx[2] = (841 - 770*x - 440*x**2 + 560*x**3 + 80*x**4 - 160*x**5)/1920
        wx[3] = (841 + 770*x - 440*x**2 - 560*x**3 + 80*x**4 + 160*x**5)/1920
        wx[4] = (237 + 750*x + 840*x**2 + 240*x**3 - 240*x**4 - 160*x**5)/3840
        wx[5] = (1 + 10*x + 40*x**2 + 80*x**3 + 80*x**4 + 32*x**5)/3840
      
        wy[0] = (1 - 10*y + 40*y**2 - 80*y**3 + 80*y**4 - 32*y**5)/3840
        wy[1] = (237 - 750*y + 840*y**2 - 240*y**3 - 240*y**4 + 160*y**5)/3840
        wy[2] = (841 - 770*y - 440*y**2 + 560*y**3 + 80*y**4 - 160*y**5)/1920
        wy[3] = (841 + 770*y - 440*y**2 - 560*y**3 + 80*y**4 + 160*y**5)/1920
        wy[4] = (237 + 750*y + 840*y**2 + 240*y**3 - 240*y**4 - 160*y**5)/3840
        wy[5] = (1 + 10*y + 40*y**2 + 80*y**3 + 80*y**4 + 32*y**5)/3840
        
        wz[0] = (1 - 10*z + 40*z**2 - 80*z**3 + 80*z**4 - 32*z**5)/3840
        wz[1] = (237 - 750*z + 840*z**2 - 240*z**3 - 240*z**4 + 160*z**5)/3840
        wz[2] = (841 - 770*z - 440*z**2 + 560*z**3 + 80*z**4 - 160*z**5)/1920
        wz[3] = (841 + 770*z - 440*z**2 - 560*z**3 + 80*z**4 + 160*z**5)/1920
        wz[4] = (237 + 750*z + 840*z**2 + 240*z**3 - 240*z**4 - 160*z**5)/3840
        wz[5] = (1 + 10*z + 40*z**2 + 80*z**3 + 80*z**4 + 32*z**5)/3840

        #print(w)

        izn = iz # min. index along z-axis
    
        for g in range(n):
    
            if izn < 0:
                r_g = izn + Mz
            elif izn > (Mz-1):
                r_g = izn - Mz
            else:
                r_g = izn

            iyn = iy # min. index along y-axis

            for i in range(n):
    
                if iyn < 0:
                    r_i = iyn + My
                elif iyn > (My-1):
                    r_i = iyn - My
                else:
                    r_i = iyn
    
                ixn = ix # min. index along x-axis
    
                for j in range(n):
    
                    if ixn < 0:
                        r_j = ixn + Mx
                    elif ixn > (Mx-1):
                        r_j = ixn - Mx
                    else:
                        r_j = ixn
        
                    #print([r_g,r_i,r_j], [wz[g],wy[i],wx[j]])
    
                    E_x_p[ipart] = E_x_p[ipart] + E_x_r[r_g,r_i,r_j]*wz[g]*wy[i]*wx[j]
                    E_y_p[ipart] = E_y_p[ipart] + E_y_r[r_g,r_i,r_j]*wz[g]*wy[i]*wx[j]
                    E_z_p[ipart] = E_z_p[ipart] + E_z_r[r_g,r_i,r_j]*wz[g]*wy[i]*wx[j]
    
                    ixn += 1
        
                iyn += 1
        
            izn += 1

    acc[:,0] = Z*E_x_p
    acc[:,1] = Z*E_y_p
    acc[:,2] = Z*E_z_p
            
    return acc
    
## FFTW version
def particle_mesh_fft_r(pos, Z, G_k, kx_v, ky_v, kz_v, rho_r, acc_f, E_x_p, E_y_p, E_z_p):
    N = glb.N

    Lx = glb.Lx
    Ly = glb.Ly
    Lz = glb.Lz

    Mx = glb.Mx
    My = glb.My
    Mz = glb.Mz

    hx = glb.hx
    hy = glb.hy
    hz = glb.hz
    
    V = Lx*Ly*Lz
    M_V = Mx*My*Mz
    
    rho_r = charge_assgn_6_numba(rho_r, pos, Z, N, Mx, My, Mz, hx, hy, hz)
    
    fftw_n = pyfftw.builders.fftn(rho_r)
    rho_k_fft = fftw_n()
    rho_k = np.fft.fftshift(rho_k_fft)

    phi_k = G_k*rho_k
       
    rho_k_real = np.real(rho_k)
    rho_k_imag = np.imag(rho_k)
    rho_k_sq = rho_k_real*rho_k_real + rho_k_imag*rho_k_imag
    
    U_f = 0.5*np.sum(rho_k_sq*G_k)/V
    
    E_kx = -1j*kx_v*phi_k
    E_ky = -1j*ky_v*phi_k
    E_kz = -1j*kz_v*phi_k
    
    E_kx_unsh = np.fft.ifftshift(E_kx)
    E_ky_unsh = np.fft.ifftshift(E_ky)
    E_kz_unsh = np.fft.ifftshift(E_kz)
    
    ifftw_n = pyfftw.builders.ifftn(E_kx_unsh)
    E_x = ifftw_n()
    ifftw_n = pyfftw.builders.ifftn(E_ky_unsh)
    E_y = ifftw_n()
    ifftw_n = pyfftw.builders.ifftn(E_kz_unsh)
    E_z = ifftw_n()
    
    E_x = M_V*E_x/V
    E_y = M_V*E_y/V
    E_z = M_V*E_z/V
    
    E_x_r = np.real(E_x)
    E_y_r = np.real(E_y)
    E_z_r = np.real(E_z)
    
    acc_f = field_assgn_6_numba(acc_f,E_x_p,E_y_p,E_z_p,E_x_r,E_y_r,E_z_r,pos,Z,N,Mx,My,Mz,hx,hy,hz)
    
    return U_f, acc_f
    
## FFTW version with modified charge-assigment and field-assignment routines (not more efficient than the routine above)
#def particle_mesh_fftw_r_wpart(kappa,G_ew,pos,Z,N,G_k,kx_v,ky_v,kz_v,Mx,My,Mz,hx,hy,hz,Lx,Ly,Lz,p,mx_max,my_max,mz_max,rho_r,acc_f,E_x_p,E_y_p,E_z_p):
#    
#    V = Lx*Ly*Lz
#    M_V = Mx*My*Mz
#    
#    rho_r, wx_part, wy_part, wz_part = charge_assgn_6_numba_wpart(rho_r,pos,Z,N,Mx,My,Mz,hx,hy,hz)
#    
#    fftw_n = pyfftw.builders.fftn(rho_r)
#    rho_k_fft = fftw_n()
#    rho_k = np.fft.fftshift(rho_k_fft)
#
#    phi_k = G_k*rho_k
#       
#    rho_k_real = np.real(rho_k)
#    rho_k_imag = np.imag(rho_k)
#    rho_k_sq = rho_k_real*rho_k_real + rho_k_imag*rho_k_imag
#    
#    U_f = 0.5*np.sum(rho_k_sq*G_k)/V
#    
#    E_kx = -1j*kx_v*phi_k
#    E_ky = -1j*ky_v*phi_k
#    E_kz = -1j*kz_v*phi_k
#    
#    E_kx_unsh = np.fft.ifftshift(E_kx)
#    E_ky_unsh = np.fft.ifftshift(E_ky)
#    E_kz_unsh = np.fft.ifftshift(E_kz)
#    
#    ifftw_n = pyfftw.builders.ifftn(E_kx_unsh)
#    E_x = ifftw_n()
#    ifftw_n = pyfftw.builders.ifftn(E_ky_unsh)
#    E_y = ifftw_n()
#    ifftw_n = pyfftw.builders.ifftn(E_kz_unsh)
#    E_z = ifftw_n()
#    
#    E_x = M_V*E_x/V
#    E_y = M_V*E_y/V
#    E_z = M_V*E_z/V
#    
#    E_x_r = np.real(E_x)
#    E_y_r = np.real(E_y)
#    E_z_r = np.real(E_z)
#    
#    acc_f = field_assgn_6_numba_wpart(acc_f,wx_part,wy_part,wz_part,E_x_p,E_y_p,E_z_p,E_x_r,E_y_r,E_z_r,pos,Z,N,Mx,My,Mz,hx,hy,hz)
#    
#    return U_f, acc_f
