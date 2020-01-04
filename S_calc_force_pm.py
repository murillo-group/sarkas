""" S_calc_force_pm.py

* Calculate force and potential based on PM algorithm.

"""

import numpy as np
import numba as nb
import pyfftw

@nb.njit
def AssignmentFunction(cao, x):
    """ Calculate the charge assignment functions from 
    Deserno et al. J Chem Phys 108, 7678 (1998).

    Parameters
    ----------
    cao : int
        charge assignment order

    x : float
        distance to the mid-point between mesh points if cao is even
        distance to closes mesh point if cao is even
    
    Returns
    ------
    W : array_like
        Charge Assignment Function

    """
    W = np.zeros( cao )

    if cao == 1:

        W[0] = 1

    elif cao==2:

        W[0] = 0.5*( 1. - 2.*x )
        W[1] = 0.5*( 1. + 2.*x )

    elif cao == 3:

        W[0] = ( 1. - 4.*x + 4.*x**2 )/8.
        W[1] = ( 3. - 4.*x**2 )/4.
        W[2] = ( 1. + 4.*x + 4.*x**2 )/8.

    elif cao == 4:

        W[0] = (1. - 6.*x + 12.*x**2 - 8.*x**3)/48.
        W[1] = (23. - 30.*x - 12.*x**2 + 24.*x**3)/48.
        W[2] = (23. + 30.*x - 12.*x**2 - 24.*x**3)/48.
        W[3] = (1. + 6.*x + 12.*x**2 + 8.*x**3)/48.

    elif cao == 5:

        W[0] = (1. - 8.*x + 24.*x**2 - 32.*x**3 + 16.*x**4)/384.
        W[1] = (19. - 44.*x + 24.*x**2 + 16.*x**3 - 16.*x**4)/96.
        W[2] = (115. - 120.*x**2 + 48.*x**4 )/192.
        W[3] = (19. + 44.*x + 24.*x**2 - 16.*x**3 - 16.*x**4)/96.
        W[4] = (1. + 8.*x + 24.*x**2 + 32.*x**3 + 16.*x**4)/384.

    elif cao == 6:
        W[0] = (1. - 10.*x + 40.*x**2 - 80.*x**3 + 80.*x**4 - 32.*x**5)/3840.
        W[1] = (237. - 750.*x + 840.*x**2 - 240.*x**3 - 240.*x**4 + 160.*x**5)/3840.
        W[2] = (841. - 770.*x - 440.*x**2 + 560.*x**3 + 80.*x**4 - 160.*x**5)/1920.
        W[3] = (841. + 770.*x - 440.*x**2 - 560.*x**3 + 80.*x**4 + 160.*x**5)/1920.
        W[4] = (237. + 750.*x + 840.*x**2 + 240.*x**3 - 240.*x**4 - 160.*x**5)/3840.
        W[5] = (1. + 10.*x + 40.*x**2 + 80.*x**3 + 80.*x**4 + 32.*x**5)/3840.
    elif cao == 7:

        W[0] = (1. - 12.*x + 60.*x*2 - 160.*x**3 + 240.*x**4 - 192.*x**5 + 64.*x**6)/46080.
        W[1] = (361. - 1416.*x + 2220.*x**2 - 1600.*x**3 + 240.*x**4 + 384.*x**5 - 192.*x**6)/23040.
        W[2] = (10543. - 17340.*x + 4740.*x**2 + 6880.*x**3 - 4080.*x**4 - 960.*x**5 + 960.*x**6)/46080.
        W[3] = (5887. - 4620.*x**2 + 1680.*x**4 - 320.*x**6)/11520.
        W[4] = (10543. + 17340.*x + 4740.*x**2 - 6880.*x**3 - 4080.*x**4 + 960.*x**5 + 960.*x**6)/46080.
        W[5] = (361. + 1416.*x + 2220.*x**2 + 1600.*x**3 + 240.*x**4 - 384.*x**5 - 192.*x**6)/23040.
        W[6] = (1. + 12.*x + 60.*x**2 + 160.*x**3 + 240.*x**4 + 192.*x**5 + 64.*x**6)/46080.
        
    return W


@nb.njit
def AssignChargesToMesh(pos,Z,N,cao,Mx,My,Mz,hx,hy,hz):
    """ Assign Charges to Mesh Points

    Parameters
    ----------
    pos : array
          particles' positions
    
    Z : array
        particles' charges
    
    N : int
        number of particles

    cao : int
          charge assignment order

    Mx : int
         number of mesh points along x-axis

    My : int
         number of mesh points along y-axis

    Mz : int
         number of mesh points along z-axis

    hx : int
         distance between mesh points along x-axis

    hy : int
         distance between mesh points along y-axis

    hz : int
         distance between mesh points along z-axis

    Returns
    -------

    rho_r : array
            charge density distributed on mesh

    """

    wx = np.zeros(cao)
    wy = np.zeros(cao)
    wz = np.zeros(cao)

    rho_r = np.zeros((Mz,My,Mz))

    for ipart in range(N):

        ix = int( pos[ipart,0]/hx ) 
        x = pos[ipart,0] - (ix + 0.5)*hx
        x = x/hx

        iy = int( pos[ipart,1]/hy )
        y = pos[ipart,1] - (iy + 0.5)*hy
        y = y/hy

        iz = int( pos[ipart,2]/hz ) 
        z = pos[ipart,2] - (iz + 0.5)*hz
        z = z/hz

        wx = assgn_func(cao,x)
        wy = assgn_func(cao,y)
        wz = assgn_func(cao,z)

        izn = iz # min. index along z-axis
    
        for g in range(cao):
    
            if izn < 0:
                r_g = izn + Mz
            elif izn > (Mz-1):
                r_g = izn - Mz
            else:
                r_g = izn

            iyn = iy # min. index along y-axis

            for i in range(cao):
    
                if iyn < 0:
                    r_i = iyn + My
                elif iyn > (My-1):
                    r_i = iyn - My
                else:
                    r_i = iyn
    
                ixn = ix # min. index along x-axis
    
                for j in range(cao):
    
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


@nb.njit
def Efield_fourier(phi_k,kx_v,ky_v,kz_v):
    """ Calculate the Electric field in Fourier space

    Parameters
    ----------
    phi_k : array
            Potential

    kx_v : array
           values of kx 

    ky_v : array
           values of ky 
    
    kz_v : array
           values of kz 
    
    Returns
    -------

    E_kx : array 
           Electric Field along kx-axis

    E_ky : array
           Electric Field along ky-axis

    E_kz : array
           Electric Field along kz-axis
    
    """

    E_kx = -1j*kx_v*phi_k
    E_ky = -1j*ky_v*phi_k
    E_kz = -1j*kz_v*phi_k
    
    return E_kx, E_ky, E_kz
    

@nb.njit
def AssignFieldToParticles(E_x_r,E_y_r,E_z_r,pos,Z,N,cao,Mass,Mx,My,Mz,hx,hy,hz):
    """ Assign Field to Particles
    
    Parameters
    ----------
    E_x_r : array
            Electric field along x-axis

    E_y_r : array
            Electric field along y-axis
    
    E_z_r : array
            Electric field along z-axis
    
    pos : array
          particles' positions
    
    Z : array
        particles' charges
    
    N : int
        number of particles

    cao : int
          charge assignment order
    
    Mass : array
           particles' masses
    Mx : int
         number of mesh points along x-axis

    My : int
         number of mesh points along y-axis

    Mz : int
         number of mesh points along z-axis

    hx : int
         distance between mesh points along x-axis

    hy : int
         distance between mesh points along y-axis

    hz : int
         distance between mesh points along z-axis

    Returns
    -------

    acc : array
          acceleration from Electric Field

    """
    E_x_p = np.zeros(N)
    E_y_p = np.zeros(N)
    E_z_p = np.zeros(N)
    
    acc = np.zeros((N,3))
    
    wx = np.zeros(cao)
    wy = np.zeros(cao)
    wz = np.zeros(cao)

    for ipart in range(N):

        ix = int( pos[ipart,0]/hx )
        x = pos[ipart,0] - (ix + 0.5)*hx
        x = x/hx

        iy = int( pos[ipart,1]/hy )
        y = pos[ipart,1] - (iy + 0.5)*hy
        y = y/hy

        iz = int( pos[ipart,2]/hz )
        z = pos[ipart,2] - (iz + 0.5)*hz
        z = z/hz

        wx = AssignmentFunction(cao,x)
        wy = AssignmentFunction(cao,y)
        wz = AssignmentFunction(cao,z)

        #print(w)

        izn = iz # min. index along z-axis
    
        for g in range(cao):
    
            if izn < 0:
                r_g = izn + Mz
            elif izn > (Mz-1):
                r_g = izn - Mz
            else:
                r_g = izn

            iyn = iy # min. index along y-axis

            for i in range(cao):
    
                if iyn < 0:
                    r_i = iyn + My
                elif iyn > (My-1):
                    r_i = iyn - My
                else:
                    r_i = iyn
    
                ixn = ix # min. index along x-axis
    
                for j in range(cao):
    
                    if ixn < 0:
                        r_j = ixn + Mx
                    elif ixn > (Mx-1):
                        r_j = ixn - Mx
                    else:
                        r_j = ixn
                    ZM = Z[ipart]/Mass[ipart]
                    E_x_p[ipart] = E_x_p[ipart] + ZM*E_x_r[r_g,r_i,r_j]*wz[g]*wy[i]*wx[j]
                    E_y_p[ipart] = E_y_p[ipart] + ZM*E_y_r[r_g,r_i,r_j]*wz[g]*wy[i]*wx[j]
                    E_z_p[ipart] = E_z_p[ipart] + ZM*E_z_r[r_g,r_i,r_j]*wz[g]*wy[i]*wx[j]
    
                    ixn += 1
        
                iyn += 1
        
            izn += 1

    acc[:,0] = E_x_p
    acc[:,1] = E_y_p
    acc[:,2] = E_z_p
            
    return acc
    
## FFTW version
def update(ptcls, params):
    """ Calculate particles' acceleration due to Electric Field

    Parameters
    ----------

    ptcls : class
            Particles' class. See S_Particles for more info

    params : class
             Simulation Parameters. See S_Params for more info

    Returns
    -------

    U_f : float
          Long range potential

    acc_f : array
            particles' accelerations due to the Electric field

    """

    pos = ptcls.pos
    N = pos.shape[0]

    Z = ptcls.charge
    Mass = ptcls.mass

    Lx = params.Lx
    Ly = params.Ly
    Lz = params.Lz

    Mx = params.P3M.Mx
    My = params.P3M.My
    Mz = params.P3M.Mz

    hx = params.P3M.hx
    hy = params.P3M.hy
    hz = params.P3M.hz
    
    V = Lx*Ly*Lz
    M_V = Mx*My*Mz
    
    G_k = params.P3M.G_k
    kx_v = params.P3M.kx_v
    ky_v = params.P3M.ky_v
    kz_v = params.P3M.kz_v

    rho_r = AssignChargesToMesh(pos, Z, N, params.P3M.cao, Mx, My, Mz, hx, hy, hz)

    fftw_n = pyfftw.builders.fftn(rho_r)
    rho_k_fft = fftw_n()
    rho_k = np.fft.fftshift(rho_k_fft)

    phi_k = G_k*rho_k
       
    rho_k_real = np.real(rho_k)
    rho_k_imag = np.imag(rho_k)
    rho_k_sq = rho_k_real*rho_k_real + rho_k_imag*rho_k_imag
    
    U_f = 0.5*np.sum(rho_k_sq*G_k)/V

    E_kx, E_ky, E_kz = Efield_fourier(phi_k, kx_v, ky_v, kz_v)
    
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
    
    acc_f = AssignFieldToParticles(E_x_r,E_y_r,E_z_r,pos,Z,N,params.P3M.cao,Mass,Mx,My,Mz,hx,hy,hz)
    
    return U_f, acc_f