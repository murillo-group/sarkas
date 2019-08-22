import numpy as np
import numba as nb
import sys
import fdint

import S_global_names as glb
import S_constants as const

@nb.jit
def init():
    l = 1 #lambda parameter (1 or 1/9 in egs paper)
    k = const.kb
    e = -const.elementary_charge
    hbar = const.hbar
    m_e = const.elec_mass
    e_0 = const.eps_0
    T  = glb.Te
    Zi = glb.Zi
    n = Zi*glb.ni

    fdint_fdk_vec = np.vectorize(fdint.fdk)
    fdint_dfdk_vec = np.vectorize(fdint.dfdk)
    fdint_ifd1h_vec = np.vectorize(fdint.ifd1h)


    beta = 1/(k*T)
    eta = fdint_ifd1h_vec(np.pi**2*(beta*hbar**2/(m_e))**(3/2)/np.sqrt(2)*n) #eq 4 inverted
    L_TF = np.sqrt((4*np.pi**2*e_0*hbar**2)/(m_e*e**2)*np.sqrt(2*beta*hbar**2/m_e)/(4*fdint_fdk_vec(k=-0.5, phi=eta))) #eq 10
    nu = (m_e*e**2)/(4*np.pi*e_0*hbar**2)*np.sqrt(8*beta*hbar**2/m_e)/(3*np.pi)*l*fdint_dfdk_vec(k=-0.5, phi=eta) #eq 14
    E_F = (hbar**2/(2*m_e))*(3*np.pi**2*n)**(2/3)
    t = k*T/E_F
    N = 1 + 2.8343*t**2 - 0.2151*t**3 + 5.2759*t**4
    dd = 1 + 3.9431*t**2 + 7.9138*t**4
    h = N/dd*np.tanh(1/t)
    gradh = -((1 + 2.8343*t**2 - 0.2151*t**3 + 5.2759*t**4)*np.cosh(1/t)**(-2))/(t**2*(1 + 3.9431*t**2 + 7.9138*t**4))
    - ((7.8862*t + 31.6552*t**3)*(1 + 2.8343*t**2 - 0.2151*t**3 + 5.2759*t**4)*np.tanh(1/t))/(1 + 3.9431*t**2 + 7.9138*t**4)**2
    + ((5.6686*t - 0.6453*t**2 + 21.1036*t**3)*np.tanh(1/t))/(1 + 3.9431*t**2 + 7.9138*t**4)
    b = 1-1/8*beta*t*(h-2*t*(gradh))*(hbar**2*L_TF**(-2))/(m_e)
    
    if(nu <= 1):
        glb.lambda_p = L_TF*np.sqrt(nu/(2*b+2*np.sqrt(b**2-nu)))
        glb.lambda_m = L_TF*np.sqrt(nu/(2*b-2*np.sqrt(b**2-nu)))
        glb.alpha = b/np.sqrt(b-nu)
        glb.nu = nu

#        return zbar(Z, m, T, n)*e**2/(2*4*np.pi*e_0)*((1+alpha)*np.exp(-r/lambda_m)+(1-alpha)*np.exp(-r/lambda_p))

    if(nu > 1):
        glb.gamma_m = L_TF*np.sqrt(nu/(np.sqrt(nu)-b))
        glb.gamma_p = L_TF*np.sqrt(nu/(np.sqrt(nu)+b))
        glb.alphap = b/np.sqrt(nu-b)
        glb.nu = nu
#        return zbar(Z, m, T, n)*e**2/(4*np.pi*e_0)*(np.cos(r/gamma_m) + alpha*np.sin(r/gamma_m))*np.exp(-r/gamma_p)


def update(pos,acc_s_r):
    pass
