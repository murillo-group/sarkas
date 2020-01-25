import numpy as np
import numba as nb
import sys
import fdint
import scipy.constants as const
import yaml

def EGS_setup(params, filename):
    """
    Setup simulation's parameters for Yukawa interaction

    Parameters
    ----------
    params : class
            Simulation's parameters. See S_params.py for more info.

    filename : string
                Input filename

    Returns
    -------
    none

    Notes
    -----
    Reference Stanton and Murillo Phys Rev E 91 033104 (2015)

    EGS_matrix[0,i,j] : kappa = 1.0/lambda_TF or given as input. Same value for all species.
    EGS_matrix[1,i,j] : nu = eq.(14) of Ref
    EGS_matrix[2,i,j] : qi qj/(4 pi esp0) Force factor between two particles.
    EGS_matrix[3:6,i,j] : other parameters see below.
    EGS_matrix[7,i,j] : Ewald parameter in the case of P3M Algorithm. Same value for all species
    """
    # constants and conversion factors    
    if (params.Control.units == "cgs"):
        fourpie0 = 1.0
    else:
        fourpie0 = 4.0*np.pi*params.epsilon_0
    
    twopi = 2.0*np.pi
    beta_i = 1.0/(params.kB*params.Ti)
        
    # open the input file to read Yukawa parameters
    with open(filename, 'r') as stream:
        dics = yaml.load(stream, Loader=yaml.FullLoader)
        for lkey in dics:
            if (lkey == "Potential"):
                for keyword in dics[lkey]:
                    for key, value in keyword.items():
                        if (key == "kappa"): # screening
                            params.Potential.kappa = float(value)

                        if (key == "Gamma"): # coupling
                            params.Potential.Gamma = float(value)

                        # electron temperature for screening parameter calculation
                        if (key == "elec_temperature"):
                            params.Te = float(value)

                        if (key == "elec_temperature_eV"):
                            T_eV = float(value)
                            params.Te = params.eV2K*float(value)

                        # lambda factor : 1 = von Weizsaecker, 1/9 = Thomas-Fermi
                        if (key == "lambda_correction"):
                            params.Potential.lmbda = float(value)

    if not hasattr(params, "Te"):
        print("Electron temperature is not defined. 1st species temperature ", params.species[0].temperature, \
                    "will be used as the electron temperature.")
        params.Te = params.species[0].temperature

    if not hasattr( params.Potential, "lmbda"):
        print("\nlambda correction factor is not defined. The Thomas-Fermi value will be used lambda = 1/9 ")
        params.Potential.lmbda = 1.0/9.0
        lmbda = 1.0/9.0 #lambda parameter (1 or 1/9 in egs paper)

    # Use MKS units to calculate EGS parameters.
    kb = const.Boltzmann
    e = const.elementary_charge
    h_bar = const.hbar
    m_e = const.elec_mass
    e_0 = const.epsilon_0

    Te  = params.Te
    Ti = params.Ti

    if (params.Control.units == "cgs"):
        ne = params.ne*1.e6    # /cm^3 --> /m^3
        ni = params.total_num_density*1.e6

    if (params.Control.units == "mks"):
        ne = params.ne    # /cm^3 --> /m^3
        ni = params.total_num_density

    ai = (3./(4*np.pi*ni))**(1./3.)

    # Fermi Integrals
    fdint_fdk_vec = np.vectorize(fdint.fdk)
    fdint_dfdk_vec = np.vectorize(fdint.dfdk)
    fdint_ifd1h_vec = np.vectorize(fdint.ifd1h)

    beta = 1.0/(kb*Te)
    # eq.(4) from Stanton et al. PRE 91 033104 (2015)
    eta = fdint_ifd1h_vec(np.pi**2*(beta*h_bar**2/(m_e))**(3/2)/np.sqrt(2)*ne)
    # eq.(10) from Stanton et al. PRE 91 033104 (2015)
    lambda_TF = np.sqrt((4.0*np.pi**2*e_0*h_bar**2)/(m_e*e**2)*np.sqrt(2*beta*h_bar**2/m_e)/(4.0*fdint_fdk_vec(k=-0.5, phi=eta))) 
    # eq. (14) from Stanton et al. PRE 91 033104 (2015)
    nu = (m_e*e**2)/(4.0*np.pi*e_0*h_bar**2)*np.sqrt(8.0*beta*h_bar**2/m_e)/(3.0*np.pi)*lmbda*fdint_dfdk_vec(k=-0.5, phi=eta)
    # Fermi Energy
    E_F = (hbar**2/(2.0*m_e))*(3.0*np.pi**2*ne)**(2./3.)
    # Degeneracy Parameter
    theta = 1.0/(beta*E_F)
    # eq. (33) from Stanton et al. PRE 91 033104 (2015)
    Ntheta = 1.0 + 2.8343*theta**2 - 0.2151*theta**3 + 5.2759*theta**4
    # eq. (34) from Stanton et al. PRE 91 033104 (2015)        
    Dtheta = 1.0 + 3.9431*theta**2 + 7.9138*theta**4
    # eq. (32) from Stanton et al. PRE 91 033104 (2015)
    h = Ntheta/Dtheta*np.tanh(1.0/theta)
    # grad h(x)
    gradh = ( -(Ntheta/Dtheta)/np.cosh(1/theta)**2/(theta**2)  # derivative of tanh(1/x) 
            - np.tanh(1.0/theta)*( Ntheta*(7.8862*theta + 31.6552*theta**3)/Dtheta**2 # derivative of 1/Dtheta
            + (5.6686*theta - 0.6453*theta**2 + 21.1036*theta**3)/Dtheta )     )       # derivative of Ntheta

    #eq.(31) from Stanton et al. PRE 91 033104 (2015)
    b = 1.0 -1.0/8.0*beta*theta*(h -2.0*theta*(gradh))*(hbar/lambda_TF)**2/(m_e)
    
    if (params.Control.units == "cgs"):
        lambda_TF = 100.0*lamdbda_TF

    params.lambda_TF = lambda_TF
    # Monotonic decay
    if (nu <= 1):
        #eq. (29) from Stanton et al. PRE 91 033104 (2015)
        params.Potential.lambda_p = lambda_TF*np.sqrt(nu/(2.0*b + 2.0*np.sqrt(b**2-nu)))
        params.Potential.lambda_m = lambda_TF*np.sqrt(nu/(2.0*b - 2.0*np.sqrt(b**2-nu)))
        params.Potential.alpha = b/np.sqrt(b-nu)
    
    # Oscillatory behavior
    if (nu > 1):
        #eq. (29) from Stanton et al. PRE 91 033104 (2015)
        params.Potential.gamma_m = lambda_TF*np.sqrt(nu/(np.sqrt(nu) - b))
        params.Potential.gamma_p = lambda_TF*np.sqrt(nu/(np.sqrt(nu) + b))
        params.Potential.alphap = b/np.sqrt(nu-b)
    
    params.Potential.nu = nu
   
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
            wp2 = params.species[i].charge**2*params.species[i].num_density/(params.species[i].mass*params.eps0)
            params.species[i].wp = np.sqrt(wp2)
            wp_tot_sq += wp2

        params.wp = np.sqrt(wp_tot_sq)

    if (params.P3M.on):
        EGS_matrix = np.zeros( (7, params.num_species, params.num_species) )
    else:
        EGS_matrix = np.zeros((8, params.num_species, params.num_species)) 
    
    EGS_matrix[0, :, :] = 1.0/params.lambda_TF
    EGS_matrix[1, :, :] = params.Potential.nu 

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

            if (nu <= 1):
                EGS_matrix[2, i, j] = (Zi*Zj)*params.qe*params.qe/(2.0*fourpie0)
                EGS_matrix[3, i, j] = (1.0 + params.Potential.alpha)
                EGS_matrix[4, i, j] = (1.0 - params.Potential.alpha)
                EGS_matrix[5, i, j] = params.Potential.lambda_m
                EGS_matrix[6, i, j] = params.Potential.lambda_p

            if (nu > 1):
                EGS_matrix[2, i, j] = (Zi*Zj)*params.qe*params.qe/(fourpie0)
                EGS_matrix[3, i, j] = 1.0
                EGS_matrix[4, i, j] = params.Potential.alphap
                EGS_matrix[5, i, j] = params.Potential.gamma_m
                EGS_matrix[6, i, j] = params.Potential.gamma_p
    
    # Effective Coupling Parameter in case of multi-species
    # see eq.(3) in Haxhimali et al. Phys Rev E 90 023104 (2014)
    params.Potential.Gamma_eff = Z53*Z_avg**(1./3.)*params.qe**2*beta_i/(fourpie0*params.aws)
    params.QFactor = params.QFactor/fourpie0
        
    params.Potential.matrix = EGS_matrix

    if (params.Potential.method == "PP"):
        params.force = EGS_force_PP

    if (params.Potential.method == "P3M"):
        print("\nP3M Algorithm not implemented yet. Good Bye!")
        sys.exit()
        params.force = EGS_force_P3M
        params.P3M.hx = params.Lx/params.P3M.Mx
        params.P3M.hy = params.Ly/params.P3M.My
        params.P3M.hz = params.Lz/params.P3M.Mz
        params.Potential.matrix[7,:,:] = params.P3M.G_ew
        # Optimized Green's Function
        #params.P3M.G_k, params.P3M.kx_v, params.P3M.ky_v, params.P3M.kz_v, params.P3M.PM_err, params.P3M.PP_err = gf_opt(params.P3M.MGrid,\
        #    params.P3M.aliases, params.Lv, params.P3M.cao, params.N, params.Potential.matrix, params.Potential.rc, fourpie0)

        # Include the charges in the Force errors. Prefactor in eq.(29) of Dharuman et al J Chem Phys 146 024112 (2017)
        # Notice that the equation was derived for a single component plasma. 
        #params.P3M.PM_err *= params.QFactor*fourpie0/np.sqrt(params.N) # the multiplication of fourpie0 is needed to avoid double division.
        #params.P3M.PP_err *= params.QFactor*fourpie0/np.sqrt(params.N)
        # Total Force Error 
        #params.P3M.F_err = np.sqrt(params.P3M.PM_err**2 + params.P3M.PP_err**2)
    return


@nb.njit
def EGS_force_PP(r,pot_matrix):
    """ 
    Calculate Potential and force between particles using the EGS Potential.
    
    Parameters
    ----------
    r : float
        particles' distance

    pot_matrix : array
                EGS potential parameters. See details below and Stanton et al. PRE 91, 033104 (2015) for more info

    Return
    ------

    U : float
        potential

    fr : float
        force

    """
    nu = pot_matrix[1]
    if (nu <= 1.0):
        #pot_matrix[2] = Charge factor
        #pot_matrix[3] = 1 + alpha
        #pot_matrix[4] = 1 - alpha
        #pot_matrix[5] = lambda_minus
        #pot_matrix[6] = lambda_plus

        temp1 = pot_matrix[3]*np.exp(-r/pot_matrix[5])
        temp2 = pot_matrix[4]*np.exp(-r/pot_matrix[6])
        U = (temp1 + temp2)*pot_matrix[2]/r
        fr = U/r + pot_matrix[2]*(temp1/pot_matrix[5] + temp2/pot_matrix[6]) 
    else: 
        #pot_matrix[2] = Charge factor
        #pot_matrix[3] = 1.0
        #pot_matrix[4] = alpha prime
        #pot_matrix[5] = gamma_minus
        #pot_matrix[6] = gamma_plus
        cos = np.cos(r/pot_matrix[5])
        sin = np.sin(r/pot_matrix[5])
        exp = pot_matrix[2]*np.exp(-r/pot_matrix[6])
        U = (pot_matrix[3]*cos + pot_matrix[4]*sin)*exp/r
        fr1 = U/r   # derivative of 1/r
        fr3 = U/pot_matrix[6] # derivative of exp
        fr2 = exp/(r*pot_matrix[5])*(sin - pot_matrix[4]*cos)
        fr = fr1 + fr2 + fr3

    return U, fr

@nb.njit
def EGS_force_P3M(pos,acc_s_r):
    pass


@nb.njit
def gf_opt(MGrid, aliases, BoxLv, p, N, pot_matrix,rcut, fourpie0):
    pass
