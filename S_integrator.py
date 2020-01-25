"""
S_integrator.py

Module of various types of integrators 

Verlet : velocity Verlet
Verlet_with_Langevin : Verlet with Langevin damping
RK45: N/A
RK45_with_Langevin: N/A
"""

import numpy as np
import numba as nb
import sys
import S_calc_force_pp as force_pp
import S_calc_force_pm as force_pm

class Integrator:
    def __init__(self, params):

        self.calc_force = PotentialAcceleration

        self.params = params

        self.dt = params.Control.dt
        self.half_dt = 0.5*self.dt
        self.N = params.N
        self.d = params.d
        self.Lv = params.Lv
        self.PBC = params.Control.PBC
        self.Lmax_v = params.Lmax_v
        self.Lmin_v = params.Lmin_v
        self.N_species = self.params.num_species
        self.T = self.params.Ti

        if (params.Integrator.type == "Verlet"):
            if(params.Langevin.on):
                if(params.Langevin.type == "BBK"):
                    self.update = self.Verlet_with_Langevin     # currently only BBK.
                else:
                    print("No such Langevin type.")
                    sys.exit()

                self.g = self.params.Langevin.gamma
                self.c1 = (1. - 0.5*self.g*self.dt)
                self.c2 = 1./(1. + 0.5*self.g*self.dt)
                self.sqrt_dt = np.sqrt(self.dt)

            else:
                self.update = Verlet
        elif (params.Integrator.type == "Magnetic_Verlet"):
            self.update = Magnetic_Verlet
            print("Only Verlet integrator is supported. Check your input file, integrator part.")
        else:
            print("Only Verlet integrator is supported. Check your input file, integrator part 2.")
        
    def RK(self, ptcls):
        """ 
        Update particle position and velocity based on the 4th order Runge-Kutta method
        More information can be found here: 
        https://en.wikipedia.org/wiki/Runge–Kutta_methods
        or on the Sarkas website. 
    
        Parameters
        ----------
        ptlcs: particles data. See S_particles.py for the detailed information
        k1: the vel, acc at the beginng
        k2: the vel, acc at the middle
        k3: the vel, acc at the middle if the acc. at the beginning was k2
        k4: the vel, acc at the end if the acc. at the beginning was k3

        Returns
        -------
        U : float
            Total potential energy
        """
        # Import global parameters (is there a better way to do this?)
        dt = self.params.dt
        half_dt = 0.5*dt
        N = self.params.N
        d = self.params.d
        Lv = self.params.Lv

        pass

    def RK45(self, ptcls):
        """ 
        Update particle position and velocity based on Explicit Runge-Kutta method of order 5(4). 
        More information can be found here: 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.RK45.html
        https://en.wikipedia.org/wiki/Runge–Kutta_methods
        or on the Sarkas website. 
    
        Parameters
        ----------
        ptlcs: class
               Particles data. See S_particles.py for the detailed information

        Returns
        -------
        U : float
            Total potential energy
        """

        # Import global parameters (is there a better way to do this?)
        # Yes use self.params or just pass params
        dt = self.params.dt
        N = self.params.N
        d = self.params.d
        Lv = self.params.Lv
        PBC = self.params.PBC
        Lmax_v = self.params.Lmax_v
        Lmin_v = self.params.Lmin_v
        pass

    def RK45_with_Langevin(self, ptcls):
        pass


def Verlet(ptcls,params):
    """ 
    Update particle position and velocity based on velocity verlet method.
    More information can be found here: https://en.wikipedia.org/wiki/Verlet_integration
    or on the Sarkas website. 

    Parameters
    ----------
    ptlcs: class
           Particles data. See S_particles.py for more info.
    
    params : class
            Simulation's parameters. See S_params.py for more info.

    Returns
    -------
    U : float
        Total potential energy
    
    """
    
    # First half step velocity update
    ptcls.vel = ptcls.vel + 0.5*ptcls.acc*params.Control.dt

    # Full step position update
    ptcls.pos = ptcls.pos + ptcls.vel*params.Control.dt

    # Periodic boundary condition
    if params.Control.PBC == 1:
        EnforcePBC(ptcls.pos,params.Lv)
        
    # Compute total potential energy and accleration for second half step velocity update                 
    U = PotentialAcceleration(ptcls,params)
    
    #Second half step velocity update
    ptcls.vel = ptcls.vel + 0.5*ptcls.acc*params.Control.dt

    return U

def Magnetic_Verlet(ptcls,params):
    """
    Update particles' positions and velocities based on velocity verlet method in the case of a 
    constant magnetic field. B = (0, 0, B0).
    More information on the Velocity Verlet Method can be found here: https://en.wikipedia.org/wiki/Verlet_integration
    or on the Sarkas website. 
    See Spreiter & Walter Journal of Computational Physics 152, 102–119 (1999) for more information on the Magnetic implementation

    Parameters
    ----------
    ptlcs: class
           Particles data. See S_particles.py for more info.
    
    params : class
            Simulation's parameters. See S_params.py for more info.
            
    Returns
    -------
    U : float
        Total potential energy
    """

    # Time step
    dt = params.dt
    dt_sq = dt*dt
    half_dt = 0.5*dt

    sp_start = 0 # start index for species loop
    sp_end = 0 # end index for species loop

    # array to temporary store velocities
    v_temp = np.zeros( (params.N, params.d) )

    for ic in range( params.num_species ):
        # Cyclotron frequency
        omega_c = params.species[ic].omega_c

        inv_omc = 1.0/(omega_c)
        inv_omc_sq = inv_omc*inv_omc
        omc_dt = omega_c*dt
        # See eqs.(31)-(32) in Spreiter & Walter Journal of Computational Physics 152, 102–119 (1999)
        sdt = np.sin( omc_dt )
        cdt = np.cos( omc_dt )
        ccodt = cdt - 1.0
        ssodt = sdt - omc_dt

        sp_end = sp_start + params.species[ic].num
        # First half step of Verlet position update
        # eq.(28)-(30) Spreiter & Walter Journal of Computational Physics 152, 102–119 (1999) 
        ptcls.pos[sp_start:sp_end,0] = ptcls.pos[sp_start:sp_end,0] \
            + inv_omc*( ptcls.vel[sp_start:sp_end,0]*sdt - ptcls.vel[sp_start:sp_end,1]*ccodt) \
            - inv_omc_sq*( ptcls.acc[sp_start:sp_end,0]*ccodt + ptcls.acc[sp_start:sp_end,1]*ssodt )
        
        ptcls.pos[sp_start:sp_end,1] = ptcls.pos[sp_start:sp_end,1] \
            + inv_omc*( ptcls.vel[sp_start:sp_end,1]*sdt + ptcls.vel[sp_start:sp_end,0]*ccodt) \
            - inv_omc_sq*( ptcls.acc[sp_start:sp_end,1]*ccodt - ptcls.acc[sp_start:sp_end,0]*ssodt )

        ptcls.pos[sp_start:sp_end,2] = ptcls.pos[sp_start:sp_end,2] + ptcls.vel[sp_start:sp_end,2]*dt \
            + 0.5*ptcls.acc[sp_start:sp_end,2]*dt_sq

        # eq.(33)-(35) Spreiter & Walter Journal of Computational Physics 152, 102–119 (1999) 
        v_temp[sp_start:sp_end,0] = ptcls.vel[sp_start:sp_end,0]*cdt + ptcls.vel[sp_start:sp_end,1]*sdt \
            + inv_omc*( ptcls.acc[sp_start:sp_end,0]*sdt - ptcls.acc[sp_start:sp_end,1]*ccodt ) \
            + inv_omc_sq*( ptcls.acc[sp_start:sp_end,0]*ccodt + ptcls.acc[sp_start:sp_end,1]*ssodt)/dt

        v_temp[sp_start:sp_end,1] = ptcls.vel[sp_start:sp_end,1]*cdt - ptcls.vel[sp_start:sp_end,0]*sdt \
            + inv_omc*( ptcls.acc[sp_start:sp_end,1]*sdt + ptcls.acc[sp_start:sp_end,0]*ccodt ) \
            + inv_omc_sq*( ptcls.acc[sp_start:sp_end,1]*ccodt - ptcls.acc[sp_start:sp_end,0]*ssodt)/dt
        
        ptcls.vel[sp_start:sp_end,2] = ptcls.vel[sp_start:sp_end,2] + 0.0*half_dt*ptcls.acc[sp_start:sp_end,2]

        sp_start = sp_end
    
    # Periodic boundary condition
    if params.Control.PBC == 1:
        EnforcePBC(ptcls.pos,params.Lv)
        
    # Compute total potential energy and accleration for second half step velocity update                 
    #U = PotentialAcceleration(ptcls,params)
    U = 0.0

    sp_start = 0
    sp_end = 0

    for ic in range(params.num_species):

        omega_c = params.species[ic].omega_c

        inv_omc = 1.0/(omega_c)
        inv_omc_sq = inv_omc*inv_omc
        omc_dt = omega_c*dt
        sdt = np.sin( omc_dt )
        cdt = np.cos( omc_dt )

        ccodt = cdt - 1.0
        ssodt = sdt - omc_dt

        sp_end = sp_start + params.species[ic].num

        # Second half step velocity update
        # eq.(28)-(30) Spreiter & Walter Journal of Computational Physics 152, 102–119 (1999) 
        ptcls.vel[sp_start:sp_end,0] = v_temp[sp_start:sp_end,0] \
            - inv_omc_sq*( ptcls.acc[sp_start:sp_end,0]*ccodt + ptcls.acc[sp_start:sp_end,1]*ssodt )/dt 

        ptcls.vel[sp_start:sp_end,1] = v_temp[sp_start:sp_end,1] \
            - inv_omc_sq*( ptcls.acc[sp_start:sp_end,1]*ccodt - ptcls.acc[sp_start:sp_end,0]*ssodt )/dt 

        ptcls.vel[sp_start:sp_end,2] = ptcls.vel[sp_start:sp_end,2] + 0.0*half_dt*ptcls.acc[sp_start:sp_end,2]
        
        sp_start = sp_end

    return U

def Verlet_with_Langevin(ptcls, params):
        """ 
        Calculate particles dynamics using the Velocity Verlet algorithm and Langevin damping.

        Parameters
        ----------
        ptlcs: class
               Particles data. See S_particles.py for more info.
        
        params : class
                Simulation's parameters. See S_params.py for more info.
                
        Returns
        -------
        U : float
            Total potential energy
        """

        dt = params.dt
        g = params.Langevin.gamma
        Gamma = params.Potential.Gamma
        N = ptcls.pos.shape[0]

        rtdt = np.sqrt(dt)

        sp_start = 0 # start index for species loop
        sp_end = 0 # end index for species loop

        beta = np.random.normal(0., 1., 3*N).reshape(N, 3)

        for ic in range( params.num_species ):
            # sigma
            sig = np.sqrt(2. * g*params.kB*params.T_desired/params.species[ic].mass)
        
            c1 = (1. - 0.5*g*dt)
            c2 = 1./(1. + 0.5*g*dt)
            
            sp_start = sp_end
            sp_end += params.species[ic].num

            ptcls.pos[sp_start:sp_end,:] = ptcls.pos[sp_start:sp_end,:] + c1*dt*ptcls.vel[sp_start:sp_end,:]\
                        + 0.5*dt**2*ptcls.acc[sp_start:sp_end,:] + 0.5*sig*dt**1.5*beta

        # Periodic boundary condition
        if params.Control.PBC == 1:
            EnforcePBC(ptcls.pos,params.Lv)

        acc_old = ptcls.acc
        U = PotentialAcceleration(ptcls,params)
        
        acc_new = ptcls.acc

        for ic in range( params.num_species ):
            # sigma
            sig = np.sqrt(2. * g*params.kB*params.T_desired/params.species[ic].mass)
        
            c1 = (1. - 0.5*g*dt)
            c2 = 1./(1. + 0.5*g*dt)
            
            sp_start = sp_end
            sp_end += params.species[ic].num

        ptcls.vel[sp_start:sp_end,:] = c1*c2*ptcls.vel[sp_start:sp_end,:] \
                    + 0.5*dt*(acc_new[sp_start:sp_end,:] + acc_old[sp_start:sp_end,:])*c2 + c2*sig*rtdt*beta
        
        return U

@nb.njit
def EnforcePBC(pos, BoxVector):
    """ 
    Enforce Periodic Boundary Conditions. 

    Parameters
    ----------
    pos : array
          particles' positions. See S_particles.py for more info.

    BoxVector : array
                Box Dimensions

    Returns
    -------

    none
    
    """

    # Loop over all particles
    for i in np.arange(pos.shape[0]):
        for p in np.arange(pos.shape[1]):
            
            # If particle is outside of box in positive direction, wrap to negative side
            if pos[i, p] > BoxVector[p]:
                pos[i, p] = pos[i, p] - BoxVector[p]
            
            # If particle is outside of box in negative direction, wrap to positive side
            if pos[i, p] < 0.0:
                pos[i, p] = pos[i, p] + BoxVector[p]

    return

@nb.njit
def DipoleTerm(pos,charge):
    """ 
    Calculate the dipole term of the energy.
    See Caillol J Chem Phys 101 6080 (1994) for more info.

    Parameters
    ----------
    pos : array
          particles' positions. See S_particles.py for more info.

    charge : array
            particles' charges. See S_particles.py for more info.
    
    Returns
    -------
    dipole : array
            dipole

    """
    dipole = np.zeros( 3 )
    for i in range( pos.shape[0] ):
        dipole += charge[i]*pos[i,:]
        
    return dipole

def PotentialAcceleration(ptcls,params):
    """ 
    Calculate the Potential and update particles' accelerations.

    Parameters
    ----------
    ptcls : class
            Particles' data. See S_particles for more information

    params : class
            Simulations data. See S_params for more information

    Returns
    -------
    U : float
        Potential

    """
    
    if (params.Potential.method == 'brute'):
        U, acc = force_pp.update_brute(ptcls,params)
        ptcls.acc = acc
    else:
        if (params.Potential.LL_on):
            U_short, acc_s_r = force_pp.update(ptcls,params)
        else:
            U_short, acc_s_r = force_pp.update_0D(ptcls,params)
    
        ptcls.acc = acc_s_r

        U = U_short

        if (params.P3M.on):
            U_long, acc_l_r = force_pm.update(ptcls,params)
            # Ewald Self-energy
            U_Ew_self = params.QFactor*params.P3M.G_ew/np.sqrt(np.pi)
            # Neutrality condition
            U_neutr = - np.pi*params.neutrality**2.0/(2.0*params.box_volume*params.P3M.G_ew**2)

            U = U + U_long - U_Ew_self + U_neutr
            ptcls.acc = ptcls.acc + acc_l_r
        
    if not (params.Potential.type == "LJ"):
        # Mie Energy
        dipole = DipoleTerm(ptcls.pos,ptcls.charge)
        U_MIE = 2.0*np.pi*(dipole[0]**2 + dipole[1]**2 + dipole[2]**2)/(3.0*params.box_volume)

        U = U + U_MIE

    return U
