import numpy as np
import sys
import S_p3m as p3m
import S_global_names as glb
import S_constants as const

def update(pos, vel, acc, it, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
    ''' Update particle position and velocity based on velocity verlet method.
    More information can be found here: https://en.wikipedia.org/wiki/Verlet_integration
    or on the Sarkas website. 
    
    Parameters
    ----------
    pos : array_like
        Positions of particles
    
    vel : array_like
        Velocities of particles
        
    acc : array_like
        Accelerations of particles
    
    it : THIS DOES NOT GET USED IN THIS FILE...
    
    Z : float
        Ionization?
    
    Returns
    -------
    pos : array_like
        Updated positions of particles
        
    vel : array_like
        Updated velocities of particles
        
    acc : array_like
        Updated accelerations of particles
    
    U : float
        Total potential energy

    Notes
    -----
    The velocity Verlet integration algorithm is comprised of a three-step update
    the following form:

    1. :math:`\mathbf{v}(t + \Delta t/2) = \mathbf{v}(t) + \mathbf{a}(t) \Delta t/2`
    2. :math:`\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \mathbf{v}(t + \Delta t/2) \Delta t`
    3. :math:`\mathbf{v}(t + \Delta t) = \mathbf{v}(t + \Delta t/2) + \mathbf{a}(t + \Delta t)\Delta t/2`   
    '''
    
    # Import global parameters (is there a better way to do this?)
    dt = glb.dt
    N = glb.N
    d = glb.d
    Lv = glb.Lv
    PBC = glb.PBC
    Lmax_v = glb.Lmax_v
    Lmin_v = glb.Lmin_v
    G_k = glb.G_k
    kx_v = glb.kx_v
    ky_v = glb.ky_v
    kz_v = glb.kz_v

    # First half step velocity update
    vel = vel + 0.5*acc*dt
    
    # Full step position update
    pos = pos + vel*dt
    
    # Periodic boundary conditions
    if PBC == 1:
        
        # Loop over all particles
        for i in np.arange(N):
            
            # Loop over dimensions (x=0, y=1, z=2)
            for p in np.arange(d):
        
                # If particle is outside of box in positive direction, wrap to negative side
                if pos[i,p] > Lmax_v[p]:
                    pos[i,p] = pos[i,p] - Lv[p]
                    
                # If particle is outside of box in negative direction, wrap to positive side
                if pos[i,p] < Lmin_v[p]:
                    pos[i,p] = pos[i,p] + Lv[p]
                    
    # Compute total potential energy and accleration for second half step velocity update 
    U, acc = p3m.force_pot(pos, acc, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p)
    
    # Second half step velocity update
    vel = vel + 0.5*acc*dt

    return pos, vel, acc, U

def update_Langevin(pos, vel, acc, Z, acc_s_r, acc_fft, rho_r, E_x_p, E_y_p, E_z_p):
    dt = glb.dt
    g = glb.g_0
    Gamma = glb.Gamma
    Lmax_v = glb.Lmax_v
    Lmin_v = glb.Lmin_v
    Lv = glb.Lv
    PBC = glb.PBC
    N = glb.N
    d = glb.d
    G_k = glb.G_k
    kx_v = glb.kx_v
    ky_v = glb.ky_v
    kz_v = glb.kz_v

    rtdt = np.sqrt(dt)

    sig = np.sqrt(2. * g*const.kb*glb.T_desired/const.proton_mass)
    if(glb.units == "Yukawa"):
        sig = np.sqrt(2. * g/(3*Gamma))

    c1 = (1. - 0.5*g*dt) 
    c2 = 1./(1. + 0.5*g*dt)
    beta = np.random.normal(0., 1., 3*N).reshape(N, 3)

    pos = pos + c1*dt*vel + 0.5*dt**2*acc + 0.5*sig* dt**1.5*beta
    
    # periodic boundary condition
    if PBC == 1:
        for i in np.arange(N):
            for p in np.arange(d):
        
                if pos[i,p] > Lmax_v[p]:
                    pos[i,p] = pos[i,p] - Lv[p]
                if pos[i,p] < Lmin_v[p]:
                    pos[i,p] = pos[i,p] + Lv[p]
        
    
    U, acc_new = p3m.force_pot(pos, acc, Z)
    
    vel = c1*c2*vel + 0.5*dt*(acc_new + acc)*c2 + c2*sig*rtdt*beta
    acc = acc_new
    
    return pos, vel, acc, U
