'''
S_integrator.py

velocity integrator

Verlet: velocity Verlet
Verlet_with_Langevin: Verlet with Langevin damping
RK45: N/A
RK45_with_Langevin: N/A
'''

import numpy as np
import sys
import S_p3m as p3m
import S_constants as const
import S_yukawa_gf_opt as yukawa_gf_opt
import S_force as force

class Integrator:
    def __init__(self, params, glb):

        if(params.potential[0].type == "Yukawa" or params.potential[0].type == "yukawa"):
            self.calc_force = p3m.force_pot


        self.params = params
        self.glb_vars = glb

        if(params.Integrator[0].type == "Verlet"):
            if(params.Langevin):
                self.update = self.Verlet_with_Langevin
            else:
                self.update = self.Verlet
        else:
            print("Only Verlet integrator is supported. Check your input file, integrator part.")
            sys.exit()

    def Verlet(self, ptcls):
        ''' Update particle position and velocity based on velocity verlet method.
        More information can be found here: https://en.wikipedia.org/wiki/Verlet_integration
        or on the Sarkas website. 
    
        Parameters
        ----------
        ptlcs: particles data. See S_particles.py for the detailed information

        Returns
        -------
        U : float
            Total potential energy
        '''
        # Import global parameters (is there a better way to do this?)
        dt = self.glb_vars.dt
        N = self.glb_vars.N
        d = self.glb_vars.d
        Lv = self.glb_vars.Lv
        PBC = self.glb_vars.PBC
        Lmax_v = self.glb_vars.Lmax_v
        Lmin_v = self.glb_vars.Lmin_v


        # First half step velocity update
        ptcls.vel = ptcls.vel + 0.5*ptcls.acc*dt
        
        # Full step position update
        ptcls.pos = ptcls.pos + ptcls.vel*dt

        # Periodic boundary condition
        if PBC == 1:
            
            # Loop over all particles
            for i in np.arange(N):
                # Loop over dimensions (x=0, y=1, z=2)
                for p in np.arange(d):
                    
                    # If particle is outside of box in positive direction, wrap to negative side
                    if ptcls.pos[i, p] > Lmax_v[p]:
                        ptcls.pos[i, p] = ptcls.pos[i, p] - Lv[p]
                    
                    # If particle is outside of box in negative direction, wrap to positive side
                    if ptcls.pos[i, p] < Lmin_v[p]:
                        ptcls.pos[i, p] = ptcls.pos[i, p] + Lv[p]

        # Compute total potential energy and accleration for second half step velocity update                 
        #U = p3m.force_pot(ptcls)
        U = self.calc_force(ptcls)
        #U = self.update_force(ptcls)
        
        # Second half step velocity update
        ptcls.vel = ptcls.vel + 0.5*ptcls.acc*dt

        return U


    



    def Verlet_with_Langevin(self, ptcls):
        dt = self.glb_vars.dt
        g = self.glb_vars.g_0
        Gamma = self.glb_vars.Gamma
        Lmax_v = self.glb_vars.Lmax_v
        Lmin_v = self.glb_vars.Lmin_v
        Lv = self.glb_vars.Lv
        PBC = self.glb_vars.PBC
        N = self.glb_vars.N
        d = self.glb_vars.d

        rtdt = np.sqrt(dt)

        sig = np.sqrt(2. * g*const.kb*self.glb_vars.T_desired/const.proton_mass)
        if(self.glb_vars.units == "Yukawa"):
            sig = np.sqrt(2. * g/(3*Gamma))

        c1 = (1. - 0.5*g*dt)
        c2 = 1./(1. + 0.5*g*dt)
        beta = np.random.normal(0., 1., 3*N).reshape(N, 3)

        ptcls.pos = ptcls.pos + c1*dt*ptcls.vel + 0.5*dt**2*ptcls.acc + 0.5*sig*dt**1.5*beta

        # periodic boundary condition
        if PBC == 1:
            for i in np.arange(N):
                for p in np.arange(d):
                    if ptcls.pos[i, p] > Lmax_v[p]:
                        ptcls.pos[i, p] = ptcls.pos[i, p] - Lv[p]
                    if ptcls.pos[i, p] < Lmin_v[p]:
                        ptcls.pos[i, p] = ptcls.pos[i, p] + Lv[p]

        acc = ptcls.acc
        #U = p3m.force_pot(ptcls)
        U = self.calc_force(ptcls)
        #U = self.update_force(ptcls)
        acc_new = ptcls.acc
        ptcls.vel = c1*c2*ptcls.vel + 0.5*dt*(acc_new + acc)*c2 + c2*sig*rtdt*beta
        return U

    def RK45(self):
        pass

    def RK45_with_Langevin(self):
        pass
