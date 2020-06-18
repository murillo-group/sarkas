"""
Module of various types of integrators 
"""

import numpy as np
import numba as nb
import fmm3dpy as fmm
import sys
import S_calc_force_pp as force_pp
import S_calc_force_pm as force_pm


class Integrator:
    """
    Assign integrator type.

    Parameters
    ----------
        params : class
        Simulation's parameters.

    Attributes
    ----------
        update : func
            Integrator choice. 'Verlet' or 'Magnetic_Verlet'.

    """

    def __init__(self, params):

        if params.Integrator.type == "Verlet":
            if params.Langevin.on:
                if params.Langevin.type == "BBK":
                    self.update = self.Verlet_with_Langevin  # currently only BBK.
                else:
                    print("No such Langevin type.")
                    sys.exit()
            else:
                self.update = Verlet
        elif params.Integrator.type == "Magnetic_Verlet":
            self.update = Magnetic_Verlet
        elif params.Integrator.type == "Magnetic_Boris":
            self.update = Boris_Magnetic_integrator
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
        dt = self.params.Control.dt
        half_dt = 0.5 * dt
        N = self.params.tot_num_ptcls
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
        dt = self.params.Control.dt
        N = self.params.tot_num_ptcls
        d = self.params.d
        Lv = self.params.Lv
        PBC = self.params.PBC
        Lmax_v = self.params.Lmax_v
        Lmin_v = self.params.Lmin_v
        pass

    def RK45_with_Langevin(self, ptcls):
        pass


def Verlet(ptcls, params):
    """ 
    Update particle position and velocity based on velocity verlet method.
    More information can be found here: https://en.wikipedia.org/wiki/Verlet_integration
    or on the Sarkas website. 

    Parameters
    ----------
    ptcls: class
        Particles data. See ``S_particles.py`` for more info.
    
    params : class
        Simulation's parameters. See ``S_params.py`` for more info.

    Returns
    -------
    U : float
        Total potential energy
    
    """

    # First half step velocity update
    ptcls.vel += 0.5 * ptcls.acc * params.Control.dt
    # Full step position update
    ptcls.pos += ptcls.vel * params.Control.dt

    # Periodic boundary condition
    if not params.Potential.method == 'FMM':
        enforce_pbc(ptcls.pos, ptcls.pbc_cntr, params.Lv)
        # Compute total potential energy and acceleration for second half step velocity update
        U = calc_pot_acc(ptcls, params)
    else:
        U = calc_pot_acc_fmm(ptcls, params)

    # Second half step velocity update
    ptcls.vel += 0.5 * ptcls.acc * params.Control.dt

    return U


def Magnetic_Verlet(ptcls, params):
    """
     Update particles' positions and velocities based on velocity verlet method in the case of a
     constant magnetic field along the :math:`z` axis. For more info see eq. (78) of Ref. [1]_

     Parameters
     ----------
     ptlcs: class
            Particles data. See ``S_particles.py`` for more info.

     params : class
             Simulation's parameters. See ``S_params.py`` for more info.

     Returns
     -------
     U : float
         Total potential energy.

     References
     ----------
     .. [1] `Chin Phys Rev E 77, 066401 (2008) <https://doi.org/10.1103/PhysRevE.77.066401>`_

     """
    # Time step
    dt = params.Control.dt
    half_dt = 0.5 * dt

    sp_start = 0  # start index for species loop

    # array to temporary store velocities
    v_B = np.zeros((params.total_num_ptcls, params.dimensions))
    v_F = np.zeros((params.total_num_ptcls, params.dimensions))

    for ic in range(params.num_species):
        # Cyclotron frequency
        omega_c = params.species[ic].omega_c
        omc_dt = omega_c * half_dt

        sdt = np.sin(omc_dt)
        cdt = np.cos(omc_dt)
        ccodt = cdt - 1.0

        sp_end = sp_start + params.species[ic].num
        # First half step of velocity update
        v_B[sp_start:sp_end, 0] = ptcls.vel[sp_start:sp_end, 0] * cdt - ptcls.vel[sp_start:sp_end, 1] * sdt
        v_F[sp_start:sp_end, 0] = - ccodt / omega_c * ptcls.acc[sp_start:sp_end, 1] \
                                  + sdt / omega_c * ptcls.acc[sp_start:sp_end, 0]

        v_B[sp_start:sp_end, 1] = ptcls.vel[sp_start:sp_end, 1] * cdt + ptcls.vel[sp_start:sp_end, 0] * sdt
        v_F[sp_start:sp_end, 1] = ccodt / omega_c * ptcls.acc[sp_start:sp_end, 0] \
                                  + sdt / omega_c * ptcls.acc[sp_start:sp_end, 1]

        ptcls.vel[sp_start:sp_end, 0] = v_B[sp_start:sp_end, 0] + v_F[sp_start:sp_end, 0]
        ptcls.vel[sp_start:sp_end, 1] = v_B[sp_start:sp_end, 1] + v_F[sp_start:sp_end, 1]
        ptcls.vel[sp_start:sp_end, 2] += half_dt * ptcls.acc[sp_start:sp_end, 2]

        # Position update
        ptcls.pos[sp_start:sp_end, 0] += (v_B[sp_start:sp_end, 0] + v_F[sp_start:sp_end, 0]) * dt
        ptcls.pos[sp_start:sp_end, 1] += (v_B[sp_start:sp_end, 1] + v_F[sp_start:sp_end, 1]) * dt
        ptcls.pos[sp_start:sp_end, 2] += ptcls.vel[sp_start:sp_end, 2] * dt

        sp_start = sp_end

    # Periodic boundary condition
    enforce_pbc(ptcls.pos, ptcls.pbc_cntr, params.Lv)

    # Compute total potential energy and acceleration for second half step velocity update
    U = calc_pot_acc(ptcls, params)

    sp_start = 0

    for ic in range(params.num_species):
        omega_c = params.species[ic].omega_c

        omc_dt = omega_c * dt
        sdt = np.sin(omc_dt)
        cdt = np.cos(omc_dt)

        ccodt = cdt - 1.0

        sp_end = sp_start + params.species[ic].num

        # Second half step velocity update
        ptcls.vel[sp_start:sp_end, 0] = (v_B[sp_start:sp_end, 0] + v_F[sp_start:sp_end, 0]) * cdt \
                                        - (v_B[sp_start:sp_end, 1] + v_F[sp_start:sp_end, 1]) * sdt \
                                        - ccodt / omega_c * ptcls.acc[sp_start:sp_end, 1] \
                                        + sdt / omega_c * ptcls.acc[sp_start:sp_end, 0]

        ptcls.vel[sp_start:sp_end, 1] = (v_B[sp_start:sp_end, 1] + v_F[sp_start:sp_end, 1]) * cdt \
                                        + (v_B[sp_start:sp_end, 0] + v_F[sp_start:sp_end, 0]) * sdt \
                                        + ccodt / omega_c * ptcls.acc[sp_start:sp_end, 0] \
                                        + sdt / omega_c * ptcls.acc[sp_start:sp_end, 1]

        ptcls.vel[sp_start:sp_end, 2] += half_dt * ptcls.acc[sp_start:sp_end, 2]

        sp_start = sp_end

    return U


def Boris_Magnetic_integrator(ptcls, params):
    """
     Update particles' positions and velocities using the Boris algorithm in the case of a
     constant magnetic field along the :math:`z` axis. For more info see eqs. (80) - (81) of Ref. [1]_

     Parameters
     ----------
     ptlcs: class
            Particles data. See ``S_particles.py`` for more info.

     params : class
             Simulation's parameters. See ``S_params.py`` for more info.

     Returns
     -------
     U : float
         Total potential energy.

     References
     ----------
     .. [1] `Chin Phys Rev E 77, 066401 (2008) <https://doi.org/10.1103/PhysRevE.77.066401>`_

     """
    # Time step
    dt = params.Control.dt
    half_dt = 0.5 * dt

    sp_start = 0  # start index for species loop

    # array to temporary store velocities
    v_B = np.zeros((params.tot_num_ptcls, params.dimensions))
    v_F = np.zeros((params.tot_num_ptcls, params.dimensions))

    # First step update velocities
    ptcls.vel += 0.5 * ptcls.acc * params.Control.dt

    # Rotate velocities
    for ic in range(params.num_species):
        # Cyclotron frequency
        omega_c = params.species[ic].omega_c
        omc_dt = omega_c * half_dt

        sdt = np.sin(omc_dt)
        cdt = np.cos(omc_dt)

        sp_end = sp_start + params.species[ic].num
        # First half step of velocity update
        v_B[sp_start:sp_end, 0] = ptcls.vel[sp_start:sp_end, 0] * cdt - ptcls.vel[sp_start:sp_end, 1] * sdt

        v_B[sp_start:sp_end, 1] = ptcls.vel[sp_start:sp_end, 1] * cdt + ptcls.vel[sp_start:sp_end, 0] * sdt

        ptcls.vel[sp_start:sp_end, 0] = v_B[sp_start:sp_end, 0]
        ptcls.vel[sp_start:sp_end, 1] = v_B[sp_start:sp_end, 1]

        sp_start = sp_end

    # Second step update velocities
    ptcls.vel += 0.5 * ptcls.acc * params.Control.dt

    # Full step position update
    ptcls.pos += ptcls.vel * params.Control.dt

    # Periodic boundary condition
    enforce_pbc(ptcls.pos, ptcls.pbc_cntr, params.Lv)

    # Compute total potential energy and acceleration for second half step velocity update
    U = calc_pot_acc(ptcls, params)

    return U


@nb.njit
def enforce_pbc(pos, cntr, BoxVector):
    """ 
    Enforce Periodic Boundary conditions. 

    Parameters
    ----------
    pos : array
        particles' positions. See ``S_particles.py`` for more info.

    cntr: array
        Counter for the number of times each particle get folded back into the main simulation box

    BoxVector : array
        Box Dimensions.

    """

    # Loop over all particles
    for p in np.arange(pos.shape[0]):
        for d in np.arange(pos.shape[1]):

            # If particle is outside of box in positive direction, wrap to negative side
            if pos[p, d] > BoxVector[d]:
                pos[p, d] -= BoxVector[d]
                cntr[p, d] += 1
            # If particle is outside of box in negative direction, wrap to positive side
            if pos[p, d] < 0.0:
                pos[p, d] += BoxVector[d]
                cntr[p, d] -= 1
    return


def calc_pot_acc(ptcls, params):
    """ 
    Calculate the Potential and update particles' accelerations.

    Parameters
    ----------
    ptcls : class
        Particles' data. See ``S_particles.py`` for more information.

    params : class
        Simulation's parameters. See ``S_params.py`` for more information.

    Returns
    -------
    U : float
        Total Potential.

    """
    if params.Potential.LL_on:
        U_short, acc_s_r = force_pp.update(ptcls.pos, ptcls.species_id, ptcls.mass, params.Lv,
                                           params.Potential.rc, params.Potential.matrix, params.force,
                                           params.Control.measure, ptcls.rdf_hist)
    else:
        U_short, acc_s_r = force_pp.update_0D(ptcls.pos, ptcls.species_id, ptcls.mass, params.Lv,
                                              params.Potential.rc, params.Potential.matrix, params.force,
                                              params.Control.measure, ptcls.rdf_hist)

    ptcls.acc = acc_s_r

    U = U_short

    if params.P3M.on:
        U_long, acc_l_r = force_pm.update(ptcls.pos, ptcls.charge, ptcls.mass,
                                          params.P3M.MGrid, params.Lv, params.P3M.G_k, params.P3M.kx_v, params.P3M.ky_v,
                                          params.P3M.kz_v, params.P3M.cao)
        # Ewald Self-energy
        U_long += params.QFactor * params.P3M.G_ew / np.sqrt(np.pi)
        # Neutrality condition
        U_long += - np.pi * params.tot_net_charge ** 2.0 / (2.0 * params.box_volume * params.P3M.G_ew ** 2)

        U += U_long

        ptcls.acc += acc_l_r

    if not (params.Potential.type == "LJ"):
        # Mie Energy of charged systems
        # J-M.Caillol, J Chem Phys 101 6080(1994) https: // doi.org / 10.1063 / 1.468422
        dipole = ptcls.charge @ ptcls.pos
        U += 2.0 * np.pi * np.sum(dipole ** 2) / (3.0 * params.box_volume * params.fourpie0)

    return U


def calc_pot_acc_fmm(ptcls, params):
    """

    Parameters
    ----------
    ptcls
    params

    Returns
    -------

    """

    if params.Potential.type == 'Coulomb':
        out_fmm = fmm.lfmm3d(eps=1.0e-07, sources=np.transpose(ptcls.pos), charges=ptcls.charge, pg=2)
    elif params.Potential.type == 'Yukawa':
        out_fmm = fmm.hfmm3d(eps=1.0e-05, zk=1j / params.lambda_TF, sources=np.transpose(ptcls.pos),
                         charges=ptcls.charge, pg=2)

    U = ptcls.charge @ out_fmm.pot.real * 4.0 * np.pi / params.fourpie0
    ptcls.acc = - np.transpose(ptcls.charge * out_fmm.grad.real / ptcls.mass) / params.fourpie0

    return U