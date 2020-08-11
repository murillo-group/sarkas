"""
Module of various types of time_evolution


"""

import numpy as np
from numba import njit
# import fmm3dpy as fmm
from sarkas.algorithm import force_pp, force_pm


def verlet(ptcls, params):
    """
    Update particle position and velocity based on velocity verlet method.
    More information can be found here: https://en.wikipedia.org/wiki/Verlet_integration
    or on the Sarkas website.

    Parameters
    ----------
    ptcls: object
        Particles data.

    params:  object
        Simulation's parameters.

    Returns
    -------
    U : float
        Total potential energy

    """

    # First half step velocity update
    ptcls.vel += 0.5 * ptcls.acc * params.integrator.dt
    # Full step position update
    ptcls.pos += ptcls.vel * params.integrator.dt

    # Periodic boundary condition
    if not params.potential.method == 'FMM':
        enforce_pbc(ptcls.pos, ptcls.pbc_cntr, params.Lv)
        # Compute total potential energy and acceleration for second half step velocity update
        U = calc_pot_acc(ptcls, params)
    # else:
    #     U = calc_pot_acc_fmm(ptcls, params)

    # Second half step velocity update
    ptcls.vel += 0.5 * ptcls.acc * params.integrator.dt

    return U


def verlet_langevin(ptcls, params):
    """
    Calculate particles dynamics using the Velocity verlet algorithm and Langevin damping.

    Parameters
    ----------
    ptcls: object
        Particles data.

    params:  object
        Simulation's parameters.

    Returns
    -------
    U : float
        Total potential energy

    """

    beta = np.random.normal(0., 1., 3 * params.total_num_ptcls).reshape(params.total_num_ptcls, 3)

    sp_start = 0  # start index for species loop
    for ic, sp in enumerate(params.species):
        sp_end = sp_start + sp.num
        # sigma
        sig = np.sqrt(2. * params.Langevin.gamma * params.kB * params.thermostat.temperatures[ic] / sp.mass)

        c1 = (1. - 0.5 * params.Langevin.gamma * params.integrator.dt)
        # c2 = 1./(1. + 0.5*g*dt)

        ptcls.pos[sp_start:sp_end, :] += c1 * params.integrator.dt * ptcls.vel[sp_start:sp_end, :] \
                                         + 0.5 * params.Contro.dt ** 2 * ptcls.acc[sp_start:sp_end, :] \
                                         + 0.5 * sig * params.integrator.dt ** 1.5 * beta

    # Periodic boundary condition
    if params.control.PBC == 1:
        enforce_pbc(ptcls.pos, ptcls.pbc_cntr, params.Lv)

    acc_old = np.copy(ptcls.acc)
    U = calc_pot_acc(ptcls, params)

    sp_start = 0
    for ic, sp in enumerate(params.species):
        sp_end = sp_start + sp.num
        # sigma
        sig = np.sqrt(2. * params.Langevin.gamma * params.kB * params.thermostat.temperatures[ic] / sp.mass)

        c1 = (1. - 0.5 * params.Langevin.gamma * params.integrator.dt)
        c2 = 1. / (1. + 0.5 * params.Langevin.gamma * params.integrator.dt)

        ptcls.vel[sp_start:sp_end, :] = c1 * c2 * ptcls.vel[sp_start:sp_end, :] \
                                        + 0.5 * c2 * params.integrator.dt * (ptcls.acc[sp_start:sp_end, :]
                                                                             + acc_old[sp_start:sp_end, :]) \
                                        + c2 * sig * np.sqrt(params.integrator.dt) * beta
        sp_start = sp_end

    return U


def magnetic_verlet(ptcls, params):
    """
    Update particles' positions and velocities based on velocity verlet method in the case of a
    constant magnetic field along the :math:`z` axis. For more info see eq. (78) of Ref. [Chin2008]_

    Parameters
    ----------
    ptcls: object
        Particles data.

    params:  object
        Simulation's parameters.

    Returns
    -------
    U : float
         Total potential energy.

    References
    ----------
    .. [Chin2008] `Chin Phys Rev E 77, 066401 (2008) <https://doi.org/10.1103/PhysRevE.77.066401>`_
    """
    # Time step
    dt = params.integrator.dt
    half_dt = 0.5 * dt

    sp_start = 0  # start index for species loop

    # array to temporary store velocities
    v_B = np.zeros((params.total_num_ptcls, params.dimensions))
    v_F = np.zeros((params.total_num_ptcls, params.dimensions))

    for ic, sp in enumerate(params.species):
        # Cyclotron frequency
        omega_c = sp.omega_c
        omc_dt = omega_c * half_dt

        sdt = np.sin(omc_dt)
        cdt = np.cos(omc_dt)
        ccodt = cdt - 1.0

        sp_end = sp_start + sp.num
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

    for ic, sp in enumerate(params.species):
        omega_c = sp.omega_c

        omc_dt = omega_c * dt
        sdt = np.sin(omc_dt)
        cdt = np.cos(omc_dt)

        ccodt = cdt - 1.0

        sp_end = sp_start + sp.num

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


def magnetic_boris(ptcls, params):
    """
    Update particles' positions and velocities using the Boris algorithm in the case of a
    constant magnetic field along the :math:`z` axis. For more info see eqs. (80) - (81) of Ref. [Chin2008]_

    Parameters
    ----------
    ptcls: object
        Particles data.

    params:  object
        Simulation's parameters.

    Returns
    -------
    U : float
         Total potential energy.

    """
    # Time step
    dt = params.integrator.dt
    half_dt = 0.5 * dt

    sp_start = 0  # start index for species loop

    # array to temporary store velocities
    v_B = np.zeros((params.tot_num_ptcls, params.dimensions))
    v_F = np.zeros((params.tot_num_ptcls, params.dimensions))

    # First step update velocities
    ptcls.vel += 0.5 * ptcls.acc * params.integrator.dt

    # Rotate velocities
    for ic, sp in enumerate(params.species):
        # Cyclotron frequency
        omega_c = sp.omega_c
        omc_dt = omega_c * half_dt

        sdt = np.sin(omc_dt)
        cdt = np.cos(omc_dt)

        sp_end = sp_start + sp.num
        # First half step of velocity update
        v_B[sp_start:sp_end, 0] = ptcls.vel[sp_start:sp_end, 0] * cdt - ptcls.vel[sp_start:sp_end, 1] * sdt

        v_B[sp_start:sp_end, 1] = ptcls.vel[sp_start:sp_end, 1] * cdt + ptcls.vel[sp_start:sp_end, 0] * sdt

        ptcls.vel[sp_start:sp_end, 0] = v_B[sp_start:sp_end, 0]
        ptcls.vel[sp_start:sp_end, 1] = v_B[sp_start:sp_end, 1]

        sp_start = sp_end

    # Second step update velocities
    ptcls.vel += 0.5 * ptcls.acc * params.integrator.dt

    # Full step position update
    ptcls.pos += ptcls.vel * params.integrator.dt

    # Periodic boundary condition
    enforce_pbc(ptcls.pos, ptcls.pbc_cntr, params.Lv)

    # Compute total potential energy and acceleration for second half step velocity update
    U = calc_pot_acc(ptcls, params)

    return U


@njit
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
    ptcls: object
        Particles data.

    params:  object
        Simulation's parameters.

    Returns
    -------
    U : float
        Total Potential.

    """
    if params.potential.LL_on:
        U_short, acc_s_r = force_pp.update(ptcls.pos, ptcls.species_id, ptcls.mass, params.Lv,
                                           params.potential.rc, params.potential.matrix, params.force,
                                           params.control.measure, ptcls.rdf_hist)
    else:
        U_short, acc_s_r = force_pp.update_0D(ptcls.pos, ptcls.species_id, ptcls.mass, params.Lv,
                                              params.potential.rc, params.potential.matrix, params.force,
                                              params.control.measure, ptcls.rdf_hist)

    ptcls.acc = acc_s_r

    U = U_short

    if params.pppm.on:
        U_long, acc_l_r = force_pm.update(ptcls.pos, ptcls.charge, ptcls.mass,
                                          params.pppm.MGrid, params.Lv, params.pppm.G_k, params.pppm.kx_v,
                                          params.pppm.ky_v,
                                          params.pppm.kz_v, params.pppm.cao)
        # Ewald Self-energy
        U_long += params.QFactor * params.pppm.G_ew / np.sqrt(np.pi)
        # Neutrality condition
        U_long += - np.pi * params.tot_net_charge ** 2.0 / (2.0 * params.box_volume * params.pppm.G_ew ** 2)

        U += U_long

        ptcls.acc += acc_l_r

    if not (params.potential.type == "LJ"):
        # Mie Energy of charged systems
        # J-M.Caillol, J Chem Phys 101 6080(1994) https: // doi.org / 10.1063 / 1.468422
        dipole = ptcls.charge @ ptcls.pos
        U += 2.0 * np.pi * np.sum(dipole ** 2) / (3.0 * params.box_volume * params.fourpie0)

    return U

# def calc_pot_acc_fmm(ptcls, params):
#     """
#
#     Parameters
#     ----------
#     ptcls
#     params
#
#     Returns
#     -------
#
#     """
#
#     if params.potential.type == 'Coulomb':
#         out_fmm = fmm.lfmm3d(eps=1.0e-07, sources=np.transpose(ptcls.pos), charges=ptcls.charge, pg=2)
#     elif params.potential.type == 'Yukawa':
#         out_fmm = fmm.hfmm3d(eps=1.0e-05, zk=1j / params.lambda_TF, sources=np.transpose(ptcls.pos),
#                          charges=ptcls.charge, pg=2)
#
#     U = ptcls.charge @ out_fmm.pot.real * 4.0 * np.pi / params.fourpie0
#     ptcls.acc = - np.transpose(ptcls.charge * out_fmm.grad.real / ptcls.mass) / params.fourpie0
#
#     return U
