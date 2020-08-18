"""
Module of various types of time_evolution


"""

import numpy as np
from numba import njit
from tqdm import tqdm
# import fmm3dpy as fmm
from sarkas.potentials import force_pm, force_pp


class Integrator:
    """
    Class used to assign integrator type.

    Parameters
    ----------
    params:  object
        Simulation's parameters.

    Attributes
    ----------
    update: func
        Integrator choice. 'verlet' or 'magnetic_verlet'.

    """

    def __init__(self):
        self.type = None
        self.dt = None
        self.kB = None
        self.magnetized = False
        self.production_steps = None
        self.equilibration_steps = None
        self.prod_dump_step = None
        self.eq_dump_step = None
        self.update = None
        self.species_num = None
        self.box_lengths = None
        self.verbose = False

    def setup(self, params, thermostat, potential):

        self.box_lengths = params.box_lengths
        self.kB = params.kB
        self.species_num = params.species_num
        self.verbose = params.verbose

        if self.prod_dump_step is None:
            self.prod_dump_step = int(0.01 * self.production_steps)

        if self.eq_dump_step is None:
            self.eq_dump_step = self.prod_dump_step

        # Run some checks
        if self.type.lower() == "verlet":
            self.update = self.verlet

        elif self.type.lower() == "verlet_langevin":

            self.sigma = np.sqrt(
                2. * self.langevin_gamma * params.kB * params.species_temperatures / params.species_masses)
            self.c1 = (1. - 0.5 * self.langevin_gamma * self.dt)
            self.c2 = 1. / (1. + 0.5 * self.langevin_gamma * self.dt)
            self.update = self.verlet_langevin

        elif self.type.lower() == "magnetic_verlet":

            self.omega_c = params.species_cyclotron_frequencies
            omc_dt = 0.5 * params.species_cyclotron_frequencies * self.dt
            self.sdt = np.sin(omc_dt)
            self.cdt = np.cos(omc_dt)
            self.ccodt = self.cdt - 1.0

            # array to temporary store velocities
            self.v_B = np.zeros((params.total_num_ptcls, params.dimensions))
            self.v_F = np.zeros((params.total_num_ptcls, params.dimensions))

            self.update = self.magnetic_verlet

        elif self.type.lower() == "magnetic_boris":

            self.update = self.magnetic_boris
        else:
            print("Only verlet integrator is supported. Check your input file, integrator part 2.")

        if not potential.method == 'FMM':
            self.update_accelerations = potential.calc_pot_acc
        else:
            self.update_accelerations = potential.calc_pot_acc_fmm

        self.thermostate = thermostat.update

    def equilibrate(self, it_start, ptcls, checkpoint):

        for it in tqdm(range(it_start, self.equilibration_steps), disable=not self.verbose):
            # Calculate the Potential energy and update particles' data
            U_therm = self.update(ptcls)
            if (it + 1) % self.eq_dump_step == 0:
                checkpoint.dump(False, ptcls, U_therm, it + 1)
            self.thermostate(ptcls, it)

        ptcls.remove_drift()

        return U_therm

    def produce(self, it_start, ptcls, checkpoint):
        ##############################################
        # Production Phase
        ##############################################
        for it in tqdm(range(it_start, self.production_steps), disable=(not self.verbose)):

            # Move the particles and calculate the potential
            U_prod = self.update(ptcls)
            if (it + 1) % self.prod_dump_step == 0:
                # Save particles' data for restart
                checkpoint.dump(True, ptcls, U_prod, it + 1)

    def verlet_langevin(self, ptcls):
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
        potential_energy : float
            Total potential energy

        """
        beta = ptcls.gaussian(0., 1., ptcls.pos.shape[0])
        sp_start = 0  # start index for species loop
        sp_end = 0
        for ic, num in enumerate(self.species_num):
            sp_end += num
            ptcls.pos[sp_start:sp_end, :] += self.c1 * self.dt * ptcls.vel[sp_start:sp_end, :] \
                                             + 0.5 * self.dt ** 2 * ptcls.acc[sp_start:sp_end, :] \
                                             + 0.5 * self.sigma[ic] * self.dt ** 1.5 * beta

        # Periodic boundary condition
        enforce_pbc(ptcls.pos, ptcls.pbc_cntr, self.box_lengths)

        acc_old = np.copy(ptcls.acc)
        potential_energy = self.update_accelerations(ptcls)

        sp_start = 0
        sp_end = 0
        for ic, num in enumerate(self.species_num):
            sp_end += num

            ptcls.vel[sp_start:sp_end, :] = self.c1 * self.c2 * ptcls.vel[sp_start:sp_end, :] \
                                            + 0.5 * self.c2 * self.dt * (ptcls.acc[sp_start:sp_end, :]
                                                                         + acc_old[sp_start:sp_end, :]) \
                                            + self.c2 * self.sigma[ic] * np.sqrt(self.dt) * beta
            sp_start = sp_end

        return potential_energy

    def verlet(self, ptcls):
        """
        Update particle position and velocity based on velocity verlet method.
        More information can be found here: https://en.wikipedia.org/wiki/Verlet_integration
        or on the Sarkas website.

        Parameters
        ----------
        ptcls: object
            Particles data.

        Returns
        -------
        potential_energy : float
            Total potential energy

        """

        # First half step velocity update
        ptcls.vel += 0.5 * ptcls.acc * self.dt
        # Full step position update
        ptcls.pos += ptcls.vel * self.dt

        # Periodic boundary condition
        enforce_pbc(ptcls.pos, ptcls.pbc_cntr, self.box_lengths)
        # Compute total potential energy and acceleration for second half step velocity update
        potential_energy = self.update_accelerations(ptcls)

        # Second half step velocity update
        ptcls.vel += 0.5 * ptcls.acc * self.dt

        return potential_energy

    def magnetic_verlet(self, ptcls):
        """
        Update particles' positions and velocities based on velocity verlet method in the case of a
        constant magnetic field along the :math:`z` axis. For more info see eq. (78) of Ref. [Chin2008]_

        Parameters
        ----------
        ptcls: object
            Particles data.

        Returns
        -------
        potential_energy : float
             Total potential energy.

        References
        ----------
        .. [Chin2008] `Chin Phys Rev E 77, 066401 (2008) <https://doi.org/10.1103/PhysRevE.77.066401>`_
        """

        sp_start = 0  # start index for species loop
        sp_end = 0
        for ic, num in enumerate(self.species_num):
            # Cyclotron frequency
            sp_end += num
            # First half step of velocity update
            self.v_B[sp_start:sp_end, 0] = ptcls.vel[sp_start:sp_end, 0] * self.cdt[ic] \
                                           - ptcls.vel[sp_start:sp_end, 1] * self.sdt[ic]
            self.v_F[sp_start:sp_end, 0] = - self.ccodt[ic] / self.self.omega_c[ic] * ptcls.acc[sp_start:sp_end, 1] \
                                           + self.sdt[ic] / self.self.omega_c[ic] * ptcls.acc[sp_start:sp_end, 0]

            self.v_B[sp_start:sp_end, 1] = ptcls.vel[sp_start:sp_end, 1] * self.cdt[ic] \
                                           + ptcls.vel[sp_start:sp_end, 0] * self.sdt[ic]
            self.v_F[sp_start:sp_end, 1] = self.ccodt[ic] / self.omega_c[ic] * ptcls.acc[sp_start:sp_end, 0] \
                                           + self.sdt[ic] / self.omega_c[ic] * ptcls.acc[sp_start:sp_end, 1]

            ptcls.vel[sp_start:sp_end, 0] = self.v_B[sp_start:sp_end, 0] + self.v_F[sp_start:sp_end, 0]
            ptcls.vel[sp_start:sp_end, 1] = self.v_B[sp_start:sp_end, 1] + self.v_F[sp_start:sp_end, 1]
            ptcls.vel[sp_start:sp_end, 2] += 0.5 * self.dt * ptcls.acc[sp_start:sp_end, 2]

            # Position update
            ptcls.pos[sp_start:sp_end, 0] += (self.v_B[sp_start:sp_end, 0] + self.v_F[sp_start:sp_end, 0]) * self.dt
            ptcls.pos[sp_start:sp_end, 1] += (self.v_B[sp_start:sp_end, 1] + self.v_F[sp_start:sp_end, 1]) * self.dt
            ptcls.pos[sp_start:sp_end, 2] += ptcls.vel[sp_start:sp_end, 2] * self.dt

            sp_start = sp_end

        # Periodic boundary condition
        enforce_pbc(ptcls.pos, ptcls.pbc_cntr, self.box_lengths)

        # Compute total potential energy and acceleration for second half step velocity update
        potential_energy = self.update_accelerations(ptcls)

        sp_start = 0
        sp_end = 0
        for ic, num in enumerate(self.species_num):
            sp_end += num

            # Second half step velocity update
            ptcls.vel[sp_start:sp_end, 0] = (self.v_B[sp_start:sp_end, 0] + self.v_F[sp_start:sp_end, 0]) * self.cdt[ic] \
                                            - (self.v_B[sp_start:sp_end, 1] + self.v_F[sp_start:sp_end, 1]) * self.sdt[
                                                ic] \
                                            - self.ccodt[ic] / self.omega_c[ic] * ptcls.acc[sp_start:sp_end, 1] \
                                            + self.sdt[ic] / self.omega_c[ic] * ptcls.acc[sp_start:sp_end, 0]

            ptcls.vel[sp_start:sp_end, 1] = (self.v_B[sp_start:sp_end, 1] + self.v_F[sp_start:sp_end, 1]) * self.cdt[ic] \
                                            + (self.v_B[sp_start:sp_end, 0] + self.v_F[sp_start:sp_end, 0]) * self.sdt[
                                                ic] \
                                            + self.ccodt[ic] / self.omega_c[ic] * ptcls.acc[sp_start:sp_end, 0] \
                                            + self.sdt[ic] / self.omega_c[ic] * ptcls.acc[sp_start:sp_end, 1]

            ptcls.vel[sp_start:sp_end, 2] += 0.5 * self.dt * ptcls.acc[sp_start:sp_end, 2]

            sp_start = sp_end

        return potential_energy

    def magnetic_boris(self, ptcls):
        """
        Update particles' positions and velocities using the Boris algorithm in the case of a
        constant magnetic field along the :math:`z` axis. For more info see eqs. (80) - (81) of Ref. [Chin2008]_

        Parameters
        ----------
        ptcls: object
            Particles data.

        Returns
        -------
        potential_energy : float
             Total potential energy.

        """
        # First step update velocities
        ptcls.vel += 0.5 * ptcls.acc * self.dt

        sp_start = 0  # start index for species loop
        sp_end = 0  # end index for species loop
        # Rotate velocities
        for ic, num in enumerate(self.species_num):
            # Cyclotron frequency
            sp_end += num
            # First half step of velocity update
            self.v_B[sp_start:sp_end, 0] = ptcls.vel[sp_start:sp_end, 0] * self.cdt[ic] \
                                      - ptcls.vel[sp_start:sp_end, 1] * self.sdt[ic]

            sp_start = 0  # start index for species loop
            self.v_B[sp_start:sp_end, 1] = ptcls.vel[sp_start:sp_end, 1] * self.cdt[ic] \
                                      + ptcls.vel[sp_start:sp_end, 0] * self.sdt[ic]

            ptcls.vel[sp_start:sp_end, 0] = self.v_B[sp_start:sp_end, 0]
            ptcls.vel[sp_start:sp_end, 1] = self.v_B[sp_start:sp_end, 1]

            sp_start = sp_end

        # Compute total potential energy and acceleration for second half step velocity update
        potential_energy = self.update_accelerations(ptcls)

        # Second step update velocities
        ptcls.vel += 0.5 * ptcls.acc * self.dt

        # Full step position update
        ptcls.pos += ptcls.vel * self.dt

        # Periodic boundary condition
        enforce_pbc(ptcls.pos, ptcls.pbc_cntr, self.box_lengths)


        return potential_energy


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


@njit
def calc_kin_temp(vel, nums, masses, kB):
    """
    Calculates the kinetic energy and temperature.

    Parameters
    ----------
    kB: float
        Boltzmann constant in chosen units.

    masses: array
        Mass of each species.

    nums: array
        Number of particles of each species.

    vel: array
        Particles' velocities.

    Returns
    -------
    K : array
        Kinetic energy of each species.

    T : array
        Temperature of each species.
    """

    num_species = len(nums)

    K = np.zeros(num_species)
    T = np.zeros(num_species)
    const = 2.0 / (kB * num_species * vel.shape[1])
    kinetic_energies = 0.5 * masses * vel * vel

    species_start = 0
    species_end = 0
    for i, num in range(nums):
        species_end += num
        K[i] = np.sum(kinetic_energies[species_start:species_end, :])
        T[i] = const[i] * K[i]
        species_start = species_end

    return K, T


@njit
def remove_drift(vel, nums, masses):
    """
    Enforce conservation of total linear momentum. Updates ``ptcls.vel``

    Parameters
    ----------
    vel: array
        Particles' velocities.

    nums: array
        Number of particles of each species.

    masses: array
        Mass of each species.

    """

    P = np.zeros((len(nums), vel.shape[1]))
    species_start = 0
    for ic in range(len(nums)):
        species_end = species_start + nums[ic]
        P[ic, :] = np.sum(vel[species_start:species_end, :], axis=0) * masses[ic]
        species_start = species_end

    if np.sum(P[:, 0]) > 1e-40 or np.sum(P[:, 1]) > 1e-40 or np.sum(P[:, 2]) > 1e-40:
        # Remove tot momentum
        species_start = 0
        for ic in range(len(nums)):
            species_end = species_start + nums[ic]
            vel[species_start:species_end, :] -= P[ic, :] / (float(nums[ic]) * masses[ic])
            species_start = species_end

