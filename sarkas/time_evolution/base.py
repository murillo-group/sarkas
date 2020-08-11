"""
Module of various types of time_evolution


"""

import numpy as np
from numba import nb
from tqdm import tqdm
from . import integrators, thermostats


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
        self.nsteps_prod = None
        self.nsteps_eq = None
        self.prod_dump_step = None
        self.eq_dump_step = None
        self.update = None

    def assign_update(self):

        if self.type.lower() == "verlet":
            self.update = integrators.verlet
        elif self.type.lower() == "verlet_langevin":
            self.update = integrators.verlet_langevin
        elif self.type.lower() == "magnetic_verlet":
            self.update = integrators.magnetic_verlet
        elif self.type.lower() == "magnetic_boris":
            self.update = integrators.magnetic_boris
        else:
            print("Only verlet integrator is supported. Check your input file, integrator part 2.")

    def equilibrate(self, ptcls, params, thermostat, checkpoint):

        if params.load_method == "therm_restart":
            it_start = params.load_therm_restart_step
        else:
            it_start = 0

        for it in tqdm(range(it_start, params.integrator.nsteps_eq), disable=not params.control.verbose):
            # Calculate the Potential energy and update particles' data
            U_therm = self.update(ptcls, params)
            if (it + 1) % params.control.therm_dump_step == 0:
                Ks, Tps = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
                checkpoint.dump(False, ptcls, Ks, Tps, U_therm, it + 1)
            thermostat.update(ptcls.vel, it)

        remove_drift(ptcls.vel, ptcls.species_num, ptcls.species_mass)

        return U_therm

    def produce(self, ptcls, params, checkpoint):

        ##############################################
        # Prepare for Production Phase
        ##############################################

        # Open output files
        if params.load_method == "restart":
            it_start = params.load_restart_step
            if params.control.writexyz:
                # Particles' positions, velocities, accelerations for OVITO
                f_xyz = open(params.control.job_dir + "/" + "pva_" + params.control.job_id + ".xyz", "a+")
                pscale = 1.0 / params.aws
                vscale = 1.0 / (params.aws * params.wp)
                ascale = 1.0 / (params.aws * params.wp ** 2)

        else:
            it_start = 0
            # Restart the pbc counter

            ptcls.pbc_cntr.fill(0.0)
            # Create array for storing energy information
            if params.control.writexyz:
                # Particles' positions, velocities, accelerations for OVITO
                f_xyz = open(params.control.job_dir + "/" + "pva_" + params.control.job_id + ".xyz", "w+")
                pscale = 1.0 / params.aws
                vscale = 1.0 / (params.aws * params.wp)
                ascale = 1.0 / (params.aws * params.wp ** 2)

        # Update measurement flag for rdf
        params.control.measure = True

        ##############################################
        # Production Phase
        ##############################################
        for it in tqdm(range(it_start, params.integrator.nsteps_prod), disable=(not params.control.verbose)):
            # Move the particles and calculate the potential
            U_prod = self.update(ptcls, params)
            if (it + 1) % params.control.dump_step == 0:
                # Save particles' data for restart
                Ks, Tps = calc_kin_temp(ptcls.vel, ptcls.species_num, ptcls.species_mass, params.kB)
                checkpoint.dump(True, ptcls, Ks, Tps, U_prod, it + 1)
                # Write particles' data to XYZ file for OVITO Visualization
                if params.control.writexyz:
                    f_xyz.writelines("{0:d}\n".format(params.total_num_ptcls))
                    f_xyz.writelines("name x y z vx vy vz ax ay az\n")
                    np.savetxt(f_xyz,
                               np.c_[ptcls.species_name, ptcls.pos * pscale, ptcls.vel * vscale, ptcls.acc * ascale],
                               fmt="%s %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e")


class Thermostat:
    """
    Thermostat object.

    Parameters
    ----------
    params : object
        Simulation's parameters

    Attributes
    ----------
    kB : float
        Boltzmann constant in correct units.

    no_species : int
        Total number of species.

    species_np : array
        Number of particles of each species.

    species_masses : array
        Mass of each species.

    relaxation_rate: float
        Berendsen parameter tau.

    relaxation_timestep: int
        Timestep at which thermostat is turned on.

    T_desired: array
        Thermostating temperature of each species.

    type: str
        Thermostat type

    """
    def __init(self):
        self.temperatures = None
        self.type = None
        self.relaxation_rate = None
        self.relaxation_timestep = None

    def assign_attributes(self, params):
        if params.thermostat.on:
            self.kB = params.kB
            self.no_species = len(params.species)
            self.species_np = np.zeros(self.no_species)
            self.species_masses = np.zeros(self.no_species)

            for i, sp in enumerate(self.species):
                self.species_np[i] = sp.num
                self.species_masses[i] = sp.mass

            assert self.type.lower() == "berendsen", "Only Berendsen thermostat is supported."
        else:
            pass

    def update(self, vel, it):
        """
        Update particles' velocities according to the chosen thermostat

        Parameters
        ----------
        vel : ndarray
            Particles' velocities to be rescaled.

        it : int
            Current timestep.

        """
        K, T = calc_kin_temp(vel, self.species_np, self.species_masses, self.kB)
        thermostats.berendsen(vel, self.temperatures, T, self.species_np, self.relaxation_timestep,
                              self.relaxation_rate, it)


@nb.njit
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

    num_species = len(masses)

    K = np.zeros(num_species)
    T = np.zeros(num_species)

    species_start = 0
    for i in range(num_species):
        species_end = species_start + nums[i]
        K[i] = 0.5 * masses[i] * np.sum(vel[species_start:species_end, :] ** 2)
        T[i] = (2.0 / 3.0) * K[i] / kB / nums[i]
        species_start = species_end

    return K, T


@nb.njit
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
