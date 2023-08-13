"""
Module containing the basic class for handling particles properties.
"""

from copy import deepcopy
from numpy import arange, empty, floor, int64
from numpy import load as np_load
from numpy import loadtxt, meshgrid, ndarray, rint, sqrt, triu_indices, zeros
from numpy.random import Generator, PCG64
from os.path import join
from scipy.linalg import norm
from scipy.spatial.distance import pdist
from warnings import warn

from .utilities.exceptions import ParticlesError, ParticlesWarning


class Particles:
    """
    Class handling particles' properties.

    Attributes
    ----------
    kB : float
        Boltzmann constant.

    fourpie0: float
        Electrostatic constant :math:`4\\pi \\epsilon_0`.

    pos : numpy.ndarray
        Particles' positions.

    vel : numpy.ndarray
        Particles' velocities.

    acc : numpy.ndarray
        Particles' accelerations.

    box_lengths : numpy.ndarray
        Box sides' lengths.

    pbox_lengths : numpy.ndarray
        Initial particle box sides' lengths.

    masses : numpy.ndarray
        Mass of each particle. Shape = (attr:`sarkas.core.Parameters.total_num_ptcls`).

    charges : numpy.ndarray
        Charge of each particle. Shape = (attr:`sarkas.core.Parameters.total_num_ptcls`).

    id : numpy.ndarray,
        Species identifier. Shape = (attr:`sarkas.core.Parameters.total_num_ptcls`).

    names : numpy.ndarray
        Species' names. (attr:`sarkas.core.Parameters.total_num_ptcls`).

    rdf_nbins : int
        Number of bins for radial pair distribution.

    no_grs : int
        Number of independent :math:`g_{ij}(r)`.

    rdf_hist : numpy.ndarray
        Histogram array for the radial pair distribution function.

    prod_dump_dir : str
        Directory name where to store production phase's simulation's checkpoints. Default = 'dumps'.

    eq_dump_dir : str
        Directory name where to store equilibration phase's simulation's checkpoints. Default = 'dumps'.

    total_num_ptcls : int
        Total number of simulation's particles.

    num_species : int
        Number of species.

    species_num : numpy.ndarray
        Number of particles of each species. Shape = (attr:`sarkas.particles.Particles.num_species`).

    dimensions : int
        Number of non-zero dimensions. Default = 3.

    particle_potential_energy : float
        Instantaneous value of the potential energy of each particle. Note that the total potential energy requires the multiplication of the array's sum by 0.5 to avoid double counting.\n
        For example: `N` = 3, `particle_potential_energy[0] = U_12 + U_13` and `particle_potential_energy[1] = U_21 + U_23` and `particle_potential_energy[1] = U_31 + U_32`

    rnd_gen : numpy.random.Generator
        Random number generator.

    """

    def __init__(self):
        self.mag_dump_dir = None
        self.rdf_nbins = None
        self.kB = None
        self.fourpie0 = None
        self.prod_dump_dir = None
        self.eq_dump_dir = None
        self.box_lengths = None
        self.pbox_lengths = None
        self.total_num_ptcls = None
        self.num_species = 1
        self.species_num = None
        self.dimensions = None
        self.rnd_gen = None

        self.pos = None
        self.vel = None
        self.acc = None

        self.virial = None
        self.energy_current = None
        self.pbc_cntr = None

        self.names = None
        self.id = None

        self.species_initial_velocity = None
        self.species_thermal_velocity = None

        self.masses = None
        self.charges = None
        self.cyclotron_frequencies = None

        self.no_grs = None
        self.rdf_hist = None

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Particles( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def __copy__(self):
        """
        Make a shallow copy of the object using copy by creating a new instance of the object and copying its __dict__."""
        # Create a new object
        _copy = type(self)()
        # copy the dictionary
        _copy.__dict__.update(self.__dict__)
        return _copy

    def __deepcopy__(self, memodict: dict = {}):
        """Make a deepcopy of the object.

        Parameters
        ----------
        memodict: dict
            Dictionary of id's to copies

        Returns
        -------
        _copy: :class:`sarkas.particles.Particles`
            A new Particles class.
        """
        id_self = id(self)  # memorization avoids unnecessary recursion
        _copy = memodict.get(id_self)
        if _copy is None:

            # Make a shallow copy of all attributes
            _copy = type(self)()
            # Make a deepcopy of the mutable arrays using numpy copy function
            for k, v in self.__dict__.items():
                if isinstance(v, ndarray):
                    _copy.__dict__[k] = v.copy()
                else:
                    _copy.__dict__[k] = deepcopy(v, memodict)

        return _copy

    def __getstate__(self):
        """Copy the object's state from self.__dict__ which contains all our instance attributes.
        Always use the dict.copy() method to avoid modifying the original state.
        Reference: https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        """

        state = self.__dict__.copy()
        # Remove the data that is stored already
        del state["pos"]
        del state["vel"]
        del state["acc"]
        del state["id"]
        del state["names"]
        del state["pbc_cntr"]
        del state["rdf_hist"]
        del state["virial"]

        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        # Initialize arrays
        self.pos = zeros((self.__dict__["total_num_ptcls"], 3))
        self.vel = zeros((self.__dict__["total_num_ptcls"], 3))
        self.acc = zeros((self.__dict__["total_num_ptcls"], 3))
        self.id = zeros(self.__dict__["total_num_ptcls"])
        self.names = zeros(self.__dict__["total_num_ptcls"])
        self.pbc_cntr = zeros((self.__dict__["total_num_ptcls"], 3))
        self.rdf_hist = zeros((self.__dict__["rdf_nbins"], self.__dict__["num_species"], self.__dict__["num_species"]))
        self.virial = zeros((self.__dict__["dimensions"], self.__dict__["dimensions"], self.__dict__["total_num_ptcls"]))

    def copy_params(self, params):
        """
        Copy necessary parameters.

        Parameters
        ----------
        params: :class:`sarkas.core.Parameters`
            Simulation's parameters.

        """

        self.kB = params.kB
        self.fourpie0 = params.fourpie0
        self.prod_dump_dir = params.prod_dump_dir
        self.eq_dump_dir = params.eq_dump_dir
        self.box_lengths = params.box_lengths.copy()
        self.pbox_lengths = params.pbox_lengths.copy()
        self.total_num_ptcls = params.total_num_ptcls
        self.total_num_density = params.total_num_density
        self.num_species = params.num_species
        self.species_num = params.species_num.copy()
        self.dimensions = params.dimensions
        self.box_volume = params.box_volume
        self.pbox_volume = params.pbox_volume
        self.load_method = params.load_method
        self.restart_step = params.restart_step
        self.particles_input_file = params.particles_input_file
        self.load_perturb = params.load_perturb
        self.load_rejection_radius = params.load_rejection_radius
        self.load_halton_bases = params.load_halton_bases

        if hasattr(params, "np_per_side"):
            self.np_per_side = params.np_per_side

        if hasattr(params, "initial_lattice_config"):
            self.lattice_type = params.initial_lattice_config

        if hasattr(params, "load_gauss_sigma"):
            self.load_gauss_sigma = params.load_gauss_sigma.copy()

        self.species_names = params.species_names

        if hasattr(params, "rdf_nbins"):
            self.rdf_nbins = params.rdf_nbins
        else:
            # nbins = 5% of the number of particles.
            self.rdf_nbins = int64(0.05 * params.total_num_ptcls)
            params.rdf_nbins = self.rdf_nbins

    def gaussian(self, mean, sigma, size):
        """
        Initialize particles' velocities according to a normalized Maxwell-Boltzmann (Normal) distribution.
        It calls ``numpy.random.Generator.normal``

        Parameters
        ----------
        size : tuple
            Size of the array to initialize. (no. of particles, dimensions).

        mean : float
            Center of the normal distribution.

        sigma : float
            Scale of the normal distribution.

        Returns
        -------
         : numpy.ndarray
            Particles property distributed according to a Normal probability density function.

        """
        return self.rnd_gen.normal(mean, sigma, size)

    def halton_reject(self, bases, r_reject):
        """
        Place particles according to a Halton sequence from 0 to LP (the initial particle box length)
        and uses a rejection radius to avoid placing particles to close to each other.

        Parameters
        ----------
        bases : numpy.ndarray
            Array of 3 ints each of which is a base for the Halton sequence.
            Defualt: bases = array([2,3,5])

        r_reject : float
            Value of rejection radius.

        """

        # Get bases
        b1, b2, b3 = bases

        # Allocate space and store first value from Halton
        x = zeros(self.total_num_ptcls)
        y = zeros(self.total_num_ptcls)
        z = zeros(self.total_num_ptcls)

        # Initialize particle counter and Halton counter
        i = 1
        k = 1

        # Loop over all particles
        while i < self.total_num_ptcls:

            # Increment particle counter
            n = k
            m = k
            p = k

            # Determine x coordinate
            f1 = 1
            r1 = 0
            while n > 0:
                f1 /= b1
                r1 += f1 * (n % int(b1))
                n = floor(n / b1)
            x_new = self.pbox_lengths[0] * r1  # new x value

            # Determine y coordinate
            f2 = 1
            r2 = 0
            while m > 0:
                f2 /= b2
                r2 += f2 * (m % int(b2))
                m = floor(m / b2)
            y_new = self.pbox_lengths[1] * r2  # new y value

            # Determine z coordinate
            f3 = 1
            r3 = 0
            while p > 0:
                f3 /= b3
                r3 += f3 * (p % int(b3))
                p = floor(p / b3)
            z_new = self.pbox_lengths[2] * r3  # new z value

            # Check if particle was place too close relative to all other current particles
            for j in range(len(x)):

                # Flag for if particle is outside of cutoff radius (1 -> not inside rejection radius)
                flag = 1

                # Compute distance b/t particles for initial placement
                x_diff = x_new - x[j]
                y_diff = y_new - y[j]
                z_diff = z_new - z[j]

                # Periodic condition applied for minimum image
                if x_diff < -self.pbox_lengths[0] / 2:
                    x_diff = x_diff + self.pbox_lengths[0]
                if x_diff > self.pbox_lengths[0] / 2:
                    x_diff = x_diff - self.pbox_lengths[0]

                if y_diff < -self.pbox_lengths[1] / 2:
                    y_diff = y_diff + self.pbox_lengths[1]
                if y_diff > self.pbox_lengths[1] / 2:
                    y_diff = y_diff - self.pbox_lengths[1]

                if z_diff < -self.pbox_lengths[2] / 2:
                    z_diff = z_diff + self.pbox_lengths[2]
                if z_diff > self.pbox_lengths[2] / 2:
                    z_diff = z_diff - self.pbox_lengths[2]

                # Compute distance
                r = sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                # Check if new particle is below rejection radius. If not, break out and try again
                if r <= r_reject:
                    k += 1  # Increment Halton counter
                    flag = 0  # New position not added (0 -> no longer outside reject r)
                    break

            # If flag true add new position
            if flag == 1:
                # Add new positions to arrays
                x[i] = x_new
                y[i] = y_new
                z[i] = z_new

                k += 1  # Increment Halton counter
                i += 1  # Increment particle number

        self.pos[:, 0] = x + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
        self.pos[:, 1] = y + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
        self.pos[:, 2] = z + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

    def initialize_accelerations(self):
        """
        Initialize particles' accelerations.
        """
        self.acc = zeros((self.total_num_ptcls, 3))

    def initialize_arrays(self):
        """Initialize the needed arrays"""
        self.pos = zeros((self.total_num_ptcls, 3))
        self.vel = zeros((self.total_num_ptcls, 3))
        self.acc = zeros((self.total_num_ptcls, 3))

        self.pbc_cntr = zeros((self.total_num_ptcls, 3))
        self.virial = zeros((3, 3, self.total_num_ptcls))
        self.energy_current = zeros((3, self.total_num_ptcls))
        self.names = empty(self.total_num_ptcls, dtype=self.species_names.dtype)
        self.id = zeros(self.total_num_ptcls, dtype=int64)

        self.species_initial_velocity = zeros((self.num_species, 3))
        self.species_thermal_velocity = zeros((self.num_species, 3))

        self.masses = zeros(self.total_num_ptcls)  # mass of each particle
        self.charges = zeros(self.total_num_ptcls)  # charge of each particle
        self.cyclotron_frequencies = zeros(self.total_num_ptcls)

        # No. of independent rdf
        self.no_grs = int64(self.num_species * (self.num_species + 1) / 2)

        self.rdf_hist = zeros((self.rdf_nbins, self.num_species, self.num_species))

        self.particle_potential_energy = zeros(self.total_num_ptcls)

    def initialize_positions(self):
        """
        Initialize particles' positions based on the load method.
        """

        # position distribution.
        if self.load_method == "lattice":
            self.lattice(self.load_perturb)

        elif self.load_method == "random_reject":
            # check
            if not hasattr(self, "load_rejection_radius"):
                raise AttributeError("Rejection radius not defined. " "Please define Parameters.load_rejection_radius.")
            self.random_reject(self.load_rejection_radius)

        elif self.load_method == "halton_reject":
            # check
            if not hasattr(self, "load_rejection_radius"):
                raise AttributeError("Rejection radius not defined. " "Please define Parameters.load_rejection_radius.")
            self.halton_reject(self.load_halton_bases, self.load_rejection_radius)

        elif self.load_method in ["uniform", "random_no_reject"]:
            self.pos = self.uniform_no_reject(
                0.5 * self.box_lengths - 0.5 * self.pbox_lengths, 0.5 * self.box_lengths + 0.5 * self.pbox_lengths
            )

        elif self.load_method == "gaussian":
            sp_start = 0
            sp_end = 0
            for sp, sp_num in enumerate(self.species_num):
                sp_end += sp_num
                self.pos[sp_start:sp_end, :] = self.gaussian(
                    self.box_lengths[0] / 2.0, self.load_gauss_sigma[sp], (sp_num, 3)
                )
                sp_start += sp_num
        else:
            raise AttributeError("Incorrect particle placement scheme specified.")

    def initialize_velocities(self, species):
        """
        Initialize particles' velocities based on the species input values. The velocities can be initialized from a
        Maxwell-Boltzmann distribution or from a monochromatic distribution.

        Parameters
        ----------
        species: list
            List of :class:`sarkas.core.Spcies`.

        """
        species_end = 0
        species_start = 0
        for ic, sp in enumerate(species):
            if sp.name != "electron_background":
                species_end += sp.num
                self.species_initial_velocity[ic, :] = sp.initial_velocity

                if sp.initial_velocity_distribution == "boltzmann":
                    if isinstance(sp.temperature, (int, float)):
                        sp_temperature = zeros(3)
                        for d in range(self.dimensions):
                            sp_temperature[d] = sp.temperature

                    self.species_thermal_velocity[ic] = sqrt(self.dimensions * self.kB * sp_temperature / (2.0 * sp.mass))
                    # Note gaussian(0.0, 0.0, N) = array of zeros
                    self.vel[species_start:species_end, :] = self.gaussian(
                        sp.initial_velocity, self.species_thermal_velocity[ic], (sp.num, 3)
                    )

                elif sp.initial_velocity_distribution == "monochromatic":
                    vrms = sqrt(self.dimensions * self.kB * sp.temperature / sp.mass)
                    self.vel[species_start:species_end, :] = vrms * self.random_unit_vectors(sp.num, self.dimensions)

                species_start += sp.num

    def kinetic_temperature(self):
        """
        Calculate the kinetic energy and temperature of each species.

        Returns
        -------
        K : numpy.ndarray
            Kinetic energy of each species. Shape=(``num_species``).

        T : numpy.ndarray
            Temperature of each species. Shape=(``num_species``).

        """
        K = zeros(self.num_species)
        T = zeros(self.num_species)
        const = 2.0 / (self.kB * self.species_num * self.dimensions)
        kinetic = 0.5 * self.masses * (self.vel * self.vel).transpose()

        species_start = 0
        species_end = 0
        for i, num in enumerate(self.species_num):
            species_end += num
            K[i] = kinetic[:, species_start:species_end].sum()
            T[i] = const[i] * K[i]
            species_start = species_end

        return K, T

    def lattice(self, perturb: float = 0.05):
        """
        Place particles in a simple cubic lattice with a slight perturbation ranging
        from 0 to 0.5 times the lattice spacing.

        Parameters
        ----------
        perturb : float
            Value of perturbation, p, such that 0 <= p <= 0.5. Default = 0.05

        """

        # Check if perturbation is below maximum allowed. If not, default to maximum perturbation.
        if perturb > 0.5:
            warn("Random perturbation must not exceed 0.5. Setting perturb = 0.5", category=ParticlesWarning)
            perturb = 0.5

        if self.lattice_type == "simple_cubic":
            # Determining number of particles per side of simple cubic lattice
            part_per_side = self.total_num_ptcls ** (1.0 / 3.0)  # Number of particles per side of cubic lattice

            # Check if total number of particles is a perfect cube, if not, place more than the requested amount
            if round(part_per_side) ** 3 != self.total_num_ptcls:
                part_per_side = rint(self.total_num_ptcls ** (1.0 / 3.0))
                raise ParticlesError(
                    f"N = {self.total_num_ptcls} cannot be placed in a simple cubic lattice. "
                    f"Use {int(part_per_side ** 3)} particles instead."
                )

            dx_lattice = self.pbox_lengths[0] / (self.total_num_ptcls ** (1.0 / 3.0))  # Lattice spacing
            dy_lattice = self.pbox_lengths[1] / (self.total_num_ptcls ** (1.0 / 3.0))  # Lattice spacing
            dz_lattice = self.pbox_lengths[2] / (self.total_num_ptcls ** (1.0 / 3.0))  # Lattice spacing

            # Create x, y, and z position arrays
            x = arange(0, self.pbox_lengths[0], dx_lattice) + 0.5 * dx_lattice
            y = arange(0, self.pbox_lengths[1], dy_lattice) + 0.5 * dy_lattice
            z = arange(0, self.pbox_lengths[2], dz_lattice) + 0.5 * dz_lattice

            # Create a lattice with appropriate x, y, and z values based on arange
            X, Y, Z = meshgrid(x, y, z)

            # Perturb lattice
            X += self.rnd_gen.uniform(-0.5, 0.5, X.shape) * perturb * dx_lattice
            Y += self.rnd_gen.uniform(-0.5, 0.5, Y.shape) * perturb * dy_lattice
            Z += self.rnd_gen.uniform(-0.5, 0.5, Z.shape) * perturb * dz_lattice

            # Flatten the meshgrid values for plotting and computation
            self.pos[:, 0] = X.ravel() + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
            self.pos[:, 1] = Y.ravel() + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
            self.pos[:, 2] = Z.ravel() + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

        elif self.lattice_type == "bcc":
            # Determining number of particles per side of simple cubic lattice
            part_per_side = int(0.5 * self.total_num_ptcls) ** (
                1.0 / 3.0
            )  # Number of particles per side of cubic lattice

            # Check if total number of particles is a perfect cube, if not, place more than the requested amount
            if round(part_per_side) ** 3 != int(0.5 * self.total_num_ptcls):
                part_per_side = rint((0.5 * self.total_num_ptcls) ** (1.0 / 3.0))
                raise ParticlesError(
                    f"N = {self.total_num_ptcls} cannot be placed in a bcc lattice. "
                    f"Use {int(2.0*part_per_side ** 3)} particles instead."
                )

            dx_lattice = self.pbox_lengths[0] / (0.5 * self.total_num_ptcls) ** (1.0 / 3.0)  # Lattice spacing
            dy_lattice = self.pbox_lengths[1] / (0.5 * self.total_num_ptcls) ** (1.0 / 3.0)  # Lattice spacing
            dz_lattice = self.pbox_lengths[2] / (0.5 * self.total_num_ptcls) ** (1.0 / 3.0)  # Lattice spacing

            # Create x, y, and z position arrays
            x = arange(0, self.pbox_lengths[0], dx_lattice) + 0.5 * dx_lattice
            y = arange(0, self.pbox_lengths[1], dy_lattice) + 0.5 * dy_lattice
            z = arange(0, self.pbox_lengths[2], dz_lattice) + 0.5 * dz_lattice

            # Create a lattice with appropriate x, y, and z values based on arange
            X, Y, Z = meshgrid(x, y, z)

            # Perturb lattice
            X += self.rnd_gen.uniform(-0.5, 0.5, X.shape) * perturb * dx_lattice
            Y += self.rnd_gen.uniform(-0.5, 0.5, Y.shape) * perturb * dy_lattice
            Z += self.rnd_gen.uniform(-0.5, 0.5, Z.shape) * perturb * dz_lattice

            half_Np = int(self.total_num_ptcls / 2)
            # Flatten the meshgrid values for plotting and computation
            self.pos[:half_Np, 0] = X.ravel() + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
            self.pos[:half_Np, 1] = Y.ravel() + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
            self.pos[:half_Np, 2] = Z.ravel() + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

            self.pos[half_Np:, 0] = X.ravel() + 0.5 * dx_lattice + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
            self.pos[half_Np:, 1] = Y.ravel() + 0.5 * dy_lattice + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
            self.pos[half_Np:, 2] = Z.ravel() + 0.5 * dz_lattice + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

        elif self.lattice_type in ["square", "tetragonal_2D"]:
            # Determining number of particles per side of simple cubic lattice
            part_per_side = rint(sqrt(self.total_num_ptcls))  # Number of particles per side of a square lattice

            # Check if total number of particles is a perfect cube, if not, place more than the requested amount
            if part_per_side**2 != self.total_num_ptcls:
                raise ParticlesError(
                    f"N = {self.total_num_ptcls} cannot be placed in a square lattice. "
                    f"Use {int(part_per_side ** 2)} particles instead."
                )

            dx_lattice = self.pbox_lengths[0] / sqrt(self.total_num_ptcls)  # Lattice spacing
            dy_lattice = self.pbox_lengths[1] / sqrt(self.total_num_ptcls)  # Lattice spacing

            # Create x, y, and z position arrays
            x = arange(0, self.pbox_lengths[0], dx_lattice) + 0.5 * dx_lattice
            y = arange(0, self.pbox_lengths[1], dy_lattice) + 0.5 * dy_lattice

            # Create a lattice with appropriate x, y, and z values based on arange
            X, Y = meshgrid(x, y)

            # Perturb lattice
            X += self.rnd_gen.uniform(-0.5, 0.5, X.shape) * perturb * dx_lattice
            Y += self.rnd_gen.uniform(-0.5, 0.5, Y.shape) * perturb * dy_lattice

            # Flatten the meshgrid values for plotting and computation
            self.pos[:, 0] = X.ravel() + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
            self.pos[:, 1] = Y.ravel() + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
            self.pos[:, 2] = 0.0

        elif self.lattice_type in ["hexagonal", "triangular"]:

            # Determining number of particles per side of simple cubic lattice
            part_per_side = round(sqrt(self.total_num_ptcls))  # Number of particles per side of cubic lattice

            # Check if total number of particles is a perfect cube, if not, place more than the requested amount
            if self.np_per_side[:2].prod() != part_per_side * (part_per_side + 1):
                raise ParticlesError(
                    f"N = {self.total_num_ptcls} cannot be placed in an hexagonal lattice. "
                    f"Use Nx = {part_per_side} and Ny = {part_per_side + 1} particles instead."
                )

            dx_lattice = self.pbox_lengths[0] / (self.np_per_side[0])  # Lattice spacing
            dy_lattice = self.pbox_lengths[1] / (self.np_per_side[1])  # Lattice spacing

            if self.np_per_side[0] > self.np_per_side[1]:
                # Create x, y, and z position arrays
                x = arange(0, self.pbox_lengths[0], dx_lattice)
                y = arange(0, self.pbox_lengths[1], dy_lattice) + 0.5 * dy_lattice

                # Create a lattice with appropriate x, y, and z values based on arange
                X, Y = meshgrid(x, y)
                # Shift the Y axis of every other row of particles
                X[:, ::2] += 0.5 * dx_lattice

            else:
                # Create x, y, and z position arrays
                x = arange(0, self.pbox_lengths[0], dx_lattice) + 0.5 * dx_lattice
                y = arange(0, self.pbox_lengths[1], dy_lattice)

                # Create a lattice with appropriate x, y, and z values based on arange
                X, Y = meshgrid(x, y)
                # Shift the Y axis of every other row of particles
                Y[:, ::2] += 0.5 * dy_lattice

            # Perturb lattice
            X += self.rnd_gen.uniform(-0.5, 0.5, X.shape) * perturb * dx_lattice
            Y += self.rnd_gen.uniform(-0.5, 0.5, Y.shape) * perturb * dy_lattice

            # Flatten the meshgrid values for plotting and computation
            self.pos[:, 0] = X.ravel() + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
            self.pos[:, 1] = Y.ravel() + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
            self.pos[:, 2] = 0.0

    def load(self):
        """
        Initialize particles' positions and velocities.
        Positions are initialized based on the load method while velocities are chosen
        from a Maxwell-Boltzmann distribution.

        """

        warn(
            "Deprecated feature. It will be removed in the v2.0.0 release. \n"
            "Use parameters.calc_electron_properties(species). You need to pass the species list.",
            category=DeprecationWarning,
        )

        self.initialize_positions()

    def load_from_file(self, f_name):
        """
        Load particles' data from a specific file.

        Parameters
        ----------
        f_name : str
            Filename
        """
        pv_data = loadtxt(f_name)
        if not (pv_data.shape[0] == self.total_num_ptcls):
            msg = (
                f"Number of particles is not same between input file and initial p & v data file. \n "
                f"Input file: N = {self.total_num_ptcls}, load data: N = {pv_data.shape[0]}"
            )
            raise ParticlesError(msg)

        self.pos[:, 0] = pv_data[:, 0]
        self.pos[:, 1] = pv_data[:, 1]
        self.pos[:, 2] = pv_data[:, 2]

        self.vel[:, 0] = pv_data[:, 3]
        self.vel[:, 1] = pv_data[:, 4]
        self.vel[:, 2] = pv_data[:, 5]

    def load_from_npz(self, file_name):
        """
        Load particles' data from an .npz data file.

        Parameters
        ----------
        file_name : str
            Path to file.

        """
        # file_name = join(self.eq_dump_dir, "checkpoint_" + str(it) + ".npz")
        data = np_load(file_name, allow_pickle=True)
        if not data["pos"].shape[0] == self.total_num_ptcls:
            msg = (
                f"Number of particles is not same between input file and particles data file. \n "
                f"Input file: N = {self.total_num_ptcls}, particles data file: N = {data['pos'].shape[0]}"
            )
            raise ParticlesError(msg)

        self.id = data["id"]
        self.names = data["names"]
        self.pos = data["pos"]
        self.vel = data["vel"]
        self.acc = data["acc"]

    def load_from_restart(self, phase, it):
        """
        Initialize particles' data from a checkpoint of a previous run.

        Raises
        ------
            : DeprecationWarning
        """

        warn(
            "Deprecated feature. It will be removed in a future release. \n" "Use load_from_checkpoint. ",
            category=DeprecationWarning,
        )

        self.load_from_checkpoint(phase, it)

    def load_from_checkpoint(self, phase, it):
        """
        Load particles' data from a checkpoint of a previous run

        Parameters
        ----------
        it : int
            Timestep.

        phase: str
            Restart phase.

        """
        if phase == "equilibration":
            file_name = join(self.eq_dump_dir, "checkpoint_" + str(it) + ".npz")
            data = np_load(file_name, allow_pickle=True)
            self.id = data["id"]
            self.names = data["names"]
            self.pos = data["pos"]
            self.vel = data["vel"]
            self.acc = data["acc"]

        elif phase == "production":
            file_name = join(self.prod_dump_dir, "checkpoint_" + str(it) + ".npz")
            data = np_load(file_name, allow_pickle=True)
            self.id = data["id"]
            self.names = data["names"]
            self.pos = data["pos"]
            self.vel = data["vel"]
            self.acc = data["acc"]
            self.pbc_cntr = data["cntr"]
            self.rdf_hist = data["rdf_hist"]

        elif phase == "magnetization":
            file_name = join(self.mag_dump_dir, "checkpoint_" + str(it) + ".npz")
            data = np_load(file_name, allow_pickle=True)
            self.id = data["id"]
            self.names = data["names"]
            self.pos = data["pos"]
            self.vel = data["vel"]
            self.acc = data["acc"]
            self.pbc_cntr = data["cntr"]
            self.rdf_hist = data["rdf_hist"]

    def potential_energy(self):
        """Calculate the total potential energy of the particles by summing the potential energy of each individual particle contained in attr:`particle_potential_energy`

        Return
        ------
        pot : float
            Total potential energy.

        """
        pot = self.particle_potential_energy.sum()  #
        return pot

    def potential_energies(self):
        """
        Calculate the potential energies of each species.

        Returns
        -------
        P : numpy.ndarray
            Potential energy of each species. Shape=(``num_species``).

        """
        P = zeros(self.num_species)

        species_start = 0
        species_end = 0
        for i, num in enumerate(self.species_num):
            species_end += num

            P[i] = self.particle_potential_energy[species_start:species_end].sum()

            species_start = species_end

        return P

    def pressure_tensors_species(self):
        """Calculate the pressure tensor of each species. """
        p, p_kin_tensor, p_pot_tensor = calc_pressure_tensor(self.vel, self.virial, self.masses, self.species_num, self.box_volume, self.dimensions)
        
        return p, p_kin_tensor, p_pot_tensor
    
    def random_reject(self, r_reject):
        """
        Place particles by sampling a uniform distribution from 0 to LP (the initial particle box length)
        and uses a rejection radius to avoid placing particles to close to each other.

        Parameters
        ----------
        r_reject : float
            Value of rejection radius.
        """

        # Initialize Arrays
        x = zeros(self.total_num_ptcls)
        y = zeros(self.total_num_ptcls)
        z = zeros(self.total_num_ptcls)

        # Set first x, y, and z positions
        x_new = self.rnd_gen.uniform(0, self.pbox_lengths[0])
        y_new = self.rnd_gen.uniform(0, self.pbox_lengths[1])
        z_new = self.rnd_gen.uniform(0, self.pbox_lengths[2])

        # Append to arrays
        x[0] = x_new
        y[0] = y_new
        z[0] = z_new

        # Particle counter
        i = 1

        cntr_reject = 0
        cntr_total = 0
        # Loop to place particles
        while i < self.total_num_ptcls:

            # Set x, y, and z positions
            x_new = self.rnd_gen.uniform(0.0, self.pbox_lengths[0])
            y_new = self.rnd_gen.uniform(0.0, self.pbox_lengths[1])
            z_new = self.rnd_gen.uniform(0.0, self.pbox_lengths[2])

            # Check if particle was place too close relative to all other current particles
            for j in range(len(x)):

                # Flag for if particle is outside of cutoff radius (True -> not inside rejection radius)
                flag = 1

                # Compute distance b/t particles for initial placement
                x_diff = x_new - x[j]
                y_diff = y_new - y[j]
                z_diff = z_new - z[j]

                # periodic condition applied for minimum image
                if x_diff < -self.pbox_lengths[0] / 2:
                    x_diff += self.pbox_lengths[0]
                if x_diff > self.pbox_lengths[0] / 2:
                    x_diff -= self.pbox_lengths[0]

                if y_diff < -self.pbox_lengths[1] / 2:
                    y_diff += self.pbox_lengths[1]
                if y_diff > self.pbox_lengths[1] / 2:
                    y_diff -= self.pbox_lengths[1]

                if z_diff < -self.pbox_lengths[2] / 2:
                    z_diff += self.pbox_lengths[2]
                if z_diff > self.pbox_lengths[2] / 2:
                    z_diff -= self.pbox_lengths[2]

                # Compute distance
                r = sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                # Check if new particle is below rejection radius. If not, break out and try again
                if r <= r_reject:
                    flag = 0  # new position not added (False -> no longer outside reject r)
                    cntr_reject += 1
                    cntr_total += 1
                    break

            # If flag true add new position
            if flag == 1:
                x[i] = x_new
                y[i] = y_new
                z[i] = z_new

                # Increment particle number
                i += 1
                cntr_total += 1

        self.pos[:, 0] = x + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
        self.pos[:, 1] = y + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
        self.pos[:, 2] = z + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

    def random_unit_vectors(self, num_ptcls, dimensions):
        """
        Initialize random unit vectors for particles' velocities (e.g. for monochromatic energies but random velocities)
        It calls ``numpy.random.Generator.normal``

        Parameters
        ----------
        num_ptcls : int
            Number of particles to initialize.

        dimensions : int
            Number of non-zero dimensions.

        Returns
        -------
        uvec : numpy.ndarray
            Random unit vectors of specified dimensions for all particles

        """

        uvec = self.rnd_gen.normal(size=(num_ptcls, dimensions))
        # Broadcasting
        uvec /= norm(uvec, axis=1).reshape(num_ptcls, 1)

        return uvec

    def remove_drift(self):
        """
        Enforce conservation of total linear momentum. Updates particles velocities
        """
        species_start = 0
        species_end = 0
        for ic, nums in enumerate(self.species_num):
            species_end += nums
            self.vel[species_start:species_end, :] -= self.vel[species_start:species_end, :].mean(axis=0)
            species_start += nums

    def setup(self, params, species):
        """
        Initialize class' attributes

        Parameters
        ----------
        params: :class:`sarkas.core.Parameters`
            Simulation's parameters.

        species : list
            List of :meth:`sarkas.plasma.Species` objects.

        """

        if hasattr(params, "rand_seed"):
            self.rand_seed = params.rand_seed
            self.rnd_gen = Generator(PCG64(params.rand_seed))
        else:
            self.rnd_gen = Generator(PCG64())

        self.copy_params(params)
        self.initialize_arrays()
        self.update_attributes(species)
        # Particles Position Initialization
        if self.load_method in [
            "equilibration_restart",
            "eq_restart",
            "magnetization_restart",
            "mag_restart",
            "production_restart",
            "prod_restart",
        ]:
            # checks
            if self.restart_step is None:
                raise AttributeError("Restart step not defined." "Please define Parameters.restart_step.")

            if type(self.restart_step) is not int:
                self.restart_step = int(self.restart_step)

            if self.load_method[:2] == "eq":
                self.load_from_restart("equilibration", self.restart_step)
            elif self.load_method[:2] == "pr":
                self.load_from_restart("production", self.restart_step)
            elif self.load_method[:2] == "ma":
                self.load_from_restart("magnetization", self.restart_step)

        elif self.load_method == "file":
            # check
            if not hasattr(self, "particles_input_file"):
                raise AttributeError("Input file not defined." "Please define Parameters.particles_input_file.")

            if self.particles_input_file[-3:] == "npz":
                self.load_from_npz(self.particles_input_file)
            else:
                self.load_from_file(self.particles_input_file)
        else:
            self.initialize_positions()
            self.initialize_velocities(species)
            self.initialize_accelerations()
    
    def thermodynamics(self):
        """Calculate thermodynamics quantities from particles data.
        
        Return
        ------
        
        data : dict
            Thermodynamics data. In case of multiple species, it returns thermodynamics quantities per species. 
            keys = [`Total Energy`, `Total Kinetic Energy`, `Total Potential Energy`, `Total Temperature`, `Total Pressure`,
            `Ideal Pressure`, `Excess Pressure, `Enthalpy`] 
        """
        kinetic_energies, temperatures = self.kinetic_temperature()
        potential_energies = self.potential_energies()
        tot_pot = self.potential_energy()
        tot_kin = kinetic_energies.sum()
        sp_pressure, sp_p_kin_tensor, sp_p_pot_tensor = self.pressure_tensors_species()
        tot_enthalpy = tot_kin + tot_pot + sp_pressure.sum() * self.box_volume
        # Save Energy data
        data = {
            "Total Energy": tot_kin + tot_pot,
            "Total Kinetic Energy": tot_kin,
            "Total Potential Energy": tot_pot,
            "Total Temperature": self.species_num.transpose() @ temperatures / self.total_num_ptcls,
            "Total Pressure": sp_pressure.sum(),
            "Ideal Pressure": sp_p_kin_tensor.sum(axis = -1).trace()/self.dimensions,
            "Excess Pressure": sp_p_pot_tensor.sum(axis = -1).trace()/self.dimensions,
            "Enthalpy": tot_enthalpy
        }

        if len(temperatures) > 1:
            for sp, (temp, kin, pot) in enumerate(zip(temperatures, kinetic_energies, potential_energies)):
                data[f"{self.species_names[sp]} Kinetic Energy"] = kin
                data[f"{self.species_names[sp]} Potential Energy"] = pot
                data[f"{self.species_names[sp]} Temperature"] = temp
                data[f"{self.species_names[sp]} Total Pressure"] = sp_pressure[sp]
                data[f"{self.species_names[sp]} Ideal Pressure"] = sp_p_kin_tensor[:,:,sp].trace()/self.dimensions
                data[f"{self.species_names[sp]} Excess Pressure"] = sp_p_pot_tensor[:,:,sp].trace()/self.dimensions
                data[f"{self.species_names[sp]} Enthalpy"] = kin + pot + sp_pressure[sp]*self.box_volume

        return data
    
    def uniform_no_reject(self, mins, maxs):
        """
        Randomly distribute particles along each direction.

        Parameters
        ----------
        mins : float
            Minimum value of the range of a uniform distribution.

        maxs : float
            Maximum value of the range of a uniform distribution.

        Returns
        -------
         : numpy.ndarray
            Particles' property, e.g. pos, vel. Shape = (``total_num_ptcls``, 3).

        """

        return self.rnd_gen.uniform(mins, maxs, (self.total_num_ptcls, 3))

    def update_attributes(self, species):
        """
        Assign particles attributes.

        Parameters
        ----------
        species : list
            List of :class:`sarkas.plasma.Species` objects.

        """
        species_end = 0
        species_start = 0

        for ic, sp in enumerate(species):
            if sp.name != "electron_background":
                species_end += sp.num

                self.names[species_start:species_end] = sp.name
                self.masses[species_start:species_end] = sp.mass

                if hasattr(sp, "charge"):
                    self.charges[species_start:species_end] = sp.charge
                else:
                    self.charges[species_start:species_end] = 1.0

                if hasattr(sp, "cyclotron_frequency"):
                    self.cyclotron_frequencies[species_start:species_end] = sp.cyclotron_frequency

                self.id[species_start:species_end] = ic
                species_start += sp.num


def calc_pressure_tensor(vel, virial, masses, species_np, box_volume, dimensions):
    """
    Calculate the pressure tensor.

    Parameters
    ----------
    vel : numpy.ndarray
        Particles' velocities.

    virial : numpy.ndarray
        Virial tensor of each particle. Shape= (3, 3, ``total_num_ptcls``).
        Note that the size of the first two axis is 3 even if the system is 2D. 

    masses : numpy.ndarray
        Mass of each species. Shape = (``num_species``)

    species_np : numpy.ndarray
        Number of particles of each species.

    box_volume : float
        Volume of simulation's box.

    dimensions : int
        Number of dimensions.

    Returns
    -------
    pressure : float
        Scalar Pressure i.e. trace of the pressure tensor

    pressure_kin : numpy.ndarray
        Kinetic part of the Pressure tensor. Shape(``no_dim``,``no_dim``, ``num_species``)

    pressure_pot : numpy.ndarray
        Potential energy part of the Pressure tensor. Shape(``no_dim``,``no_dim``, `num_species`)

    """
    sp_start = 0
    sp_end = 0

    # Rescale vel of each particle by their individual mass
    pressure_kin = zeros( (3,3, species_np))
    pressure_pot = zeros( (3,3, species_np))

    for sp, num in enumerate(species_np):
        sp_end += num
        vel[sp_start:sp_end, :] *= sqrt(masses[sp_start:sp_end])
        pressure_kin[:,:,sp] = (vel[sp_start:sp_end, :].transpose() @ vel[sp_start:sp_end,:]) / box_volume
        pressure_pot[:,:,sp] = virial[:,:,sp_start:sp_end].sum(axis = -1) / box_volume
        # .trace is not supported by numba. hence no njit
        # Pressure of each species
        pressure = (pressure_kin[:,:,sp] + pressure_pot[:,:,sp] ).trace() / dimensions
        sp_start += num

    # Calculate the total pressure tensor
    # pressure_tensor = (pressure_kin + pressure_pot).sum(axis = -1)

    return pressure, pressure_kin, pressure_pot
