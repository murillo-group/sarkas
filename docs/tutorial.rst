.. _tutorial:
========
Tutorial
========

Here we provide a step-by-step guide to illustrate how to run Sarkas and how to obtain useful quantities
from your simulation data once you run is complete. The topics that are listed in this tutorial are as follows

- Input file
- Testing
- Running a simulation
- Post Processing

We will illustrate these steps using the example file ``yukawa_mks_p3m.yaml`` in the ``examples`` folder.
This file contains parameters for a :math:`NVE` simulation of a One component strongly coupled plasma whose
particles interact via a Yukawa potential, i.e. an YOCP.

Input File
========

Particles
---------
The first step in any MD simulation is the creation of an input file containing all the relevant parameters
of our simulation. We start by considering the physical parameter of our system. Assume we want to simulate
a strongly coupled plasma comprised of :math:`N = 10\, 000` hydrogen ions with
a number density of :math:`n = 1.6e+30 N/m^3` at a temperature :math:`0.5 eV`.
These parameters are defined in our ``yaml`` file by the block ``Particles`` and its attribute ``species``,

.. code-block:: python

    Particles:
        - species:
            name: H
            number_density: 1.62e+30    # /m^3
            mass: 1.673e-27             # kg
            num: 10000                  # total number of particles of ion1
            Z: 1.0                      # degree of ionization
            temperature_eV: 0.5
            # temperature: 5802         # Kelvin

        - load:
            method: random_no_reject    # loading method
            rand_seed: 123456789        # random seed

It is very important to maintain the syntax as shown here. As you can see the first attribute of ``Particles``
defines the particles species and its physical attributes. In the case of a multi-component plasma we need only add
another ``species`` attribute with its corresponding physical parameters, see ``ybim_mks_p3m.yaml``. The attributes of
``species`` take only numerical value in the correct choice of units which is defined in the block ``Control``,
see below. Notice that in this section we also define the mass of the particles, ``mass``, and their charge ``Z``.
Future development of Sarkas are aiming in automatically calculate the degree of ionization given by the density and
temperature of the system, but for now we need to define it. The parameters given here are not the only options,
more information of all the possible inputs can be found in the page ``input file``.

The next attribute of ``Particle`` defines how particles should be initialized for our simulation. In this case
we chose to initialize them from an uniform random distribution, without a rejection radius, as specified in the attribute
``method`` using a random seed defined by ``rand_seed``.

Interaction
-----------
The next section of the input file defines our interaction potential's parameters

.. code-block:: python

    Potential:
        - type: Yukawa
        - method: P3M            # Particle-Particle Particle-Mesh
        - kappa: 0.5
        - rc: 2.79946255e-10     # [m]


The instance ``type`` defines the interaction potential. Currently Sarkas supports the following interaction potentials:
Coulomb, Yukawa, Exact-gradient corrected Yukawa, Quantum Statistical Potentials, Moliere, Lennard-Jones 6-12. More info
on each of these potential can be found in :ref:`potentials`. Next we define the screening parameter ``kappa``,
see :ref:`potentials-yukawa`. Notice that this a non-dimensional parameter, that is the real screening length will be
calculated from :math:`\lambda = a/\kappa` where :math:`a` is the Wigner-Seitz radius. Finally we define the cut-off radius for the Particle-Particle part of the P3M algorithm
by ``rc``.

Interaction Algorithm
---------------------
The ``P3M`` section that follows is essential for our simulation as it defines important parameters.
The parameters in this section are what differentiate a good simulation from a bad simulation.
Details on how to choose these parameters are given later in this page, but for now we limit to describing them

.. code-block:: python

    P3M:
        - MGrid: [64,64,64]
        - aliases: [3,3,3]
        - cao: 6
        - alpha_ewald: 1.16243741e+10  # 1/[m]

The ``MGrid`` instance is a list of 3 elements corresponding to the number of mesh points in each of the three cartesian
directions, ``aliases`` indicates the number of aliases for anti-aliasing, see <link to anti-aliasing>. ``cao`` stands
for Charge Order Parameter and indicates the number of mesh points per direction on which the each particle's charge is
to distributed and finally ``alpha_ewald`` refers to the :math:`\alpha` parameter of the Gaussian charge cloud
surrounding each particle.

Boundary Conditions
-------------------

Next we define the boundary conditions for our simulation.

.. code-block:: python

    BoundaryConditions:
        - periodic: ["x", "y", "z"]

The instance ``periodic`` takes in a list of three string elements which correspond to the each of the three cartesian
direction.
At the moment Sarkas supports only ``periodic`` boundary conditions and ``open`` boundary conditions which requires
the ``FMM`` algorithm in all directions at once. Future implementations of Sarkas accepting mixed
boundary conditions are under way, but not fully supported. We accept pull request :) !

Integrator
----------

Notice that we have not defined our integrator yet. This is done in the section ``Integrator`` of the input file

.. code-block:: python

    Integrator:
        - type: Verlet

Here ``Verlet`` refers to the common ``Velocity Verlet`` algorithm in which particles velocity are updated first, not to be
confused with the ``Position Verlet`` algorithm. The two algorithms are equivalent, however, Velocity Verlet is the most
efficient and the preferred choice in most MD simulations. Currently Sarkas supports also the magnetic Velocity Verlet,
see ``ybim_mks_p3m_mag.yaml`` and more details are discussed in ... . Further integrators scheme are under development: these
include adaptive Runge-Kutta, symplectic high order integrators, multiple-timestep algorithms. The Murillo group
is currently looking for students willing to explore all of the above.

Thermostat
----------
Most MD simulations require an thermalization phase in which the system evolves in time in an :math:`NVT` ensemble
so that the initial configuration relaxes to the desired thermal equilibrium. The parameters
of the thermalization phase are defined in the ``Thermostat`` section of the input file.

.. code-block:: python

    Thermostat:
        - type: Berendsen               # thermostat type
        - thermostating_temperatures_eV: 0.5
        - timestep: 2000
        - tau: 5.0

The first instance defines the type of Thermostat. Currently Sarkas supports only the Berendsen type, but other
thermostats like Langevin, Nose-Hoover, etc are, you guessed it!, in development. The second instance defines the
temperature (be careful with units!) at which the system is to be thermalized. Notice that this takes a single value
as input in the case of a single species, while it takes is a list in the case of multicomponent plasmas. Note that
these temperatures need not be the same as those defined in ``Particles.species.temperature`` as it might be the case
that you want to study temperature relaxation in plasma mixtures.
The ``timestep`` instance indicates the timestep number at which the Berendsen thermostat will be turned on.
In this case for timesteps < 2000 particles' velocities will be rescaled by the desired equilibrium temperatures. This
is not a desirable choice as it does not allow for temperature fluctuations and can lead to misleading results. The
instance ``tau`` indicates the relaxation rate of the Berendsen thermostat, see :ref:`thermostats` for more details.

Control
-------
The next section defines some general parameters

.. code-block:: python

    Control:
        - units: mks                  # units
        - dt: 1.193536e-17            # sec
        - Neq: 10000                  # number of timesteps for the equilibrium
        - Nsteps: 30000               # number of timesteps afater the equilibrium
        - dump_step: 1000             # dump time step
        - verbose: yes
        - simulations_dir: Simulations
        - output_dir: yukawa_mks_p3m  # dir name to save data.
        - dump_dir: Particles_Data
        - job_id: YOCP_T05eV  # dir name to save data.
        - writexyz: yes               # no xyz output

The first instance defines the choice of units (mks or cgs) which must be consistent with all the other dimensional parameters
defined in previous sections. The second instance is the value of the timestep always given in sec independent of the
choice of units. ``Neq`` is the number of thermalization (or equilibration) timesteps. ``Nsteps`` is the number of
timesteps of the production phase. ``dump_step`` is the interval timestep over which Sarkas will save simulations data
for restarts. ``verbose`` is flag for printing progress to screen. This is useful in the initialization phase of an MD
simulation. The next five instances are not needed, however, they are useful for organizing your work. ``simulations_dir``
is the directory where all the simulations will be saved. The default value is ``Simulations`` and this will be
created in your current working directory. Next, ``output_dir`` is the name of the directory of this specific simulation
which we chose to call ``yukawa_mks_p3m``. This directory will contain a ``pickle`` storing all your simulations
parameters and physical constants, a log file of your simulation, an csv file for storing energy information at each
dump, and all the other file produced in the post-processing phase. Every ``dump_step`` Sarkas will save particles'
position, velocities, acceleration, and other relevant data in an ``.npz`` file in the ``dump_dir`` directory inside the
``output_dir`` directory. Finally ``job_id`` is an appendix for all the file names identifing this specific run. This
is useful when you have many runs that differ only in the choice of ``random_seed``. Finally ``writexyz`` is a flag for
whether to save and ``.xyz`` file used for visualization by OVITO. Notice that an ``.xyz`` file of your simulation can
be created also in the Post processing phase and need not be written in the production phase.

Post Processing
---------------

The last section is ``PostProcessing`` and contains all those parameters relevant to the physical observable that need
be calculated during the production phase. The radial distribution function (RDF) is a very common quantity that is more
efficiently calculated in the production phase than in the post-processing phase. Hence, we chose to divide our
RDF into 300 bins

.. code-block:: python

    PostProcessing:
        - rdf_nbins: 300
        - dsf_no_ka_values: [20, 20, 20]
        - ssf_no_ka_values: [20, 20, 20]

The other two instances define the max number of harmonics of the :math:`ka` vector for the calculation of the
Dynamical Structure Factor (DSF) and Static Structure Factor (SSF). These last two are not necessary as the DSF and SSF
can be easily calculated in the post-processing phase. They are here so that we don't have to define them later.

Testing
=======

Running a simulation
====================

Post Processing
===============
