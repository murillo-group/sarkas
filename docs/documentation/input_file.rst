==========
Input File
==========
The first step in any MD simulation is the creation of an input file containing all the relevant parameters
of our simulation. Take a look at the file ``yukawa_mks_p3m.yaml`` that can be
found `here <https://raw.githubusercontent.com/murillo-group/sarkas/master/docs/documentation/Tutorial_NB/input_files/yukawa_mks_p3m.yaml>`_.


It is very important to maintain the syntax shown in the example YAML files.
This is because the content of the YAML file is returned as a dictionary of dictionaries.


Particles
---------
The ``Particles`` block contains the attribute ``Species`` which defines the first type of particles, i.e. species,
and their physical attributes.

.. code-block:: yaml

    Particles:
        - Species:
            name: H
            number_density: 1.62e+32       # /m^3
            mass: 1.673e-27                # kg, ptcl mass of ion1
            num: 10000                     # total number of particles of ion1
            Z: 1.0                         # degree of ionization
            temperature_eV: 0.5


In the case of a multi-component plasma we need only add another ``Species`` attribute with corresponding physical
parameters, see ``ybim_mks_p3m.yaml``. The attributes of ``Species`` take only numerical values, apart from ``name``,
in the correct choice of units which is defined in the block ``Control``, see below.
Notice that in this section we also define the mass of the particles, ``mass``, and their charge number ``Z``.
Future developments of Sarkas are aiming to automatically calculate the degree of ionization given by the density and
temperature of the system, but for now we need to define it. The parameters given here are not the only options,
more information of all the possible inputs can be found in the page ``input file``.

The initial velocity distribution can be set by ``initial_velocity_distribution`` and defaults to a ``boltzmann``
distribution but can also be set to ``monochromatic`` where a fixed energy is applied to the particles with a random
distribution of the directions.

Interaction
-----------
The next section of the input file defines our interaction potential's parameters

.. code-block:: yaml

    Potential:
        type: Yukawa
        method: P3M                       # Particle-Particle Particle-Mesh
        kappa: 0.5
        rc: 2.79946255e-10                # [m]
        pppm_mesh: [64, 64, 64]
        pppm_aliases: [3,3,3]
        pppm_cao: 6
        pppm_alpha_ewald: 1.16243741e+10  # 1/[m]

The instance ``type`` defines the interaction potential. Currently Sarkas supports the following interaction potentials:
Coulomb, Yukawa, Exact-gradient corrected Yukawa, Quantum Statistical Potentials, Moliere, Lennard-Jones 6-12. More info
on each of these potential can be found in :ref:`potentials`. Next we define the screening parameter ``kappa``.
Notice that this a non-dimensional parameter, i.e. the real screening length will be calculated
from :math:`\lambda = a/\kappa` where :math:`a` is the Wigner-Seitz radius.

The following parameters refer to our choice of the interaction algorithm. Details on how to choose these parameters
are given later in this page, but for now we limit to describing them. First, we find the cut-off radius, ``rc``,
for the Particle-Particle part of the P3M algorithm.
The ``pppm_mesh`` instance is a list of 3 elements corresponding to the number of mesh points in each of the three
cartesian coordinates, ``pppm_aliases`` indicates the number of aliases for anti-aliasing, see <link to anti-aliasing>.
``pppm_cao`` stands for Charge Order Parameter and indicates the number of mesh points per direction
on which the each particle's charge is to distributed and finally ``pppm_alpha_ewald`` refers to
the :math:`\alpha` parameter of the Gaussian charge cloud surrounding each particle.

To deal with diverging potentials a short-range cut-off radius, ``rs``, can be specified. If specified, the potential
:math:`U(r)` will be cut to :math:`U(rs)` for interparticle distances below ``rs``. This short-range cut-off is meant to
suppress unphysical scenarios where fast particles emerge due to the potential going to infinity. However, this feature 
should be used with great care as is can also screen the short-range part of the interaction to unphysical values. That
is why the default value is zero so that the short-range cut-off is not in use.

Integrator
----------
Notice that we have not defined our integrator yet. This is done in the section ``Integrator`` of the input file

.. code-block:: yaml

    Integrator:
        type: Verlet                  # velocity integrator type
        equilibration_steps: 10000    # number of timesteps for the equilibrium
        production_steps: 100000      # number of timesteps after the equilibrium
        eq_dump_step: 100
        prod_dump_step: 100

Here ``Verlet`` refers to the common Velocity Verlet algorithm in which particles velocities are updated first. This must
not to be confused with the Position Verlet algorithm. The two algorithms are equivalent, however, Velocity Verlet
is the most efficient and the preferred choice in most MD simulations.
Currently Sarkas supports also the magnetic Velocity Verlet, see ``ybim_mks_p3m_mag.yaml`` and more details are
discussed in ... .
``equilibration_steps`` and ``production_steps`` are the number of timesteps of the equilibration and production phase,
respectively. ``eq_dump_step`` and ``prod_dump_step`` are the interval timesteps over which Sarkas will save simulations
data.

Further integrators scheme are under development: these include adaptive Runge-Kutta, symplectic high order integrators,
multiple-timestep algorithms. The Murillo group is currently looking for students willing to explore all of the above.

Thermostat
----------
Most MD simulations require an thermalization phase in which the system evolves in time in an :math:`NVT` ensemble
so that the initial configuration relaxes to the desired thermal equilibrium. The parameters
of the thermalization phase are defined in the ``Thermostat`` section of the input file.

.. code-block:: yaml

    Thermostat:
        type: Berendsen               # thermostat type
        relaxation_timestep: 50
        berendsen_tau: 1.0
        
The first instance defines the type of Thermostat. Currently Sarkas supports only the Berendsen and Langevin type,
but other thermostats like Nose-Hoover, etc are, you guessed it!, in development.
The ``relaxation_timestep`` instance indicates the timestep number at which the Berendsen thermostat will be turned on.
The instance ``berendsen_tau`` indicates the relaxation rate of the Berendsen thermostat, see :ref:`thermostats` for more details.

The last instance defines the temperature (be careful with units!) at which the system is to be thermalized.
Notice that this takes a single value in the case of a single species, while it takes is a list in the case of
multicomponent plasmas. Note that these temperatures need not be the same as those defined in the ``Particles`` block as
it might be the case that you want to study temperature relaxation in plasma mixtures.


Parameters
----------
The next section defines some general parameters

.. code-block:: yaml

    Parameters:
        units: mks                    # units
        dt: 2.000e-18                 # sec
        load_method: random_no_reject
        boundary_conditions: periodic
        
The first instance defines the choice of units (mks or cgs) which must be consistent with all the other dimensional
parameters defined in previous sections. The second instance is the value of the timestep in seconds.
``load_method`` defines the way particles positions are to be initialized. The options are

- ``random_no_reject`` for a uniform spatial distribution
- ``random_reject`` for a uniform spatial distribution but with a minimum distance between particles
- ``halton``

Next we define the ``boundary_conditions`` of our simulation. At the moment Sarkas supports only ``periodic`` and
``absorbing`` boundary conditions. 
Future implementations of Sarkas accepting open and mixed boundary conditions will be available in the future.
We accept pull request :) !

By specifying ``Lx``, ``Ly`` and ``Lz`` the simulation box can be specified explicitly and expanded with respect
to the initial particle distribution. This moves the walls where boundary conditions are applied away from the
initial particle volume.

Input/Output
------------
The next section defines some IO parameters

.. code-block:: yaml

    IO:
        verbose: yes
        simulations_dir: Simulations
        job_dir: yocp_pppm  # dir name to save data.
        job_id: yocp

``verbose`` is flag for printing progress to screen. This is useful in the initialization phase of an MD
simulation. The next instances are not necessary, as there are default values for them, however, they are useful for organizing your work. ``simulations_dir``
is the directory where all the simulations will be stored. The default value is ``Simulations`` and this will be
created in your current working directory. Next, ``job_dir`` is the name of the directory of this specific simulation
which we chose to call ``yocp_pppm``. This directory will contain ``pickle`` files storing all your simulations
parameters and physical constants, a log file of your simulation, the ``Equilibration`` and ``Production``
directories containing simulations dumps, and ``PreProcessing`` and ``PostProcessing`` directories. Finally ``job_id`` is an appendix for all the file names identifing
this specific run. This is useful when you have many runs that differ only in the choice of ``random_seed``.

Post Processing
---------------

The last two blocks are ``Observables`` and ``TransportCoefficients``. They indicate the quantities 
we want to calculate and their parameters. 

Observables
***********
The observables we want to calculate are 

.. code-block:: yaml

    Observables:
        - RadialDistributionFunction:
            no_bins: 500

        - Thermodynamics:
            phase: production
                
        - DynamicStructureFactor:
            no_slices: 1
            max_ka_value: 8

        - StaticStructureFactor:
            max_ka_value: 8

        - CurrentCorrelationFunction:
            max_ka_value: 8

Note that ``Observables`` is again a list of dictionaries. This is because each observable is returned as 
an object in the simulation. The lines below the observables' names are the parameters needed for the calculation. 
The parameters are different depending on the observable. We will discuss them in the next pages of this tutorial.


Transport Coefficients
**********************

.. code-block:: yaml

    TransportCoefficients:
        - Diffusion:
            time_averaging: False

The available transport coefficients at this moment are: ``Diffusion``, ``Interdiffusion``, ``ElectricalConductivity``, ``Viscosity``. Note 
that ``Interdiffusion`` is supported only in the case of binary mixtures. Soon we will have support for any mixture.
