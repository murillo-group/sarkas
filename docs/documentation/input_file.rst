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
            name: H                                     # REQUIRED
            num: 10000                                  # REQUIRED
            Z: 1.0                                      # REQUIRED/OVERWRITTEN if charge is used
            # charge: 1.602177e-19                      # REQUIRED unless Z is used.
            number_density: 1.62e+32                    # REQUIRED/OPTIONAL if mass_density is used
            mass: 1.673e-27                             # REQUIRED/OPTIONAL if mass_density is used
            # atomic_weight: 1.0                        # OPTIONAL/REQUIRED if mass_density is used
            # mass_density: 2.710260e+05
            temperature_eV: 0.5                         # REQUIRED/OPTIONAL if temperature is used
            # temperature:  5.802259e+03                # REQUIRED/OPTIONAL if temperature_eV is used
            initial_velocity_distribution: boltzmann    # OPTIONAL

In the case of a multi-component plasma we need only add another ``Species`` attribute with corresponding physical
parameters, see the H-He mixture in the example page. The attributes of ``Species`` take only numerical values,
apart from ``name``, in the correct choice of units which is defined in the block ``Parameters``, see below.
Notice that in this section we also define the mass of the particles, ``mass``, and their charge number ``Z``.
Future developments of Sarkas are aiming to automatically calculate the degree of ionization given by the density and
temperature of the system, but for now we need to define it. The parameters given here are not the only options,
more information of all the possible inputs can be found in the page ``sarkas.plasma.Species``.

The initial velocity distribution can be set by ``initial_velocity_distribution`` and defaults to a ``boltzmann``
distribution but can also be set to ``monochromatic`` where a fixed energy is applied to the particles with a random
distribution of the directions.

Interaction
-----------
The next section of the input file defines our interaction potential's parameters

.. code-block:: yaml

    Potential:
        type: Yukawa                            # REQUIRED
        screening_length_type: "thomas-fermi"   # REQUIRED for screened potentials
        electron_temperature_eV: 1.25e+3        # REQUIRED if 'thomas-fermi' type
        method: pppm                            # REQUIRED
        rc: 6.2702e-11                          # REQUIRED
        pppm_mesh: [64, 64, 64]                 # REQUIRED
        pppm_aliases: [3,3,3]                   # REQUIRED
        pppm_cao: 6                             # REQUIRED
        pppm_alpha_ewald: 5.4659e+10            # REQUIRED

The instance ``type`` defines the interaction potential. Currently Sarkas supports the following interaction potentials:
Coulomb, Yukawa, Exact-gradient corrected Yukawa, Quantum Statistical Potentials, Moliere, Lennard-Jones 6-12. More info
on each of these potential can be found in `Potentials <./Features_files/potentials.rst>`_. Next we define the type of
screening we desire. The available choices are [`kappa`, `thomas-fermi`, `debye-huckel`, `custom`]. In our case we chose
`kappa` which means that the screening length will be calculated from :math:`\lambda = a_{ws}/\kappa` where
:math:`a_{ws}` is the Wigner-Seitz radius and :math:`kappa` is the value of the next attribute `kappa`.
Notice that this a non-dimensional parameter.


The following parameters refer to our choice of the interaction algorithm (`method`). Details on how to choose these
parameters are given later in this page, but for now we limit to describing them. First, we find the cut-off radius,
``rc``, for the Particle-Particle part of the PPPM algorithm. The ``pppm_mesh`` attribute is a list of three elements
corresponding to the number of mesh points in each of the three cartesian coordinates, ``pppm_aliases`` indicates
the number of aliases for anti-aliasing, ``pppm_cao`` stands for Charge Order Parameter and indicates the number of mesh
points per direction on which the each particle's charge is to be distributed and finally ``pppm_alpha_ewald`` refers to
the :math:`\alpha` parameter of the Gaussian charge cloud surrounding each particle.

To deal with diverging potentials a short-range cut-off radius, ``a_rs``, can be specified. If specified, the potential
:math:`U(r)` will be cut to :math:`U(a_{rs})` for interparticle distances below ``a_rs``. This short-range cut-off is meant to
suppress unphysical scenarios where fast particles emerge due to the potential going to infinity. However, this feature
should be used with great care as is can also screen the short-range part of the interaction to unphysical values. That
is why the default value is zero so that the short-range cut-off is not in use.

Integrator
----------
Notice that we have not defined our integrator yet. This is done in the section ``Integrator`` of the input file

.. code-block:: yaml

    Integrator:
        dt: 2.000e-18                       # REQUIRED
        equilibration_type: verlet          # REQUIRED
        production_type: verlet             # REQUIRED
        boundary_conditions: periodic       # REQUIRED
        thermalization: yes                 # OPTIONAL. Default = yes
        thermostat_type: Berendsen          # REQUIRED if thermalization is yes
        thermalization_timestep: 50         # REQUIRED if thermalization is yes
        berendsen_tau: 1.0                  # REQUIRED if thermostat: berendsen
        thermostate_temperatures_eV: 0.5    # OPTIONAL Default = Species.temperature_eV

The attribute `dt` indicates the timestep, in seconds, of our simulation. Next we find our choice of integrator. In this case we
need not pass both ``equilibration_type`` and ``production_type`` and a simple ``type: verlet`` would suffice. However,
we use both types here for educational purposes. It could be the case that you want to use different integrators
for different simulation phases, e.g. a Langevin integrator for the equilibration phase and a verlet integrator for
the production phase. ``verlet`` refers to the common Velocity Verlet algorithm in which particles velocities
are updated first. This must not to be confused with the Position Verlet algorithm.
The two algorithms are equivalent, however, Velocity Verlet is the most efficient and the preferred choice in most MD simulations.

Next we define the ``boundary_conditions`` of our simulation. At the moment Sarkas supports only ``periodic`` and
``absorbing`` boundary conditions.
Future implementations of Sarkas accepting open and mixed boundary conditions will be available in the future.
We accept pull request :) !

Next we find information for our thermostat. If we do not wish to thermalize our system with a bath we need set
``thermalization: no``. The default value is ``yes`` and it could be omitted, however, we must define the ``thermostat_type``
and ``thermalization_timestep`` if we are using a thermostat. ``thermalization_timestep`` indicates the timestep number
at which the Berendsen thermostat will be turned on and the instance ``berendsen_tau`` indicates the relaxation rate of
the Berendsen thermostat, see `Berendsen Thermostat <./Features_files/Berendsen_NB/Berendsen_Thermostat.ipynb>`_ for more details. These last two
instances have no default value and as such they must be defined. Currently Sarkas supports only the Berendsen thermostat.

The last instance defines the temperature at which the system is to be thermalized (be careful with units!) .
Notice that this takes a single value in the case of a single species, while it takes is a list in the case of
multicomponent plasmas. Note that these temperatures need not be the same as those defined in the ``Particles`` block as
it might be the case that you want to study temperature relaxation.

``equilibration_steps`` and ``production_steps`` are the number of timesteps of the equilibration and production phase,
respectively. ``eq_dump_step`` and ``prod_dump_step`` are the interval timesteps over which Sarkas will save simulations
data.

Further integrators scheme are under development: these include adaptive Runge-Kutta, symplectic high order integrators,
multiple-timestep algorithms. The Murillo group is currently looking for students willing to explore all of the above.

Parameters
----------
The next section defines some general parameters

.. code-block:: yaml

    Parameters:
        units: mks                          # REQUIRED
        load_method: random_no_reject       # REQUIRED
        equilibration_steps: 5000           # REQUIRED
        production_steps: 5000              # REQUIRED
        eq_dump_step: 10                    # REQUIRED
        prod_dump_step: 10                  # REQUIRED

The first instance defines the choice of units (mks or cgs) which must be consistent with all the other dimensional
parameters defined in previous sections. ``load_method`` defines the way particles positions are to be initialized.
The options are

- ``random_no_reject`` for a uniform spatial distribution
- ``random_reject`` for a uniform spatial distribution but with a minimum distance between particles
- ``halton``
- ``lattice`` either a 3D simple cubic or a 2D hexagonal

By specifying ``Lx``, ``Ly`` and ``Lz`` the simulation box can be specified explicitly and expanded with respect
to the initial particle distribution. This moves the walls where boundary conditions are applied away from the
initial particle volume.

Input/Output
------------
The next section defines some IO parameters

.. code-block:: yaml

    IO:
        verbose: yes                        # OPTIONAL. Default is yes
        simulations_dir: Simulations        # OPTIONAL. Default is Simulations
        job_dir: yocp_pppm                  # REQUIRED
        job_id: yocp                        # OPTIONAL. Default is the job_dir values

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

The last two blocks are ``Observables`` and ``TransportCoefficientss``. They indicate the quantities
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

        - VelocityAutoCorrelationFunction
            no_slices: 4

Note that ``Observables`` is again a list of dictionaries. This is because each observable is returned as
an object in the simulation. The lines below the observables' names are the parameters needed for the calculation.
The parameters are different depending on the observable. We will discuss them in the next pages of this tutorial.


Transport Coefficients
**********************

.. code-block:: yaml

    TransportCoefficients:
        - Diffusion:
            no_slices: 4

The available transport coefficients at this moment are: ``Diffusion``, ``Interdiffusion``, ``ElectricalConductivity``,
``Viscosity``. Note that ``Interdiffusion`` is supported only in the case of binary mixtures.
Soon we will have support for any mixture.
