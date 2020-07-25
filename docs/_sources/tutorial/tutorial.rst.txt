.. _tutorial:

========
Tutorial
========

Here we provide a step-by-step guide to illustrate how to run Sarkas and how to obtain useful quantities
from your simulation data once you run is complete. The topics that are listed in this tutorial are as follows

- Options
- Input file
- Testing
- Running a simulation
- Post Processing

We will illustrate these steps using the example file ``yukawa_mks_p3m.yaml`` in the ``examples`` folder.
This file contains parameters for a :math:`NVE` simulation of a One component strongly coupled plasma whose
particles interact via a Yukawa potential, i.e. an YOCP. There are a couple of ways in which we can run a simulation.

#. From a terminal window

#. From a python script/jupyter notebook

I will describe the first option for now

Options
=======
First and foremost we run the help command in a terminal window

.. code-block:: bash

    $ sarkas_simulate -h

This will produce the following output

.. figure:: Help_output.png
    :alt: Figure not found

This output prints out the different options with which you can run Sarkas.

- ``-i`` is required and is the path to the YAML input file of our simulation.
- ``-c`` which can be either ``prod`` or ``therm`` and indicates whether we want to check the thermalization or production phase of the run.
- ``-t`` is a boolean flag indicating whether to run a test of our input parameter or not.
- ``-p`` is a boolean flag indicating whether to show plots to screen.
- ``-v`` boolean for verbose output.
- ``-d`` indicates the name of the directory where to save the simulation's output.
- ``-j`` refers to the name we want to append to each output file.
- ``-s`` sets the random number seed.

The last three option are not required and are mostly used in the case we want to run many simulations without creating
multiple input files and only want to change only few control parameters. Very often we need to run many simulations
with similar parameters but with different initial conditions. Hence we could write a bash script like so

.. code-block:: bash

    conda activate sarkas

    sarkas_simulate -i sarkas/examples/yukawa_mks_p3m.yaml -s 125125 -d run1
    sarkas_simulate -i sarkas/examples/yukawa_mks_p3m.yaml -s 281756 -d run2
    sarkas_simulate -i sarkas/examples/yukawa_mks_p3m.yaml -s 256158 -d run3
    sarkas_simulate -i sarkas/examples/yukawa_mks_p3m.yaml -s 958762 -d run4
    sarkas_simulate -i sarkas/examples/yukawa_mks_p3m.yaml -s 912856 -d run5

    conda deactivate

If you are familiar with bash scripting you could make the above statements in a loop and make many more simulations.

This was the old way of running simulations. Sarkas allows you to run write a Python script to run and analyze all your
runs. A simple script looks like this

.. code-block:: python

    import numpy as np

    from sarkas.simulation.params import Params
    from sarkas.simulation import simulation

    # Let's define some common variables
    args = dict()
    args["input_file"] = "../sarkas/examples/yukawa_mks_p3m.yaml" # choose the right location


    # Initialize the simulation parameter class
    params = Params()

    for i in range(10):
        args["job_dir"] = "YOCP_" + str(i)
        args["job_id"] = "yocp_" + str(i)
        args["seed"] = np.random.randint(0, high=123456789)
        params.setup(args)
        simulation.run(params)

At the same time let's assume we want to run many simulations to span a range of coupling parameters

.. code-block:: python

    import numpy as np

    from sarkas.simulation.params import Params
    from sarkas.simulation import simulation
    from sarkas.potentials import yukawa

    # Let's define some common variables
    args = dict()
    args["input_file"] = "../sarkas/examples/yukawa_mks_p3m.yaml" # choose the right location
    kappas = np.linspace(0.1, 1, 10)

    for i, kappa in enumerate(kappas):
        args["job_dir"] = "YOCP_" + str(i)
        args["job_id"] = "yocp_" + str(i)
        args["seed"] = np.random.randint(0, high=123456789)

        # Initialize the simulation parameter class
        params = Params()
        # Read the common simulation's parameters
        params.common_parser(args["input_file"])
        # Let's make sure we are not printing to screen
        params.Control.verbose = False
        # Create the directories of each simulation output
        params.create_directories(args)
        # Create simulation's parameters
        params.assign_attributes()
        # Let's change the screening parameter
        params.Potential.kappa = kappa
        # Calculate potential dependent parameters
        yukawa.setup(params, False)
        # Run the simulation
        simulation.run(params)
        # Delete params and restart
        del params

In the next section we will explain the input file.

Input File
==========

Particles
---------
The first step in any MD simulation is the creation of an input file containing all the relevant parameters
of our simulation. We start by considering the physical parameter of our system. Assume we want to simulate
a strongly coupled plasma comprised of :math:`N = 10\, 000` hydrogen ions with
a number density of :math:`n = 1.6 \times 10^{30} N/m^3` at a temperature :math:`0.5 eV`.
These parameters are defined in our ``yaml`` file by the block ``Particles`` and its attribute ``species``,

.. code-block:: yaml

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

.. code-block:: yaml

    Potential:
        - type: Yukawa
        - method: P3M            # Particle-Particle Particle-Mesh
        - kappa: 0.5
        - rc: 2.79946255e-10     # [m]


The instance ``type`` defines the interaction potential. Currently Sarkas supports the following interaction potentials:
Coulomb, Yukawa, Exact-gradient corrected Yukawa, Quantum Statistical Potentials, Moliere, Lennard-Jones 6-12. More info
on each of these potential can be found in :ref:`potentials`. Next we define the screening parameter ``kappa``. Notice that this a non-dimensional parameter, that is the real screening length will be
calculated from :math:`\lambda = a/\kappa` where :math:`a` is the Wigner-Seitz radius. Finally we define the cut-off radius for the Particle-Particle part of the P3M algorithm
by ``rc``.

Interaction Algorithm
---------------------
The ``P3M`` section that follows is essential for our simulation as it defines important parameters.
The parameters in this section are what differentiate a good simulation from a bad simulation.
Details on how to choose these parameters are given later in this page, but for now we limit to describing them

.. code-block:: yaml

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

.. code-block:: yaml

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

.. code-block:: yaml

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

.. code-block:: yaml

    Thermostat:
        - type: Berendsen               # thermostat type
        - temperatures_eV: 0.5
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

.. code-block:: yaml

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

.. code-block:: yaml

    PostProcessing:
        - rdf_nbins: 300
        - dsf_no_ka_values: [20, 20, 20]
        - ssf_no_ka_values: [20, 20, 20]

The other two instances define the max number of harmonics of the :math:`ka` vector for the calculation of the
Dynamical Structure Factor (DSF) and Static Structure Factor (SSF). These last two are not necessary as the DSF and SSF
can be easily calculated in the post-processing phase. They are here so that we don't have to define them later.

Pre Simulation Testing
======================
Now that we have created our input file we need to verify that our simulation parameters will give a good simulation.
This is done by running

.. code-block:: bash

    $ python src/Sarkas.py -i examples/yukawa_mks_p3m.yaml -t

in your terminal or

.. code-block:: python

    %run src/Sarkas.py -i examples/yukawa_mks_p3m.yaml -t

in your IPython kernel or Jupyter Notebook (to be expanded). The number at the end indicates the number of loops
over which we wish to average the force calculation time. The first part of the output of this command looks something
like this

.. image:: S_testing_output_1.png
    :alt: S_testing_output_1.png not found

As you can see most of the simulation parameters defined in the input file are repeated here together with other
important information. For example, in the section "Length scales:" we find the value of the Wigner-Seitz radius, the
number of non zero dimensions, and the length of the simulation box sides in terms of :math:`a_{ws}` and its numerical
value in the chosen units. Few lines below we find the Potential section which shows all the relevant parameters of our
chosen potential. Note that this section depends on the type of potential and as such it varies. Next we find
the Algorithm section. This is particularly verbose in the case of the P3M algorithm since we have parameters for the PP
and PM part of the algorithm. The two important parameter are: the Ewald parameter :math:`\alpha` and
the cutoff radius, :math:`r_c`. Below the line ``Mesh = [64 64 64]`` the number of cells
per dimension for the Linked Cell algorithm and the number of particles inside a spheres of radius rcut. Next we find
the most important information: the error in the force calculation.

Before explaining the force error calculation we show the second part of the output of the command which gives the
average time for the PP and PM part of the force calculation and an estimate of the total run of the simulation

.. image:: S_testing_output_2.png
    :alt: S_testing_output_2.png not found

.. note::

    These times will vary depending on the computer hardware. For this tutorial we used a 2019 Dell XPS 8930
    with Intel Core i7-8700K @ 3.70Ghz and 16GB of RAM running Ubuntu 18.04.

As you can see the calculation of the optimal Green's function takes a long time. Fortunately this needs only be
calculated once at the beginning of the simulation. We note also that the PP part takes more than twice the time it
takes for the PM part. This is specific to this hardware and the opposite case could true on other machines.

In addition to this screen output the command produces two plots that will help in the decision of the P3M parameters.
These plots are saved in the job directory ``Simulations/yukawa_mks_p3m``, but before viewing them we need to explain
how these plots are calculated.

Force Error calculation
-----------------------
The Force error is the error incurred when we cut the potential interaction after a certain distance. Following the works
of :cite:`Kolafa1992,Stern2008,Dharuman2017` we define the total force error for our P3M algorithm as

.. math::

    \Delta F_{\textrm{tot}} = \sqrt{ \Delta F_{\mathcal R}^2 + \Delta F_{\mathcal F}^2 }

where :math:`\Delta F_{\mathcal R}` is the error obtained in the PP part of the force calculation and
:math:`\Delta F_{\mathcal F}` is the error obtained in the PM part, the subscripts :math:`\mathcal{R, F}` stand for
real space and Fourier space respectively. :math:`\Delta F_{\mathcal R}` is calculated as follows

.. math::

    \Delta F_{\mathcal R} = \sqrt{\frac{N}{V} } \left [ \int_{r_c}^{\infty} d^3r
        \left | \nabla \phi_{\mathcal R}( \mathbf r) \right |^2  \right ]^{1/2},

where :math:`\phi_{\mathcal R}( \mathbf r)` is the short-range part of the chosen potential. In our example case of a
Yukawa potential we have

.. math::

    \phi_{\mathcal R}(r) = \frac{Q^2}{2r}
        \left [ e^{- \kappa r} \text{erfc} \left( \alpha r - \frac{\kappa}{2\alpha} \right )
            + e^{\kappa r} \text{erfc} \left( \alpha r + \frac{\kappa}{2\alpha} \right ) \right ],

where :math:`\kappa, \alpha` are the dimensionless screening parameter and Ewald parameter respectively and, for the
sake of clarity, we have a charge :math:`Q = Ze/\sqrt{4\pi \epsilon_0}` with an ionization state of :math:`Z = 1`. Integrating this potential,
and neglecting fast decaying terms, we find

.. math::

    \Delta F_{\mathcal R} \simeq 2 Q^2 \sqrt{\frac{N}{V}} \frac{e^{-\alpha^2 r_c^2}}{\sqrt{r_c}} e^{-\kappa^2/4 \alpha^2}.

On the other hand :math:`\Delta F_{\mathcal F}` is calculated from the following formulas

.. math::

    \Delta F_{\mathcal F} =  \sqrt{\frac{N}{V}} \frac{Q^2 \chi}{\sqrt{V^{1/3}}}

.. math::

    \chi^2V^{2/3}  = \left ( \sum_{\mathbf k \neq 0} G_{\mathbf k}^2 |\mathbf k |^2 \right )
        - \sum_{\mathbf n} \left [ \frac{\left ( \sum_{\mathbf m} \hat{U}_{\mathbf{k + m}}^2
        G_{\mathbf{k+m}} \mathbf{k_n} \cdot \mathbf{k_{n + m}} \right )^2 }{ \left( \sum_{\mathbf m} \hat{U}_{\mathbf{k_{n+m}}}^2 \right )^2 |\mathbf{k_{n} }|^2 } \right ].

This is a lot to take in, so let's unpack it. The first term is the RMS of the force field in Fourier space
obtained from solving Poisson's equation :math:`-\nabla \phi(\mathbf r) = \delta( \mathbf r - \mathbf r')` in Fourier
space. In a raw Ewald algorithm this term would be the PM part of the force. However, the P3M variant
solves Poisson's equation on a Mesh, hence, the second term which is non other than the RMS of the force obtained on the mesh.
:math:`G_{\mathbf k}` is the optimal Green's function which for the Yukawa potential is

.. math::
    G_{\mathbf k} = \frac{4\pi e^{-( \kappa^2 + \left |\mathbf k \right |^2)/(4\alpha^2)} }{\kappa^2 + |\mathbf {k}|^2}

where

.. math::

     \mathbf k ( n_x, n_y, n_z) = \mathbf{k_n} = \left ( \frac{2 \pi n_x}{L_x},
                                                        \frac{2 \pi n_y}{L_y},
                                                        \frac{2 \pi n_z}{L_z} \right ).

:math:`\hat{U}_{\mathbf k}` is the Fourier transform of the B-spline of order :math:`p`

.. math::

    \hat U_{\mathbf{k_n}} = \left[ \frac{\sin(\pi n_x /M_x) }{ \pi n_x/M_x} \right ]^p
    \left[ \frac{\sin(\pi n_y /M_y) }{ \pi n_y/M_y} \right ]^p
    \left[ \frac{\sin(\pi n_z /M_z) }{ \pi n_z/M_z} \right ]^p,

where :math:`M_{x,y,z}` is the number of mesh points along each direction. Finally the :math:`\mathbf{m}` refers to the
triplet of grid indices :math:`(m_x,m_y,m_z)` that contribute to aliasing. Note that in the above equations
as :math:`\kappa \rightarrow 0` (Coulomb limit), we recover the corresponding error estimate for the Coulomb potential.

The reason for this discussion is that by inverting the above equations we can find optimal parameters
:math:`r_c,\; \alpha` given some desired errors :math:`\Delta F_{\mathcal {R,F}}`. While
the equation for :math:`\Delta F_{\mathcal R}` can be easily inverted for :math:`r_c`, such task seems impossible for
:math:`\Delta F_{\mathcal F}` without having to calculate a Green's function for each chosen :math:`\alpha`. As you can
see in the second part of the output the time it takes to calculate :math:`G_{\mathbf k}` is in the order of seconds,
thus, a loop over several :math:`\alpha` values would be very time consuming. Fortunately researchers
have calculated an analytical approximation allowing for the exploration of the whole :math:`r_c,\; \alpha` parameter
space :cite:`Dharuman2017`. The equations of this approximation are

.. math::
    \Delta F_{\mathcal F}^{(\textrm{approx})} \simeq Q^2 \sqrt{\frac{N}{V}} A_{\mathcal F}^{1/2},

.. math::
    A_{\mathcal F} \simeq \frac{3}{2\pi^2} \sum_{m = 0}^{p -1 } C_{m}^{(p)} \left ( \frac{h}2 \right )^{2 (p + m)}
                            \frac{2}{1 + 2(p + m)} \beta(p,m),

.. math::
    \beta(p,m) = \int_0^{\infty} dk \; G_k^2 k^{2(p + m + 2)},

where :math:`h = L_x/M_x` and the coefficients :math:`C_m^{(p)}` are listed in Table I of :cite:`Deserno1998`.

Finally, by calculating

.. math::

    \Delta F_{\textrm{tot}}^{(\textrm{apprx})}( r_c, \alpha) = \sqrt{ \Delta F_{\mathcal R}^2 +
            ( \Delta F_{\mathcal F}^{(\textrm{approx})} ) ^2 }

we are able to investigate which parameters :math:`r_c,\; \alpha` are optimal for our simulation.

As mentioned before running ``S_testing.py`` produces two figures. These are used to find the best parameters for our
force calculations by comparing
:math:`\Delta F_{\textrm{tot}}^{(\textrm{apprx})}` and :math:`\Delta F_{\textrm{tot}}`. The first figure
produced by our example is shown below and it is a contour map of :math:`\Delta F_{\textrm{tot}}^{(\textrm{apprx})}`
in the :math:`r_c,\, \alpha` parameters space

.. image:: Pre_Run_TestForceError_ClrMap_yukawa_mks_p3m.png
    :alt: Figure not found

The numbers on the white contours indicate the value of :math:`\Delta F_{\textrm{tot}}^{(\textrm{apprx})}` along those
lines and the black dot indicates where our choice of parameters fall into this parameter space. We notice that our
parameter choice falls exactly on the white line, and thus it is :\math:`\sim 1e-5`. Comparing this with the value printed
on screen from the first figure above we find that our analytical approximation is quite close to the real value
:math:`\Delta F_{\textrm{tot} }`. Furthermore, this plot tells us that if we want a force error of the order 1e-6 we need
to choose values that fall into the small purple triangle at the top.

However, our choice of parameters while being good, it might not be optimal. In order to find the best choice we look at
the second figure created by ``S_testing.py``, given below

.. image:: Pre_Run_TestForceError_LinePlot_yukawa_mks_p3m.png
    :alt: Figure not found

The left panel is a plot of :math:`\Delta F_{\textrm{tot}}^{(\textrm{apprx})}` vs :math:`r_c/a_{ws}` at
five different values of :math:`\alpha a_{ws}` while the right panel is a plot of
:math:`\Delta F_{\textrm{tot}}^{(\textrm{apprx})}` vs :math:`\alpha a_{ws}` at
five different values of :math:`r_c/a_{ws}`. The vertical black dashed lines indicate the values of
:math:`\alpha a_{ws}` and :math:`r_c/a_{ws}` chosen in the input file. The horizontal black dashed lines, instead,
indicate the value of :math:`\Delta F_{\textrm{tot}}`.
Again you can see that our analytical approximation is a very good approximation and that our choice of parameters is not
optimal. Notice that the cyan line corresponds to our choice of :math:`\alpha` and :math:`r_c`.
The left panel shows that the cyan line reaches its minimum value at :math:`r_c \simeq 6.0 a_{ws}`.
Any value greater than this would cause the code to be inefficient since we will be calculating the interaction
for many more particles without actually reducing the force error. Similarly, the right panel shows that our choice
of :math:`r_c` is close to optimal given :math:`\alpha a_{ws} = 0.614`.

Some good rules of thumb to keep in mind while choosing the parameters are

- larger (smaller) :math:`\alpha` lead to a smaller (larger) PM error, but to a larger (smaller) PP error,
- larger (smaller) :math:`r_c` lead to a smaller (greater) PP part but do not affect the PM error,
- keep an eye on the PM and PP calculation times.
- larger :math:`r_c` lead to a longer time spent in calculating the PP part of the force since there are more neighbors,
- larger or smaller :math:`\alpha` do not affect the PM calculation time since this depends on the number of mesh points,
- choose the number of mesh points to be a power of 2 since FFT algorithms are most efficient in this case.

.. note::

    Notice that the above investigation is useful in choosing the parameters :math:`r_c` and :math:`\alpha`
    for fixed values of the charge approximation order, :math:`p`,
    the number of mesh points, :math:`M_x = M_y = M_z`, and number of aliases :math:`m_x = m_y = m_z`.


Running a simulation
====================

Once we have chosen the parameters, we are ready to start a simulation by typing

.. code-block:: bash

    $ python src/Sarkas.py -i examples/yukawa_mks_p3m.yaml

Since we have chosen ``verbose = yes`` this simulation will print a progress bar to screen, thanks to the package ``tqdm``

.. image:: SimRun1.png
    :alt: Figure not found

Did you think that you could get away so easily? We need to check if our run is doing what we want. To do so, in a
different terminal window we run

.. code-block:: bash

    $ python src/Sarkas.py -i examples/yukawa_mks_p3m.yaml -c therm

Note the option ``-c`` takes the value ``therm`` if we are in the thermalization phase, otherwise we give ``prod``.
This produces the following plot


Post Processing
===============

Now comes the fun part! The first thing we want to do is to check for energy conservation again.


Plot of the Total Energy as a function of time.



.. bibliography:: references.bib

