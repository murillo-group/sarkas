
Sarkas Quickstart
=================
Before you can begin working with Sarkas, you will need to have python3 installed. A useful python 
distribution that has many of the necessary packages for Sarkas is Anaconda. Anaconda can be downloaded here_.

.. _here: https://www.anaconda.com

Obtaining Sarkas
----------------
Sarkas is undergoing heavy development but the most recent version can be found on GitHub_, which 
can then be cloned.

.. _GitHub: https://github.com/murillo-group/sarkas-repo


Preliminary Packages
~~~~~~~~~~~~~~~~~~~~
If you already have a version of python3 installed, you may still need to download additional packages
such as pyfftw_, fdint_, and numba_. If you have the most recent version of Anaconda installed, 
you should already have numba, installed. If you have the python package manager pip_ installed,
you can install these packages, by simply opening a terminal and running the following commands:

To install pyfftw

.. code-block:: bash

   $ pip install pyfftw

To install numba

.. code-block:: bash

   $ pip install numba

To install fdint

.. code-block:: bash

   $ pip install fdint

.. _pyfftw: https://pypi.org/project/pyFFTW/
.. _fdint: https://pypi.org/project/fdint/
.. _numba: https://numba.pydata.org
.. _pip: https://pip.pypa.io/en/stable/


Using Sarkas
------------

Input files
~~~~~~~~~~~
Unless adding new features to Sarkas such as new integrators, thermostats, potentials, etc, the only thing you will  need to modify is the `yaml.inp` file. This file is responsible for specifying the simulation parameters such as the number of particles, number of timesteps, and initialization type for example. 

Example yaml.inp file
~~~~~~~~~~~~~~~~~~~~~~~
To specify you simulation parameters, open the `yaml.inp` file in your text editor and alter the values to
the right of the keywords. Below is a description of what each keyword is used for in Sarkas. 

* Num (int): The number of particles
* method (string): Particle position initialization schemes. So far the options are
   * file: read particles position and velocity from a file "init.out". File format is six columns, and N lows. First
     three columns are px, py, and pz. Second three columns are vx, vy, and vz. Each row represents particle.
   * lattice: Places particle down in a simple cubic lattice with a random perturbation. Note that `Num` must be a perfect cube if using this method.
   * random_reject: Places particles down by sampling a uniform distribution and uses a rejection radius to avoid placing particles too close together.
   * halton_reject: Places particles down according to a Halton sequence for a choice of bases in addition to using a rejection radius.
   * random: The default if no scheme is selected. Places particles down by sampling a uniform distribution. No rejection radius.
* rand_seed: Random seed for random_reject initialization scheme.
* perturb (double): perturbation for particle at lattice point when using the `lattice` initialization scheme. Must be a value between 0 and 1. If 0, particles occupy the lattice points. If 1, particles are randomly perturbed 0.5 the lattice spacing. 
* halton_bases: The bases used to define the halton sequence. See this webpage__ for more information
* Gamma (double): Coulomb coupling parameter
* Kappa (double): Electron screening parameter
* rc (double): Cutoff radius for PP interactions
* gamma (double): Magnitude of Langevin random kick
* dt (double): Timestep
* Neq (int): Number of thermostat timesteps
* Nstep (int): Number of simulation timesteps
* BC (str): Boundary conditions - so far only have periodic
* ptcls_init (deprecated): just leave as `init` and ignore
* writeout (str): yes/no. Writes to an out file where first three columns are positions, next three velocity, and last 3 acceleration.
* writexyz (str): yes/no. Writes in xyz format
* dump_step (int): Save particle information after `dump_step` steps in simulation
* random_seed (int): Random seed for Langevin kick
* restart (int): 0/1 Restart simulation from input file
* verbose (str): yes/no Output simulation data as it is running.

.. _webpage: test.com

Units
~~~~~
Currently, Sarkas uses Yukawa units to specify the system the user wants to simulate. For example,
the user might want to model strongly coupled plasmas for a specific ion species and would need to
supply the corresponding `coulomb coupling paramters`, :math:`\Gamma`, and `electron screening parameter`
:math:`\kappa`. The coulomb coupling parameter between species :math:`i` and :math:`j` is defined as

.. math::
   \Gamma_{ij} = \frac{Z_i Z_j e^2}{a_{ij} T_{ij}},

where :math:`Z_s` is the effective charge for species :math:`s`, :math:`a_{ij} = (4 \pi n/3)^{-1/3}`
is the 
ion-sphere radius, :math:`n = n_i + n_j` is the total particle number density, :math:`e` is the elementary 
charge, and :math:`T_{ij} = (T_i + T_j)/2` is the temperature of the system. 

Additionally, the non-dimensional electron screening parameter is defined as

.. math::
   \kappa = \frac{a_{ij}}{\lambda_e},

where :math:`\lambda_e` is the electron screening length defined as

.. math::
   \lambda_e^2 = \frac{\sqrt{ T_{ij} + \left(\frac{2}{3} E_F \right)^2 }}{4\pi n_e e^2}.

In the above expression, :math:`E_F` is the Fermi energy, and :math:`n_e` is the electron number density.

Running Sarkas
--------------
To run Sarkas once you have edited the yaml file, simply type the command

.. code-block:: bash
   
   $ python3 Sarkas.py yaml.inp
