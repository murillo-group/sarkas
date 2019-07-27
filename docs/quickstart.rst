
Quickstart
==========
Before you can begin working with Sarkas, you will need to have python3 installed. A useful python 
distribution that has many of the necessary packages for Sarkas is Anaconda which an be downloaded here_.

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
Unless adding new features to Sarkas such as new integrators, thermostats, potentials, etc, the only thing you will  need to modify is the `input.yaml` file. This file is responsible for specifying the simulation parameters such as the number of particles, number of timesteps, and initialization type for example. 

Using the input file
~~~~~~~~~~~~~~~~~~~~
To specify you simulation parameters, open the `input.yaml` file in your text editor and alter the values to
the right of the keywords. Below is a description of what each keyword is used for in Sarkas. More information on .yaml files can be found here: `https://learn.getgrav.org/16/advanced/yaml`.


.. csv-table:: Table for "Load" section key and value pairs in the input.yaml file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "species_name", "string", "Name for particle species (e.g. ion1, C, etc.)"
   "Num", "int", "Number of particles for desired species (eg. 500)"
   "method", "string", "Particle position initialization schemes"
   "rand_seed", "int", "Random seed for random_reject and lattice initialization schemes"
   "r_reject", "float", "Rejection radius for 'random_reject' and 'halton' initilization schemes. (e.g, 0.1, 1e-2, 1, etc.)"
   "perturb", "float", "Perturbation for particle at lattice point for 'lattice' initialization scheme. Must be between 0 and 1."
   "halton_bases", "python list", "List of 3 numbers to be used as the 'bases' for the 'halton_reject' initialization scheme."

.. csv-table:: Table for "Potential" section key and value pairs in the input.yaml file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "type", "string", "Name of desired potential. See <link to potentials page> for a list of supported potentials"
   "algorithm", "string", "Specify algorithm used. See <link to algorithms page> for a list of supported algorithms"
   "kappa", "float", "Electron screening length. See 'Units' section below for more detail"
   "gamma", "float", "Coulomb coupling parameter for system. See 'Units' section below for more detail"
   "rc", "float", "Potential cutoff radius. Contributions to force beyond this distance ignored"

.. csv-table:: Table for "Thermostat" section key and value pairs in the input.yaml file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "type", "string", "Name of desired thermostat to be used during equilibration phase. See <link to initilization/equlibration page> for a list of supported thermostats"

.. csv-table:: Table for "Langevin" section key and value pairs in the input.yaml file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "type", "string", "Name of desired Langevin model to be used."
   "gamma", "float", "Magnitude of Langevin 'kick'"

.. csv-table:: Table for "Control" section key and value pairs in the input.yaml file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "dt", "float", "Size of timestep used in both equilibration and production phases (e.g. 0.1)"
   "Neq", "int", "Number of equilibration steps (e.g. 1000)"
   "Nstep", "int", "Number of production steps (e.g. 5000)"
   "BC", "string", "Type of boundary conditions on all edges of simulation cell. Currently, 'periodic' is only supported boundary condition"
   "ptcls_init", "string (deprecated)", "Just leave as `init` and ignore"
   "writeout", "string", "Determines if .out file will be generated with positions, velocities, and accelerations for each particle during the extent of the simulation. Options are: 'yes or no'"
   "writexyz", "string", "Determines if .xyz file, following the 'xyz' formatting standarsds, will be generated during the extent of the simulation. Options are: 'yes or no'"
   "dump_step", "int", "Number of steps between saving particle data"
   "random_seed", "int (deprecated)", "Just leave as '1' and ignore"
   "restart", "int", "Restarts the simulation using information from a previous run or from a text file. Options: 1 (yes) or 0 (no)"
   "verbose", "string", "Writes simulation information to standard output. Options are yes or no"


* lattice: Places particle down in a simple cubic lattice with a random perturbation. Note that `Num` must be a perfect cube if using this method.
* random_reject: Places particles down by sampling a uniform distribution and uses a rejection radius to avoid placing particles too close together.
* halton_reject: Places particles down according to a Halton sequence for a choice of bases in addition to using a rejection radius.
* random: The default if no scheme is selected. Places particles down by sampling a uniform distribution. No rejection radius.


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
To run Sarkas once you have edited the input.yaml file, simply type the command

.. code-block:: bash
   
   $ python3 Sarkas.py input.yaml
