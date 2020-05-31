Quickstart
==========
Once installation is complete you can start running Sarkas.
In the following we will run a quick example.
Sarkas, like any other code, requires an input file containing all the simulation's parameters.
Examples of input files can be found in the `example` folder.
More information on the input files can be found below inputfile_

Running Sarkas
--------------
Once you have created your input file, say `yukawa_mks.yaml`, you can run Sarkas by simply typing the command
(in the sarkas directory)

.. warning::

    This will change once the Jupyter GUI is ready.

.. code-block:: bash

   $ python3 src/Sarkas.py examples/yukawa_mks.yaml

Depending on the option `Control: verbose` in the input file, information about the state of the simulations are printed to screen.
In this example case `Control: verbose: yes`.

Simulation's data is stored in the folder given in `Control:output_dir:` option of the input file.
In this example case `Control:output_dir:YOCP_mks_pp`. In this folder you can find a
log file, containing simulations' parameters, physical constants, and run times; a series of checkpoint files
containing particles' data needed for restarting the simulation; a file containing the radial distribution function,
and a plot of the radial distribution function saved as a `.png`.

.. _inputfile:

Input file
~~~~~~~~~~~
Let us open `ybim_mks_p3m_mag.yaml` file in a text editor. This file contains parameters for a simulation of a Carbon-Oxygen mixture interacting via a Yukawa potential and under the influence of a constant Magnetic field. The first thing to notice is that there are eight sections each of which contains a set of parameters. Each section corresponds to a subclass of the `Params` class. The order is relatively important since some section parameters might depend on a previous section. For example: the Magnetized section must come after the Integrator section since the option electrostatic_thermalization, if chosen to be True, it modifies the integrator type. Below we present a description of what each keyword is used for in Sarkas. More information on .yaml files can be found here: `https://learn.getgrav.org/16/advanced/yaml`.

.. csv-table:: Table for "Particles - species" section key and value pairs in the input file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "name", "string", "Name for particle species (e.g. ion1, C, etc.)."
   "number_density", "float", "Number density of species."
   "mass", "float", "Mass of each particle of species."
   "num", "int", "Number of simulation particles for desired species."
   "Z", "float", "Charge number of species."
   "A", "float", "Atomic mass. Note that if the keyword `mass` is present `A` would not be used for calculating the mass of each particle."
   "temperature", "float", "Desired temperature of the system."

.. csv-table:: Table for "Particles - load" section key and value pairs in the input file
   :header: "key", "Value Data Type", "Description"
   :widths: auto

   "method", "string", "Particle position initialization schemes. Options are 'random_reject, random_no_reject, restart'"
   "rand_seed", "int", "Random seed for random_reject and lattice initialization schemes"
   "r_reject", "float", "Rejection radius for 'random_reject' and 'halton' initilization schemes. (e.g, 0.1, 1e-2, 1, etc.)"
   "perturb", "float", "Perturbation for particle at lattice point for 'lattice' initialization scheme. Must be between 0 and 1."
   "halton_bases", "python list", "List of 3 numbers to be used as the 'bases' for the 'halton_reject' initialization scheme."
   "restart_step", "int", "Step number from which to restart the simulation"

.. csv-table:: Table for "Potential" section key and value pairs in the input file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "type", "string", "Name of desired potential. See <link to potentials page> for a list of supported potentials."
   "method", "string", "Specify algorithm used (P3M or PP). See <link to algorithms page> for a list of supported algorithms."
   "rc", "float", "Short-range Potential cutoff radius. Contributions to force beyond this distance are ignored."

.. csv-table:: Table for "P3M" section key and value pairs in the input file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "MGrid", "Int Array", "Number of mesh points in each of the cartesian direction [x,y,z]"
   "aliases", "int array", "Number of aliases to sum over"
   "cao", "int", "Charge order parameter aka order of the B-Spline charge approximation"
   "alpha_ewald", "float", "Alpha parameter for Ewald decomposition See <link to P3M page> for more information"

.. csv-table:: Table for "Integrator" section key and value pairs in the input file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "type", "string", "Type of integrator to be used"

.. csv-table:: Table for "Magnetized" section key and value pairs in the input file
   :header: "Key", :Value Data Type", "Description"
   :widths: auto

   "B_Gauss", "float", "Magnitude of the magnetic field in Gauss units"
   "B_Tesla", "float", "Magnitude of the magnetic field in Tesla units"
   "electrostatic_thermalization", "int", "Flag for magnetic thermalization. If 1 (True) the system will be first thermalized without magnetic field and then thermalized again with the magnetic field"
   "Neq_mag", "int", "Number of thermalization steps with a constant magnetic field"

.. csv-table:: Table for "Thermostat" section key and value pairs in the input.yaml file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "type", "string", "Name of desired thermostat to be used during equilibration phase. See <link to initilization/equlibration page> for a list of supported thermostats"
   "tau", "float", "Berendsen parameter. It should be a positive number greater than zero. See <link to Berendesen page> for more information"
   "timestep", "int", "Number of timesteps to wait before turning on the Berendsen thermostat. It should be less than the Neq"

.. csv-table:: Table for "Langevin" section key and value pairs in the input.yaml file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "type", "string", "Name of desired Langevin model to be used."
   "gamma", "float", "Magnitude of Langevin 'kick'"

.. csv-table:: Table for "Control" section key and value pairs in the input.yaml file
   :header: "Key", "Value Data Type", "Description"
   :widths: auto

   "units", "string", "Unit system to use. 'cgs' or 'mks'"
   "dt", "float", "Size of timestep used in both equilibration and production phases (e.g. 0.1)"
   "Neq", "int", "Number of equilibration steps (e.g. 1000)"
   "Nsteps", "int", "Number of production steps (e.g. 5000)"
   "BC", "string", "Type of boundary conditions on all edges of simulation cell. Currently, 'periodic' is only supported boundary condition"
   "writexyz", "string", "Determines if .xyz file, following the 'xyz' formatting standards, will be generated during the extent of the simulation. Options are: 'yes or no'"
   "dump_step", "int", "Number of steps between saving particle data"
   "random_seed", "int", "Seed of random number generator"
   "verbose", "string", "Flag for printing simulation information to screen. Options are 'yes' or 'no'"
   "output_dir", "string", "Directory where to store checkpoint files for restart and post processing."
   "fname_app", "string", "Appendix to filenames. Default = output_dir"


* lattice: Places particle down in a simple cubic lattice with a random perturbation. Note that `Num` must be a perfect cube if using this method.
* random_reject: Places particles down by sampling a uniform distribution and uses a rejection radius to avoid placing particles too close together.
* halton_reject: Places particles down according to a Halton sequence for a choice of bases in addition to using a rejection radius.
* random: The default if no scheme is selected. Places particles down by sampling a uniform distribution. No rejection radius.






