.. _quickstart:

==========
Quickstart
==========
Once installation is complete you can start running Sarkas. In the following we will run a quick example comparing
the old and new way to run plasma MD simulations. Sarkas allows the use of both methods so that we don't loose the
advantages of each method. The two ways in which we can run a simulation are

- Old school: using a terminal window and ``bash`` scripting

- New school: using a python script/jupyter notebook

Old School
==========
The original workflow of MD simulation looked something like this

#. Write your own MD code in a low-level language such as ``C/C++`` or (even better) ``Fortran`` to exploit their computational speed.

#. Write input file to be read by the MD code containing all the simulation's parameters.

#. Run multiple simulations with different initial conditions. This requires a different input file for each simulation depending on the MD code.

#. Analyze the output of each simulation. This is usually done in a interpreted, high-level language like Python.

Here we find the first advantage of Sarkas: removing the need to know multiple languages. Sarkas is not a Python wrapper
around an existing MD code. It is entirely written in Python to allow the user to modify the codes for their specific needs.
This choice, however, does not come to at the expense of speed. In fact, Sarkas makes heavy use of Numpy and Numba
so that the code can run as fast, if not faster, than low-level languages like ``C/C++`` and ``Fortran``.

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
Once the simulations are done it's time to analyze the data. This is usually done by a python script.
This was the old way of running simulations.

New School
==========
Sarkas was created with the idea of incorporating the entire simulation
workflow in a single Python script. Let's say we want to run 10 simulation of a Yukawa OCP and measure the velocity
autocorrelation function. An example script looks like this

.. code-block:: python

    import numpy as np
    import os
    from sarkas.simulation.params import Params
    from sarkas.simulation import simulation
    from sarkas.tools.postprocessing import VelocityAutocorrelationFunction

    # Let's define some common variables
    args = dict()
    # Assuming you are inside the directory where you downloaded the sarkas repo
    args["input_file"] = os.path.join('sarkas',
                            os.path.join('examples', 'yukawa_mks_p3m.yaml') )


    # Initialize the simulation parameter class
    params = Params()

    for i in range(10):
        # Save each simulation's data into own its directory
        args["job_dir"] = "YOCP_" + str(i)
        args["job_id"] = "yocp_" + str(i)
        args["seed"] = np.random.randint(0, high=123456789)

        params.setup(args)
        simulation.run(params)
        # Initialize the VACF object
        vacf = VelocityAutocorrelationFunction(params)
        vacf.compute()


At the same time let's assume we want to run many simulations to span a range of screening parameters

.. code-block:: python

    import numpy as np
    import os
    from sarkas.simulation.params import Params
    from sarkas.simulation import simulation
    from sarkas.potentials import yukawa

    # Let's define some common variables
    args = dict()
    # Assuming you are inside the directory where you downloaded the sarkas repo
    args["input_file"] = os.path.join('sarkas',
                            os.path.join('examples', 'yukawa_mks_p3m.yaml') )

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
        # Create simulation's directories
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
