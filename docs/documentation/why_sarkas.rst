==========
Why Sarkas
==========

Problem
-------
The typical workflow of MD simulations in plasma physics looks something like this

#. Write your own MD code, or use a pre-existing code, in a low-level language such as C/C++ or (even better) Fortran to exploit their computational speed.

#. Run multiple simulations with different initial conditions.

#. Analyze the output of each simulation. This is usually done in a interpreted, high-level language like Python.

#. Make plots and publish

There are two main issues with the above workflow as it requires `i)` a high-level of computing knowledge to write/understand and run
an MD code, `ii)` a graduate level of plasma physics for calculating physical observables and transport coefficients.

Solution
--------
Sarkas: a fast pure-Python molecular dynamics suite for non-ideal plasmas.

Sarkas aims at lowering the entry barrier for computational plasma physics by providing a comprehensive MD suite complete
with pre- and post-processing tools commonly found in plasma physics.

Sarkas is entirely written in Python without calls to C hence avoiding a two-language problem. It relies on the most
common Python scientific packages, *e.g.* `NumPy <https://numpy.org/>`_, `Numba <http://numba.pydata.org/>`_,
`SciPy <https://www.scipy.org/>`_, and `Pandas <https://pandas.pydata.org/>`_, which provide a solid foundation built,
optimized, and well documented by one of the largest community of developers.
Furthermore, Sarkas is developed using an object-oriented approach allowing users to add new features
in a straight-forward way.

Sarkas targets a broad user base: from experimentalists to computational physicists, from students approaching plasma
physics for the first time to seasoned researchers. Therefore Sarkas' design revolves around two primary requirements:
ease-of-use and extensibility.


Old School
==========
First and foremost we run the help command in a terminal window

.. code-block:: bash

    $ sarkas_simulate -h

This will produce the following output

.. figure:: Help_output.png
    :alt: Figure not found

This output prints out the different options with which you can run Sarkas.

- ``-i`` or ``--input`` is required and is the path to the YAML input file of our simulation.
- ``-c`` or ``--check_status`` which can be either ``equilibration`` or ``production`` and indicates whether we want to check the equilibration or production phase of the run.
- ``-t`` or ``--pre_run_testing`` is a boolean flag indicating whether to run a test of our input parameters and estimate the simulation times.
- ``-p`` or ``--plot_show`` is a boolean flag indicating whether to show plots to screen.
- ``-v`` or ``--verbose`` boolean for verbose output.
- ``-d`` or ``--sim_dir`` name of the directory storing all the simulations.
- ``-j`` or ``--job_id`` name of the directory of the current run.
- ``-s`` or ``--seed`` sets the random number seed.
- ``-r`` or ``--restart`` for starting the simulation from a specific point.

The ``--input`` option is the only required option as it refers to the input file.
If we wanted to run multiple simulations of the same system but with different initial conditions
a typical bash script would look like this

.. code-block:: bash

    conda activate sarkas

    sarkas_simulate -i sarkas/examples/yukawa_mks_p3m.yaml -s 125125 -d run1
    sarkas_simulate -i sarkas/examples/yukawa_mks_p3m.yaml -s 281756 -d run2
    sarkas_simulate -i sarkas/examples/yukawa_mks_p3m.yaml -s 256158 -d run3
    sarkas_simulate -i sarkas/examples/yukawa_mks_p3m.yaml -s 958762 -d run4
    sarkas_simulate -i sarkas/examples/yukawa_mks_p3m.yaml -s 912856 -d run5

    conda deactivate

If you are familiar with ``bash`` scripting you could make the above statements in a loop and make many more simulations.
Once the simulations are done it's time to analyze the data. This is usually done by a python script.
This was the old way of running simulations.

Here we find the first advantage of Sarkas: removing the need to know multiple languages. Sarkas is not a Python wrapper
around an existing MD code. It is entirely written in Python to allow the user to modify the codes for their specific needs.
This choice, however, does not come at the expense of speed. In fact, Sarkas makes heavy use of ``Numpy`` and ``Numba``
packages so that the code can run as fast, if not faster, than low-level languages like ``C/C++`` and ``Fortran``.

New School
==========
Sarkas was created with the idea of incorporating the entire simulation workflow in a single Python
script. Let's say we want to run a set of ten simulations of a Yukawa OCP for different
screening parameters and measure their diffusion coefficient. An example script looks like this

.. code-block:: python

    from sarkas.processes import Simulation, PostProcess
    from sarkas.tools.observables import VelocityAutoCorrelationFunction
    from sarkas.tools.transport import TransportCoefficients
    import numpy as np
    import os

    # Path to the input file
    examples_folder = os.path.join('sarkas', 'examples')
    input_file_name = os.path.join(examples_folder,'yukawa_mks.yaml')

    # Create arrays of screening parameters
    kappas = np.linspace(1, 10)
    # Run 10 simulations
    for i, kappa in enumerate(kappas):
        # Note that we don't want to overwrite each simulation
        # So we save each simulation in its own folder by passing
        # a dictionary of dictionary with folder's name
        args = {
            "IO":
                {
                    "job_dir": "yocp_kappa{}".format(kappa)
                },
            "Potential":
                {"kappa": kappa}
        }
        # Initialize and run the simulation
        sim = Simulation(input_file_name)
        sim.setup(read_yaml=True, other_inputs=args)
        sim.run()
        # Make Temperature and Energy plots.
        postproc = PostProcess(input_file_name)
        postproc.setup(read_yaml = True, other_inputs = args)
        postproc.run()
        # Calculate the VACF
        vacf = VelocityAutoCorrelationFunction()
        vacf.setup(postproc.parameters)
        vacf.compute()
        # Calculate the diffusion coefficient
        tc = TransportCoefficients(postproc.parameters)
        tc.diffusion(vacf, plot=True)


Notice how both the simulation and the postprocessing can be done all in one script.
