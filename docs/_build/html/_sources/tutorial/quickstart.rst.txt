.. _quickstart:

==========
Quickstart
==========
Once installation is complete you can start running Sarkas. This quickstart guide will walk you through
a simple example in order to check that everything is running smoothly.

In a IPython kernel you can run the command

.. code-block:: python

    %run quickstart_example

This is a Python script ``quickstart_example.py`` that was download with the repository.
This example script will run a short simulation using the ``egs_langevin.yaml`` input file
found in the examples directory in sarkas.

The output to screen should look something like this

.. figure:: quickstart_output_1.png
    :alt: Quickstart Output Part 1

The Sarkas figlet will probably look different since it is created with random font. If nothing has
changed in the input yaml file all the simulation parameters will be the same as in the figure above.
Below is the rest of the output.

.. figure:: quickstart_output_2.png
    :alt: Quickstart Output Part 2

The initialization times will likely be different than in the figure as they are hardware dependent.

The white bar at the end is a nice feature from the package ``tqdm``. It is a progress bar that indicates the current
status of the simulation. The numbers 3247 / 5000 indicate the completed timesteps and total timesteps, respectively, of
the Production phase of the simulation.

Depending on your hardware the numbers in the brackets at the end will be different. The first number, 00:53,
indicates the elapsed time since the beginning of the production loop, the second number, 00:28, indicates
the estimated total time of the production phase, and the last number indicates the number of iteration per second that
your computer is performing. Note this number change continuously since they are constantly being calculated. If you
want to know more on how they are calculated see the python package ``tqdm``.

Once the simulation is complete your output should look something like this

.. figure:: quickstart_output_3.png
    :alt: Quickstart Output Part 3