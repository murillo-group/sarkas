.. _tutorial:

.. _Input File: input_file.rst
.. _Pre Simulation Testing: Pre_Simulation_Testing.ipynb

========
Tutorial
========

In the next pages we provide an in-depth step-by-step guide on a typical MD simulation using Sarkas.
The topics that are listed in this tutorial are as follows

- `Input File`_
- `Pre Simulation testing`_
- Running a simulation
- Post Processing

We will illustrate these steps using the example file ``yukawa_mks_p3m.yaml`` in the ``examples`` folder.
This file contains parameters for a :math:`NVE` simulation of a One component strongly coupled plasma comprised
of :math:`N = 10\, 000` hydrogen ions with a number density of :math:`n = 1.62 \times 10^{32} N/m^3`
at a temperature :math:`0.5 eV` interacting via a Yukawa potential, i.e. an YOCP.
