.. _reason:

==========
Why Sarkas
==========

The original workflow of MD simulation looked something like this

#. Write your own MD code in a low-level language such as ``C/C++`` or (even better) ``Fortran`` to exploit their computational speed.

#. Write input file to be read by the MD code containing all the simulation's parameters.

#. Run multiple simulations with different initial conditions. This requires a different input file for each simulation depending on the MD code.

#. Analyze the output of each simulation. This is usually done in a interpreted, high-level language like Python.

Here we find the first advantage of Sarkas: removing the need to know multiple languages. Sarkas is not a Python wrapper
around an existing MD code. It is entirely written in Python to allow the user to modify the codes for their specific needs.
This choice, however, does not come at the expense of speed. In fact, Sarkas makes heavy use of ``Numpy`` and ``Numba``
packages so that the code can run as fast, if not faster, than low-level languages like ``C/C++`` and ``Fortran``.
