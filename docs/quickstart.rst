Sarkas Quickstart
=================
Sarkas is a pure python MD code designed for computational plasma physics. Before you can begin working
with Sarkas, you will need to have python installed. A useful python distribution is Anaconda that has many of the 
necessary packages for Sarkas can be downloaced here_.

.. _here: https://www.anaconda.com

Obtaining Sarkas
----------------
Sarkas is undergoing heavy development but the most recent version can be found on GitHub_, which can then be
cloned.

.. _GitHub: https://github.com/murillo-group/sarkas-repo


Preliminary Packages
--------------------
If you already have a version of python installed, you may still need to download additional packages such as
`pyfftw` and `numba`. If you have the python package manager `pip` installed, you can install these packages, 
by simply opening a terminal and running the following commands:

.. code-block:: bash

   $ pip install pyfftw

.. code-block:: bash

   $ pip install numba

Testing Your Install
--------------------
Once you have Sarkas and all necessary packages installed, you can run a test to make sure that all packages are
installed correclty by navigating to the directory that contains Sarkas and running the command:

.. code-block:: bash
   
   $ python3 Sarkas.py test.inp

Sarkas should then begin running and you should see the following messages from your command line: <Add 
something here?>

Using Sarkas
------------
To use Sarkas for 
