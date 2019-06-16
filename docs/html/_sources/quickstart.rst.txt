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
~~~~~~~~~~~~~~~~~~~~
If you already have a version of python installed, you may still need to download additional packages such as
pyfftw_, fdint_, and numba_. If you have the most recent version of Anaconda installed, you should already 
have numba, installed. If you have the python package manager pip_ installed, you can install these packages,
by simply opening a terminal and running the following commands:

.. code-block:: bash

   $ pip install pyfftw

.. code-block:: bash

   $ pip install numba

.. code-block:: bash

   $ pip install fdint

.. _pyfftw: https://pypi.org/project/pyFFTW/
.. _fdint: https://pypi.org/project/fdint/
.. _numba: https://numba.pydata.org
.. _pip: https://pip.pypa.io/en/stable/

Testing Your Install
~~~~~~~~~~~~~~~~~~~~
Once you have Sarkas and all necessary packages installed, you can run a test to make sure that all packages are
installed correclty by navigating to the directory that contains Sarkas and running the command:

.. code-block:: bash
   
   $ python3 Sarkas.py test.inp

Sarkas should then begin running and you should see the following messages from your command line: <Add 
something here?>

Using Sarkas
------------

Input files
~~~~~~~~~~~
Unless adding new features to Sarkas such as new integrators, thermostats, potentials, etc, the only thing you will 
need to modify is the `yaml.inp` file. This file is responsible for specifying the simulation parameters such as
the number of particels, number of timesteps, and initialization type for example. 

Units
~~~~~
Currently, Sarkas' uses Yukawa units to specify the system the user wants to simulate. For example, the user might
want to model strongly coupled plasmas for a specific ion species and would need to supply the corresponding
`coulomb coupling paramters`, :math:`\Gamma`, and `electron screening length`, :math:`\kappa`. The coulomb coupling parameter for between species :math:`i` 
and :math:`j` is defined as

.. math::
   \Gamma_{ij} = \frac{Z_i Z_j e^2}{a_{ij} T_{ij}},

where :math:`Z_s` is the effective charge for species :math:`s`, :math:`a_{ij} = (4 \pi n/3)^{-1/3}` is the 
ion-sphere radius, :math:`n = n_i + n_j` is the total particle number density, :math:`e` is the elementary charge,
and :math:`T_{ij} = (T_i + T_j)/2` is the temperature of the system. 

Additionally, the non-dimensional electron screening length is defined as

.. math::
   \kappa = \frac{a_i}{\lambda_s},

where :math:`\lambda_s` is the screening length. In the case that the screening length is the `Debye-Huckel` 
screening length, :math:`\lambda_s = \lambda_{DH} = \sqrt{\frac{T}{4\pi n Z_i Z_j e^2}}`.

Output
~~~~~~
