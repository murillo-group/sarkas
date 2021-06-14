***********
Get Started
***********

It is good practice to create virtual environment for your each of your programming projects. Below are instructions
for creating the ``sarkas`` virtual environment. Otherwise you can jump to the next section and :ref:`install <sec_installation>` Sarkas


Virtual Environment
===================

Start by checking if you have ``conda`` installed

.. code-block:: bash

    $ which conda

This command will print the path of your ``conda`` binaries. If nothing is printed then you need to install it. 
Visit `Anaconda`_ and download their Python 3.* installer.

You can create a virtual environment via

.. code-block:: bash

    $ conda create --name sarkas python=3.7 pip

This command will create the virtual environment ``sarkas`` with python 3.7 and ``pip`` installed. 
The environment can be found in the ``envs`` directory of your conda directory (the one printed above by the command ``which``). 

Once the enviroment has been created you can activate it by

.. code-block:: bash

    $ conda activate sarkas

and deactivate it by

.. code-block:: bash

    $ conda deactivate


.. _sec_installation:

Installation
============
Once the environment has been activated you can install sarkas system wide via

.. code-block:: bash

    $ pip install sarkas


External packages
-----------------

Sarkas uses two external packages: `FFTW <http://www.fftw.org/>`_ and `FMM3D <https://fmm3d.readthedocs.io/en/latest/>`_ . 
The first is used to perform fast Fourier transforms and the second to simulate systems with open boundary conditions.

FFTW3 is a very commong library in scientific computation, however, it may happen that you don't have it already installed on your computer.
In this case, follow their `instructions <http://www.fftw.org/#documentation>`_ to install it.

FMM3D is package written in Fortran to compute the potential field using the Multipole expansion. The documentation for installing it is `here <https://fmm3d.readthedocs.io/en/latest/install.html>`_.
You can find their `section <https://fmm3d.readthedocs.io/en/latest/install.html#building-python-wrappers>`_ on a Python wrapper.


Run the code
============
In the following pages you will find a quickstart notebook to check that Sarkas runs correctly and a long tutorial on how to 
setup and run simulations.

.. toctree::
    :maxdepth: 1

    Tutorial_NB/Quickstart
    tutorial

.. _Anaconda: https://www.anaconda.org
.. _repository: https://github.com/murillo-group/sarkas-repo
.. _fork: https://docs.github.com/en/github/getting-started-with-github/fork-a-repo
.. _clone: https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository
.. _download: https://www.anaconda.com/products/individual