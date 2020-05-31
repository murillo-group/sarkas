
Installation
============
To  install Sarkas, choose one of the three methods below, then verify you have all necessary prerequisites.

Option 1: GitHub Repository
---------------------------
Sarkas is undergoing heavy development. The most recent version can be found on GitHub_.

.. _GitHub: https://github.com/murillo-group/sarkas-repo

Option 2: Pip  Install
----------------------
Instruction on how to install Sarkas using Anaconda. This will be done once setup.py is complete

Option 3: Docker Image
----------------------
Alternatively, you can install a whole Sarkas package including all dependencies/preliminary-packages using Docker_.
To install Sarkas using Docker, run the following commands: 

.. code-block:: bash

   $ git clone https://github.com/murillo-group/sarkas.git
   $ cd sarkas
   $ docker build -t sarkas -f Docker/Dockerfile .

Once you install Sarkas using Docker, you can go inside the Docker container by running the following: 

.. code-block:: bash

   $ docker run -u 0 -it sarkas bash

.. _Docker: https://www.docker.com/products/docker-desktop

Prerequisites
-------------
- Required

  + Python >= 3.6
  + Numba >= 0.49.1
  + pyfftw >= 0.12.0
  + tqdm >= 4.46.0

Before you can begin working with Sarkas, you will need to have python3 installed. A useful python 
distribution that has many of the necessary packages for Sarkas is Anaconda which can be downloaded here_.

.. _here: https://www.anaconda.com

.. note::
    We strongly recommend to install SARKAS using Anaconda since this will prevent any incompatibility between
    packages' versions.

If you already have a version of python3 installed, you may still need to download additional packages
such as pyfftw_, fdint_, tqdm_, and numba_. If you have the most recent version of Anaconda installed, you should already have numba, installed.
Sarkas requires the following packages: numba_ which allows it to run as fast (if not faster) than C++ or Fortran,
pyfftw_, to perform Fourier transforms, and fdint_ to calculate Fermi-Dirac integrals.

If you have the python package manager pip_ installed,
you can install these packages, by simply opening a terminal and running the following commands:

To install numba

.. code-block:: bash

   $ pip install numba

To install pyfftw

.. code-block:: bash

   $ conda install -c conda-forge pyfftw

To install tqdm

.. code-block:: bash

   $ conda install -c conda-forge tqdm


We highly recommend to use conda since it will install FFTW package,
if not already installed, and it can prevent compatibility errors between old versions.
To install fdint

.. code-block:: bash

   $ pip install fdint

.. _pyfftw: https://pypi.org/project/pyFFTW/
.. _fdint: https://pypi.org/project/fdint/
.. _numba: https://numba.pydata.org
.. _pip: https://pip.pypa.io/en/stable/
.. _tqdm: https://tqdm.github.io/