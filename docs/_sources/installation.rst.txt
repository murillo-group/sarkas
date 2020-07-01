.. _install:

============
Installation
============

To install Sarkas you first need to download the source code from its GitHub repository_.

Cloning a Repository
====================
The instructions that follow are copy of those found in the GitHub help page_. If you have problem cloning the repository
please visit this help page_.

#. Navigate to the main page of Sarkas' repository_

#. Above the list of files, click on the green button **Clone**

#. At this point you can either

    * a. Click on **Download ZIP** and save it in the folder where you want to store your copy of Sarkas.
    * b. Copy the link shown

#. Depending on your choice above

        * a. Unzip the folder and enter the ``sarkas-master`` folder. Jump to point 6.
        * b. Open a terminal window and ``cd`` into the folder where you want to store your copy of Sarkas. Then type

        .. code-block:: bash

            $ git clone https://github.com/murillo-group/sarkas.git

At this point you can choose to

    #. Create a virtual environment for you work
    #. Use the docker image
    #. Manually install all the required packages

.. note::
    We strongly suggest Option 1.

Option 1: Virtual environment
=============================
It is good practice to create virtual environment for your each of your programming projects. Below are instructions
for creating the ``sarkas`` virtual environment.

#. Check if you have ``conda`` installed

    .. code-block:: bash

        $ which conda

    This command will print the path of your ``conda`` binaries. If nothing is printe then you need to install it. Visit
    Anaconda.org and download_ their Python 3.* installer.

#. Create your virtual environment via

    .. code-block:: bash

        $ conda env create -f sarkas_env.yaml

    This command will read the file (``-f`` option) ``sarkas_env.yaml`` which contains all the necessary packages for
    running Sarkas. It will create the virtual environment ``sarkas`` in the ``envs`` directory of your conda directory
    (the one printed above by the command ``which``).

#. Once the enviroment has been created you can activate it by

    .. code-block:: bash

        $ conda activate sarkas

    and deactivate it by

    .. code-block:: bash

        $ conda deactivate

Option 2: Docker Image
======================
Alternatively, you can install a whole Sarkas package including all dependencies/preliminary-packages using Docker_.
To install Sarkas using Docker, run the following commands:

.. code-block:: bash

   $ cd sarkas
   $ docker build -t sarkas -f Docker/Dockerfile .

Once you install Sarkas using Docker, you can go inside the Docker container by running the following:

.. code-block:: bash

   $ docker run -u 0 -it sarkas bash

.. _Docker: https://www.docker.com/products/docker-desktop

Option 3: Manual installation
=============================
If you don't have a version of Python 3.* installed, visit Anaconda_ and download_ their Python 3.* installer.
Then jump back to Option 1.

If you have Python 3.* installed jump back to Option 1.

.. _Anaconda: https://www.anaconda.org
.. _repository: https://github.com/murillo-group/sarkas-repo
.. _page: https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository
.. _download: https://www.anaconda.com/products/individual