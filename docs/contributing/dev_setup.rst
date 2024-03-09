*****
Setup
*****

In order to make changes to Sarkas you first need to download the source code from its GitHub repository_.
It is highly recommended to fork Sarkas repo for any code development. This is to ensure that your code changes
do not conflict with Sarkas master branch.

If you already have a copy of the repository on your computer you can update it (as detailed below) or
jump to the next section.

Updating your repository
------------------------

The first thing to do is to update your local repository with the GitHub repo.
Open a terminal (command prompt in Windows) and move to the directory where you stored your repository. Assuming you
have your ``git`` all set up, run the command

.. code-block:: console

    $ git pull

This updates your repo files with the latest from the trunk.

Alternatively you can download the zip file again and copy the extracted files into your local Sarkas repo.
However, this might cause problem later on when push committing your changes.


Forking and Cloning Sarkas repository
-------------------------------------

The instructions that follow are a copy of those found in the Github help fork_ page. If you have problems cloning
the repository please visit the fork_ page.

A fork is a copy of a repository. Forking a repository allows you to freely experiment with changes without
affecting the original project.

#. Navigate to the main page of Sarkas' repository_

#. In the top right corner of the page, under your account icon, click on **Fork**.

#. Great you have successfully forked Sarkas repository. Now we need to clone it on your local computer.

#. Navigate to your Github page and select your Sarkas repository

#. Above the list of files, click on the green button **Clone**

#. At this point you can either

    * a. Click on **Download ZIP** and save it in the folder where you want to store your copy of Sarkas.
    * b. Copy the link shown

#. Depending on your choice above

        * a. Unzip the folder and enter the ``sarkas`` folder. Jump to point 8.
        * b. Open a terminal window and ``cd`` into the folder where you want to store your copy of Sarkas. Then type

        .. code-block:: console

            $ git clone https://github.com/<yourGitHubUsername>/sarkas

#. At this point we need to create a virtual environment for you work


Create a virtual environment
----------------------------

It is good practice to create virtual environment for each of your programming projects. This way any changes made
to Sarkas or any other code will remain in this environment and will not affect your python packages.
Below are instructions for creating the ``sarkas`` virtual environment.

#. Enter the unzipped folder ``sarkas`` and open a terminal window (or command prompt in Windows).

#. Switch to the development branch with ``git checkout dev`` as it has the up-to-date (and possibly buggy) code.

#. Check if you have ``conda`` installed

    .. code-block:: console

        $ which conda

    This command will print the path of your ``conda`` binaries. If nothing is printed then you need to install it. Visit
    Anaconda.org and download_ their Python 3.* installer.

#. *(optional)* Update your ``conda`` version with

    .. code-block:: console

        $ conda update -n base conda -y

    It is possible that the above command does not update ``conda`` at all. It is a common issue: checkout `issue #9469 <https://github.com/conda/conda/issues/9469>`_ and `issue #12718 <https://github.com/conda/conda/issues/12718>`_. In this case try the following:

    .. code-block:: console

        $ conda update -n base -c defaults conda --repodata-fn=repodata.json
    
    Note that it might take a while. Once it is done check that it actually updated ``conda`` with ``conda --version``.

#. *(optional)* Install the newer and **faster** anaconda solver ``libmamba``:

    .. code-block:: console
        
        $ conda install -n base conda-libmamba-solver
        $ conda config --set solver libmamba
    
    learn more at `Conda Blog <https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community>`_.
    
#. Create your virtual environment via

    .. code-block:: console

        $ conda env create -f sarkas_env.yaml

    This command will read the file (``-f`` option) ``sarkas_env.yaml`` which contains all the necessary packages for
    running Sarkas. It will create the virtual environment ``sarkas`` in the ``envs`` directory of your conda directory
    (the one printed above by the command ``which``).

#. Once the enviroment has been created you can activate it by

    .. code-block:: console

        $ conda activate sarkas

    and deactivate it by

    .. code-block:: console

        (sarkas) $ conda deactivate


Install Sarkas in development mode
----------------------------------

Once the environment has been activated you can install Sarkas in Development mode via

.. code-block:: console

    $ pip install -e .

.. note::
    Don't forget the final dot ``.`` after ``-e`` as that is the location ``pip`` will look for a ``setup.py``

The development mode is useful so that you do not need to reinstall Sarkas everytime you change something in the source code.
In more detail, ``pip`` will create a symlink to Sarkas' files in this folder, instead of copying the source code in your python directory.

For example: If you are using Anaconda the path to the directory will look something like this
``path_to_directory/anaconda3/envs/sarkas/lib/python3.9/site-packages/``. In here you will find ``sarkas.egg-link``
if in development mode or ``sarkas-1.0.0-py3.9.egg`` if default installation. Note that the ``1-0-0-py3.9``
refers to Sarkas version and python version.

To uninstall Sarkas you can run

.. code-block:: console

    $ pip uninstall sarkas


.. _Anaconda: https://www.anaconda.org
.. _repository: https://github.com/murillo-group/sarkas-repo
.. _fork: https://docs.github.com/en/github/getting-started-with-github/fork-a-repo
.. _clone: https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository
.. _download: https://www.anaconda.com/products/individual
