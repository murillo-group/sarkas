.. _install:

************
Installation
************

.. warning::
    Sarkas is under heavy development and not yet available via package managers like ``pip`` and ``conda``.
    Therefore, please follow the instructions :ref:`here<dev_install>`  for installation.

Virtual environment
===================
It is good practice to create virtual environment for your each of your programming projects. Below are instructions
for creating the ``sarkas`` virtual environment.

#. Check if you have ``conda`` installed

    .. code-block:: bash

        $ which conda

    This command will print the path of your ``conda`` binaries. If nothing is printed then you need to install it. Visit
    Anaconda.org and download_ their Python 3.* installer.

#. Create your virtual environment via

    .. code-block:: bash

        $ conda env create sarkas

    This command will create the virtual environment ``sarkas`` in the ``envs`` directory of your conda directory
    (the one printed above by the command ``which``).

#. Once the enviroment has been created you can activate it by

    .. code-block:: bash

        $ conda activate sarkas

    and deactivate it by

    .. code-block:: bash

        $ conda deactivate

#. Once the environment has been activated you can install sarkas system wide via

    .. code-block:: bash

        $ pip install sarkas-md


.. _Anaconda: https://www.anaconda.org
.. _repository: https://github.com/murillo-group/sarkas-repo
.. _fork: https://docs.github.com/en/github/getting-started-with-github/fork-a-repo
.. _clone: https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository
.. _download: https://www.anaconda.com/products/individual