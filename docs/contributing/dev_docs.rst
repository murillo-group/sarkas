*************
Documentation
*************

High-quality and consistent documentation is very important for any coding project, but even more when distributing your
project. A well written documentation allows new users to find out how to use Sarkas as well as helps developers (like you!)
to understand the best practices.

Sarkas uses the popular Python documentation generator Sphinx_.
Sphinx translates a set of plain text source files (often written in reStructuredText_) to HTML files,
automatically producing cross-references, indices, etc.
If you haven't worked with Sphinx_ before, you should first read their
`quickstart <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_ guide.

Install Required Packages
-------------------------

Sarkas documentation build requires additional packages. Open a Terminal window and activate your sarkas environment

.. code-block:: console

    $ conda activate sarkas

Move into the root directory ``sarkas/`` directory, using ``cd`` commands, and type the following

.. code-block:: console

    (sarkas) $ conda env update -f sarkas_env.yaml

The above command will update if necessary and install the newer packages. 
If your changes require the installation of a new package or a different version of an existing package please add it to ``sarkas/sarkas_env.yaml`` and to the ``docs/requirements.txt``. Remember that ``pip`` uses two equal signs while ``conda`` uses only one.

Writing Documentation
---------------------

When making or adding changes to Sarkas source code an example ``.ipynb`` notebook and/or an ``.rst`` file should be created to demonstrate how it works.
A good example are the notebooks in the `tutorial page <../documentation/tutorial.rst>`_.

You will also need to add the ``.rst`` file and/or ``.ipynb`` notebook's path in the relevant section in the
``index.rst`` file (you can find this in ``sarkas/docs/``).

Besides this, the functions and classes in your code must always contain **docstrings**.

Sphinx uses these docstrings to auto-generate the `API documentation <../api/api.rst>`_ for the entire Sarkas package.

.. note::

    Please make sure that you have correctly formatted the docstrings by checking how the corresponding module's API looks
    once you build the documentation.

Add Notebook To The Example Gallery
-----------------------------------

Create a folder inside ``sarkas/docs/examples/`` following the structure of the other examples. The folder structure should look like the following::

    docs/
        examples/
            New_Example/
                input_files/
                    config_stuff.yaml
                CoolStuff.ipynb

Link your notebook in the file ``examples.rst`` underneath the ``.. nbgallery::`` by writing the folder name followed by the notebook file name without the extension::
    
    .. nbgallery::
        
        New_Example/CoolStuff

By default, the Sarkas logo will be used as the thumbnail of a notebook. However, if you want a particular output to be used as the thumbnail, you need to insert the following cell tag ``nbsphinx-thumbnail``.

If you do not know how to add cell tags to your Jupyter notebook, you can follow the `official guide <https://jupyterbook.org/en/stable/content/metadata.html>`_. If you use VS Code: 

- install the **Jupyter Cell Tags**
- on the top right of the selected code cell click on the ellipsis (``...``)  .
- click on *Add Cell Tag*.


Building Documentation
----------------------

Once you have updated the documentation we can check our changes by building it locally on your machine. Here are the steps:

Make sure to activate your ``sarkas`` environment: open a terminal window and type

.. code-block:: console

    $ conda activate sarkas

This is needed because when building the documentation you import the ``sarkas`` package, see the line ``import sarkas``  in the file ``conf.py``.

Then run the following

.. code-block:: console

    (sarkas) $ make clean && make html

The first command removes all the content in ``_build/html``. This command is not necessary, but suggested to avoid any conflicts. 

The second command updates the documentation with your changes and creates html files in the folder ``_build/html``.
Fix any Error and/or Warning messages. You might need to run the command few times.
You can ignore warnings about duplicate citations.

Check the changes in the Documentation by opening any of the new/updated ``.html`` files using a browser. Note you need to look at the files in ``_build/html`` as these are the new/updated ones.

Congratulations! The Documentation is up to date.

.. _Sphinx: https://www.sphinx-doc.org/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
