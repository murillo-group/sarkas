.. _dev_docs:

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

Writing Documentation
---------------------
When making or adding changes to Sarkas source code an example ``.ipynb`` notebook and/or an ``.rst`` file
should be created to demonstrate how it works.
A good example are the notebooks in the `tutorial page <https://murillo-group.github.io/sarkas/tutorial/tutorial.html>`_.

You will also need to add the ``.rst`` file and/or ``.ipynb`` notebook's path in the relevant section in the
``index.rst`` file (you can find this in ``sarkas-master/docs/``).

Besides this, the functions and classes in your code must always contain **docstrings**.

Sphinx uses these docstrings to auto-generate the `API documentation <https://murillo-group.github.io/sarkas/api/modules.html>`_
for the entire Sarkas package.

.. note::

    Please make sure that you have correctly formatted the docstrings by checking how the corresponding module's API looks
    once you build the documentation.


Building Documentation
----------------------
Once you have updated the documentation we can check our changes by building it locally on your machine.
Here are the steps

    #. Make changes to the documentation by creating new ``.rst`` files or ``.ipynb`` notebooks.

    #. Enter the ``docs`` folder in your Sarkas repo. For example ``sarkas-master/docs``.

    #. Open a terminal window and run the following

        .. code-block:: bash

            $ make clean
            $ make html

        The first command removes all the content in ``_build/html``. This command is not necessary, but suggested
        so to avoid any conflicts.

        The second command updates the documentation with your changes and creates html files in the folder ``_build/html``.
        Fix any Error and/or Warning messages. You might need to run the command few times.
        You can ignore warnings about duplicate citations.

    #. Check the changes in the Documentation by opening the any of the new/updated ``.html`` files using a browser. Note you need to look at the files in ``_build/html`` as these are the new/updated ones.

    #. Now you need to copy all the content of the ``_build/html`` folder into ``docs``. On Unix system run the command

        .. code-block:: bash

            $ cp -rf _build/html/* .

        The option ``-rf`` is needed for copying folders and avoid any confirmation.
        
        On Windows run (?)

        .. code-block:: bash

            $ Xcopy /E /I _build\html\* .

        The option ``/E`` makes sure that empty subfolders are copied to the destination. 
        The option  ``/I`` avoids prompting if the destination is a folder or file.

Congratulations! The Documentation is up to date.

.. _Sphinx: https://www.sphinx-doc.org/>
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html