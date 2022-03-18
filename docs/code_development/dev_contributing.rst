**********************
Contributing to Sarkas
**********************

The first thing to do is to make sure your distribution is up to date. Type

.. code-block:: console

    $ git status

This shows the state of your local repository. Assuming no changes have been made you will receive the following

.. code-block:: console

    Your branch is up to date with 'origin/master'.

It is good practice to create a new branch for each modification to the code. For example, let us say you want to create a new branch called ``plot_style``.
This can be done by

.. code-block:: console

    $ git checkout -b plot_style
    Switched to a new branch 'plot_style'

Now you are ready to modify the code or add files, always according to our coding guidelines. For this example, say you want to add four logo files.
Once you are done, review the changes by

.. code-block:: console

    $ git status
    On branch plot_style
    Untracked files:
      (use "git add <file>..." to include in what will be committed)
	    docs/graphics/logo/logo_green_orange_v3.png
	    docs/graphics/logo/logo_green_orange_v3.svg
	    docs/graphics/logo/logo_orange_gray_v3.png
	    docs/graphics/logo/logo_orange_gray_v3.svg

    nothing added to commit but untracked files present (use "git add" to track)

Make sure you are in the correct branch: you can see this on the first line right after ``git status``.

Add and commit your changes by

.. code-block:: console

    $ git add -A
    $ git commit -m  'Added logo files.'
    [plot_style 33654b2] Added logo files.
    4 files changed, 844 insertions(+)
    create mode 100644 docs/graphics/logo/logo_green_orange_v3.png
    create mode 100644 docs/graphics/logo/logo_green_orange_v3.svg
    create mode 100644 docs/graphics/logo/logo_orange_gray_v3.png
    create mode 100644 docs/graphics/logo/logo_orange_gray_v3.svg

The option ``-A`` indicates all files. The option ``-m`` is required and refers to the message of the commit, in this case ``Added logo files.``.
Note that this should be a short message describing what you have done.

Your changes are now only on your local repository. You can push them to GitHub via

.. code-block:: console

    $ git push
    fatal: The current branch plot_style has no upstream branch.
    To push the current branch and set the remote as upstream, use

        git push --set-upstream origin plot_style

In this case, there is no ``plot_style`` branch in your GitHub repository (``upstream``) and you need to create it. To do so, follow the given instructions.

.. code-block:: console

    $ git push --set-upstream origin plot_style
    Username for 'https://github.com': username
    Password for 'https://username@github.com':
    Enumerating objects: 1891, done.
    Counting objects: 100% (1864/1864), done.
    Delta compression using up to 4 threads
    Compressing objects: 100% (457/457), done.
    Writing objects: 100% (1616/1616), 18.07 MiB | 1.54 MiB/s, done.
    Total 1616 (delta 1159), reused 1550 (delta 1112)
    remote: Resolving deltas: 100% (1159/1159), completed with 134 local objects.
    remote:
    remote: Create a pull request for 'plot_style' on GitHub by visiting:
    remote:      https://github.com/murillo-group/sarkas/pull/new/plot_style
    remote:
    To https://github.com/murillo-group/sarkas
     * [new branch]      plot_style -> plot_style
    Branch 'plot_style' set up to track remote branch 'plot_style' from 'origin'.

The output of the ``git push`` command will be different depending on your changes.

Finally, you need to make a Pull Request (PR) for your changes to be included in the SARKAS main repository.
You can do this by following the instructions on this `link <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request>`_.
