****************
Code Development
****************

The following pages present a workflow for editing, adding your own features, and contributing to Sarkas.

.. toctree::
    :maxdepth: 1

    dev_setup
    dev_report_issues
    dev_docs
    dev_contributing

Imposter Syndrome Disclaimer
----------------------------

**We want your help. No really, we do.**

There might be a little voice inside that tells you you are not ready; that you need to do one more tutorial, or learn another framework,
or write a few more blog posts before you can help me with this project.

We assure you, that is not the case.

Sarkas has some clear "Contribution Guidelines" that you can read below in the following sections.

The contribution guidelines outline the process that you will need to follow to get a patch merged. By making expectations and process explicit,
we hope it will make it easier for you to contribute.

And you do not just have to write code. You can help out by writing documentation, tests, or even by giving feedback about this work.
(And yes, that includes giving feedback about the contribution guidelines.)

(Thanks to `Adrienne Lowe <https://github.com/adriennefriend/imposter-syndrome-disclaimer>`_  who came up with this disclaimer and yeah,
made it OPEN for anyone to use!)

How can I contribute?
----------------------------

There are multiple ways in which you can help us:

- Found a bug in Sarkas? Report it to us!
- Caught a typo in documentation or want to make it better to understand? Edit it!
- Know how to fix an issue or add a new feature? Make a patch!
- Love using Sarkas? Share it with others!
- Anything else we missed?

Reporting a Bug
---------------

Sarkas is under constant development. There is no surprise that you may encounter something that does not work for your use case.
Or maybe you have some suggestions about how can we improve some functionality. Feel free to share any of it with us by
`opening an issue <https://docs.github.com/en/github/managing-your-work-on-github/creating-an-issue>`_ `here <https://github.com/murillo-group/sarkas/issues>`_.

Please make sure that you provide all the necessary information requested by prompts in the issue body - it will not only make our work easier but will also help you to communicate your problem better.

Editing the Documentation
-------------------------

There is always a scope of improvement in documentation to add some missing information or to make it easier for reading.
Here lies an opportunity for you. You can edit the documentation. This is stored in the ``docs`` directory of Sarkas.

After editing the file locally, build the docs as described in `these instructions <dev_docs>`_ and then you can submit
your changes to us by making a patch as described in the next section.

Making a Patch
--------------

If you have peeked in our codebase and realized how to fix a problem or if you know how to add a new feature, well done!
If not, do not worry - just pick an `issue <https://github.com/murillo-group/sarkas/issues>`_ and get started to fix it.

To contribute your code to Sarkas, you will need to make a `pull request <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_
from your fork of Sarkas repository. This development workflow using Git may look daunting at first, but it is not if
you follow `this guide <dev_contributing>`_ that we have prepared for you.

When you make a pull request, please provide all the necessary information requested by prompts in the pull request body.
Also, make sure that the code you are submitting always accounts for the following three:

- **Maintaining code quality:** Your code must follow the PEP8 style guide, should cover edge cases, etc.
- **Documenting the code:** You must write docstrings in functions/classes, put a relevant example in Sarkas docs and make sure docs get built correctly.
- **Testing the code:** There should be unit-tests for most of the functions/methods and they must pass our testing framework.

Spreading the word of mouth
---------------------------

If you find Sarkas helpful, you can share it with your peers, colleagues, and anyone who can benefit from Sarkas.
If you have used Sarkas in your research, please make sure to cite us.
By telling other people about how we helped you, you will help us in turn, in extending our impact.
