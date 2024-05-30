.. _ci:

Continuous Integration
=============================

Using `Git hooks`_, some processes have been automated:

``pre-commit``
    This hook is invoked before ``git commit`` and includes the following processes.

    - It checks the style of the code using `Pylint`_.
    - It runs a quick integration test to check the main functionality of the code.
    - It renders the local documentation if the documentation sources are modified.

    If either of the processes fails, the commit will be aborted.

``pre-push``
    This hook is invoked before ``git push`` and includes the following processes.

    - If pushing to the main branch, it runs all the unit and integration test cases.
    - If the all the tests pass, it updates the UML diagrams (``dot`` files). If they
      are different than before, it renders them (``svg`` files) and creates a new
      commit before pushing.

    If a test fails, the push will be aborted.

``post-merge``
    This hook is invoked after ``git merge`` -if it does not fail due to conflicts-
    and ``git pull`` if the local repository is not already updated.
    It includes the following processes.

    - It renders the local documentation from the -recently updated- documentation sources.

.. _`Git hooks`: https://git-scm.com/docs/githooks
.. _`Pylint`: https://pylint.pycqa.org/en/latest/

.. note::

    To bypass the hooks, pass the ``--no-verify`` flag with the desired command.

.. warning::

    The encoding of the symbolic links in ``./.git/hooks/`` has to be UTF-8.

To activate the hooks, the first time that the repository is being set up,
the following commands can be used.

.. code:: bash

    # Link hooks to the corresponding files in the repo
    echo $'#!/bin/sh' $'\nhooks/post-merge' > .git/hooks/post-merge;
    echo $'#!/bin/sh' $'\nhooks/pre-commit' > .git/hooks/pre-commit;
    echo $'#!/bin/sh' $'\nhooks/pre-push' > .git/hooks/pre-push;
    # Make the hooks executable
    chmod u+x .git/hooks/post-merge;
    chmod u+x .git/hooks/pre-commit;
    chmod u+x .git/hooks/pre-push;
