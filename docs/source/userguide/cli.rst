.. _cli:

Command-line Interface
========================

Cohort Builder is accessible via the ``cb`` command on the :ref:`server`.

To see the available subcommands you can enter:

.. code:: bash

    cb --help

To see the available flags for each subcommand, you can enter:

.. code:: bash

    cb <name-of-the-subcommand> --help

Example:

.. code:: bash

    cb build --help

Commands for some typical use cases are available in :ref:`gettingstarted`.

The following arguments and flags allow you to change the behavior of Cohort Builder
for each subcommand.

.. autoprogram:: run:get_parser()
    :prog: cb
    :strip_usage:
