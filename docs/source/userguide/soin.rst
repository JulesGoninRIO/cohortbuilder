.. _soinuserguide:

Using Cohort Builder on SOIN
======================================

Using Cohort Builder on the SOIN workspace is slightly different than using it on
the :ref:`devoted host server <server>` on the FHVi network.

.. note::

    The :ref:`Discovery <discovery>` instance on the SOIN network is completely
    separate from the ones on the FHVi network.

The code is located in ``/data/soin/cohortbuilder/``.
The ``cb`` command does not work on SOIN. In all the instructions in :ref:`cli`,
this command needs to be replaced by ``/data/soin/cohortbuilder/run.py``.

Example:

.. code:: bash

    /data/soin/cohortbuilder/run.py build --help

.. warning::

    The version of Cohort Builder on SOIN might be different than the version documented
    in this documentation. Please read the documentation from the server or use the
    ``--help`` flag to see the corresponding features and arguments.

.. note::

    The :ref:`upload-pids <upload>` feature is not supported on the SOIN workspace
    since the Heyex servers are not accessible.
