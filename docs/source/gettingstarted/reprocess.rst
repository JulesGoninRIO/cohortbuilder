.. _reprocess:

Reprocess
=======================================

The reprocess functionality can be started by the ``reprocess-workbook`` subcommand.

.. note::
    If you used Cohort Builder for uploading the files and no unsuccessful
    processes were reported, reprocessing is not necessary.
    The :ref:`upload` functionality handles the failed processes and relaunches
    them until they are successfully processed. The only exception is when a file
    is not successfully processed after a maximum number of attempts. This number can be
    modified in :ref:`settings`.

---------------------------------------
Usage
---------------------------------------

In this section you can see a typical usage of this command.
For more detailed descriptions, please refer to :ref:`cli` or simply enter the following command:

.. code:: bash

    cb reprocess-workbook --help

The following command will detect and reprocess the failed processes of the "Acute" workbook
in the "DLMA" project on the "fhv_jugo" `Discovery instances <discovery_instances>`.

.. code:: bash

    cb reprocess-workbook -i fhv_jugo -p DLMA -w Acute
