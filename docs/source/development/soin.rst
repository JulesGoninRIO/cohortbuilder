.. _setupsoin:

Setup on SOIN
======================================

There is a clone of Cohort Builder in ``/data/soin/cohortbuilder/cohortbuilder`` with
the necessary changes. The environment is available in ``/opt/miniconda3/envs/cb``
and can be activated with ``conda activate cb``.

The following settings need to be modified every time that the code is
pulled from the repository:

.. code-block:: json-object

    {
        "general": {
            "keys": "/data/soin/cohortbuilder/keys.json",
            "cache": "/data/soin/cohortbuilder/cache"
        },
        "logging": {
            "folder": "/data/soin/cohortbuilder/logs",
        }
    }

