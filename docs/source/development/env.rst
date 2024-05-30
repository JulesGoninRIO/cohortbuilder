Updating the Environment
===================================

-----------------------------
``environment.yml``
-----------------------------

Try to keep ``environment.yml`` updated as much as possible.

You can follow these steps for re-generating ``environment.yml``
from scratch in case you lose track of the package versions:

#. Export the environment:
.. code:: bash

    conda env export --no-builds

#. Update the versions of the old packages.

#. Add the new packages with their versions.

The environment can be built from scratch using the following command.
Make sure to not have another environment named ``cb`` before creating
this one.

.. code:: bash

    conda env create --file environment.yml

To activate and deactivate the environment:

.. code:: bash

    source /opt/miniconda3/bin/activate cb
    conda deactivate

When installing a new package, you need to run conda as root (yes, a very bad idea).
You can then fix the resulting permission problems with the ``/opt/fix-conda-perms.sh``
script, or by recursively changing ownership of everything to ``root:conda``, and permissions to ``755``.