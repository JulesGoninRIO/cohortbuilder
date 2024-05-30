Documentation
=================

- The documentation files are located in ``docs/``.

- Before starting improving the documentation, you should get familiar
  with restructured text, `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_,
  and the `autodoc extension <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_.

- You can follow the steps below to regenerate the documentaion.

    0. Make sure to activate the conda environment:

      .. code:: bash

          source /opt/miniconda3/bin/activate cb

    2. Remove the old generated files:

      .. code:: bash

          rm -rf /cohortbuilder/cohortbuilder/docs/source/api/*/generated/

    3. Clean the previous build:

      .. code:: bash

          make --directory=docs clean

    4. Build the HTML documentation:

      .. code:: bash

          make --directory=docs html

    .. note::

        The documentation shortcut in ``T:/Studies/CohortBuilder/`` opens the documentation
        from a backup of the server. In order to have an updated documentation on ``T:/Studies``,
        you need to take the steps in :ref:`serverbackup`.



Monitoring resource usages
------------------------------

To monitor the usage of CPU and memory, the following command can be used:

.. code:: bash

  cb <subcommand> <arguments> & & top -b -d 1 -n 300 -E k -S -p $! > <path-to-an-output-file>

This will write the usages with 1 second intervals for 300 seconds to a file.
This file can then be used to plot the usages.

Consider using the following code for plotting the usages:

.. code:: python

  from pathlib import Path
  import pandas as pd
  from matplotlib import pyplot as plt

  def plot_profile_output(out):

      with open(out, 'r') as f:
          lines = f.readlines()

      # Read the report lines
      reports = []
      for idx, line in enumerate(lines):
          if line.startswith('top'):
              if int(lines[idx+1].split(',')[0].split('   ')[1].split(' ')[0]) < 1:
                  continue
              reports.append(lines[idx+7])

      # Construct dataframe
      logs = []
      for r in reports:
          items = [item for item in r.split(' ') if item]
          if items[5].endswith('k'):
              res = float(items[5][:-1])
          elif items[5].endswith('m'):
              res = float(items[5][:-1]) * 1024
          elif items[5].endswith('g'):
              res = float(items[5][:-1]) * 1024 * 1024
          else:
              res = float(items[5])

          logs.append({
              'pid': int(items[0]),
              'uid': items[1],
              'res': res,
              'cpu': float(items[8]),
              'mem': float(items[9]),
          })
      logs = pd.DataFrame(logs)

      # Plot
      fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout='tight')
      logs[['cpu', 'mem']].plot(ax=axs[0])
      axs[0].set(xlabel='Time (s)', ylabel='Usage (%)');
      (logs.res / 1024).plot(ax=axs[1])
      axs[1].set(xlabel='Time (s)', ylabel='Resident Memory (MiB)');
      fig.suptitle(out.stem)
      plt.show()


Documentation TODOs
---------------------------

Here is a list of inline todos (``.. todo::``) regarding the documentation.

.. todolist::
