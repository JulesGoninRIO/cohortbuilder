.. _resourceusage:

Resource Utilization Estimates
======================================

:ref:`upload`
---------------------------------------

Uploading 63 files (208 MiB) from `/cohortbuilder/data/tests/upload/` to **one** instance of Discovery
(on `soinsrv01.fhv.ch`) has been performed with `upload-dir` using different options and the resource
usages are monitored.
The memory usage goes up to 600 MiB in the first 10 seconds and slowly increases afterwards.
Note that the experiments were made as the only active Discovery upload process.
Having multiple upload processes at the same time affects the total processing time of all of them.

The table below summarizes the performance for each set of options used.

+--------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
|   Command    |   Are files already processed?   |  Wait for the processings?  |   threads   |   Max. Memory usage (MiB)   |   Time (s)   |
+==============+==================================+=============================+=============+=============================+==============+
| `upload-dir` |               Yes                |              No             |      05     |           500               |      73      |
+--------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-dir` |               Yes                |              No             |      10     |           500               |      52      |
+--------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-dir` |               Yes                |              No             |      20     |           600               |      42      |
+--------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-dir` |               Yes                |              No             |      40     |           700               |      35      |
+--------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-dir` |               Yes                |              No             |      40     |           700               |      35      |
+--------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-dir` |               Yes                |              No             |      80     |           700               |      35      |
+--------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-dir` |               Yes                |              No             |      80     |           700               |      35      |
+--------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-dir` |               No                 |              Yes            |      20     |           700               |      900     |
+--------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-dir` |               No                 |              No             |      20     |           700               |      14      |
+--------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+

Uploading 1213 files (1.6 GiB), 122 of which accepted by Discovery, to **one** instance of Discovery
(on `soinsrv01.fhv.ch`) has been performed with `upload-pids` using different options and the resource
usages are monitored.
All the files belong to only one patient. With more patients, more time is needed to fetch Slims and apply patient-level filters.
The memory usage goes up to 2200 MiB in the first 80 seconds (reading the cached metadata) then drops to 700 MiB for the rest of the process.
The CPU is intermittently fully busy and idle.
Note that the experiments were made as the only active Discovery upload process.
Having multiple upload processes at the same time affects the total processing time of all of them.

The table below summarizes the performance for each set of options used.

+--------------+----------------------------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
|   Command    |   Are files already processed?   | Copy the files before uploading? |  Wait for the processings?  |   threads   |   Max. Memory usage (MiB)   |   Time (s)   |
+==============+==================================+==================================+=============================+=============+=============================+==============+
| `upload-pids`|            Yes                   |          Yes                     |           No                |      20     |         2200                |     370      |
+--------------+----------------------------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-pids`|            Yes                   |          No                      |           No                |      05     |         2200                |     600      |
+--------------+----------------------------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-pids`|            Yes                   |          No                      |           No                |      20     |         2200                |     340      |
+--------------+----------------------------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-pids`|            Yes                   |          No                      |           No                |      40     |         2200                |     290      |
+--------------+----------------------------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+
| `upload-pids`|            Yes                   |          No                      |           No                |      80     |         2200                |     270      |
+--------------+----------------------------------+----------------------------------+-----------------------------+-------------+-----------------------------+--------------+

.. note::

    Updating the image pools metadata before performing the uploads takes around 4 hours extra.


:ref:`reprocess`
---------------------------------------

Reprocessing a workbook with 8 patients and a total of 18 scans is monitored with different number of threads.
Memory usage is around 250 MiB regardless of the number of the threads used.
The CPU is fully busy during the whole process except for when it is waiting for Discovery to process scans.
The whole process takes 18s, 13s, and 9s with 5, 20, and 50 threads, respectively.

.. note::

    If the workbook contains scans that are not correctly processed, :ref:`reprocess` takes significantly more time
    depending on the number of these files.

:ref:`build`
---------------------------------------

Building a workbook with 8 patients and a total of 18 scans is monitored with different number of threads
and different configuration files.
No filters were applied and the scans are not moved to another workbook.

When parent files are downloaded, the memory usage is significantly higher.
Otherwise, it is almost constant.

The results are summarized in the table below.

+--------------+------------------+--------------------------+-----------------------------------------------+-------------+-----------------------------+--------------+
|   Command    |   Parent Files   | Images and segmentations |  Thicknesses, volumes, biomarkers, and ECRFs  |   threads   |   Max. Memory usage (MiB)   |   Time (s)   |
+==============+==================+==========================+===============================================+=============+=============================+==============+
|   `build`    |       Yes        |            Yes           |                       Yes                     |      5      |            1750             |      100     |
+--------------+------------------+--------------------------+-----------------------------------------------+-------------+-----------------------------+--------------+
|   `build`    |        No        |            Yes           |                       Yes                     |      5      |            250              |       70     |
+--------------+------------------+--------------------------+-----------------------------------------------+-------------+-----------------------------+--------------+
|   `build`    |        No        |             No           |                       Yes                     |      5      |            250              |       33     |
+--------------+------------------+--------------------------+-----------------------------------------------+-------------+-----------------------------+--------------+
|   `build`    |       Yes        |            Yes           |                       Yes                     |      20     |            1750             |      100     |
+--------------+------------------+--------------------------+-----------------------------------------------+-------------+-----------------------------+--------------+
|   `build`    |        No        |            Yes           |                       Yes                     |      20     |             270             |       65     |
+--------------+------------------+--------------------------+-----------------------------------------------+-------------+-----------------------------+--------------+
|   `build`    |        No        |             No           |                       Yes                     |      20     |             250             |       30     |
+--------------+------------------+--------------------------+-----------------------------------------------+-------------+-----------------------------+--------------+
|   `build`    |       Yes        |            Yes           |                       Yes                     |      50     |            1750             |      110     |
+--------------+------------------+--------------------------+-----------------------------------------------+-------------+-----------------------------+--------------+
|   `build`    |        No        |            Yes           |                       Yes                     |      50     |             300             |       70     |
+--------------+------------------+--------------------------+-----------------------------------------------+-------------+-----------------------------+--------------+
|   `build`    |        No        |             No           |                       Yes                     |      50     |             250             |       35     |
+--------------+------------------+--------------------------+-----------------------------------------------+-------------+-----------------------------+--------------+
