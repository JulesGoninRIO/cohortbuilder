Server mode
======================================

The server mode automatically keeps the number of running threads below 80 at all times, and runs jobs in the background for you.

Using the server mode implies having both a server (in the background), and a client (you!). Once the server is running, it will stay active in the background, and you don't need to worry about it.


Check the server is running
-------------------------------------
If a systemd service is set up (the following command will give an error if it isn't), you can directly check the status with:

.. code:: bash

    systemctl status cohortbuilder-server.service


Starting the server
-------------------------------------
You can ideally start the server with:

.. code:: bash

    systemctl start cohortbuilder-server.service

Alternatively, you can use the following command:

.. code:: bash

    cb server

If you did the latter, closing your terminal with terminate the server, so consider running it in a :ref:`screen session<server>`.

Using the server to run jobs
-------------------------------------
As a user, you can send your jobs to the server very easily: all you have to do is add the ``--client`` option to your command-line. If there is a problem sending the job, the error should be quite self-explanatory.

Checking job status
-------------------------------------

Run

.. code:: bash

    systemctl status cohortbuilder-server.service

The output gives a list of all programs running as part of the service. Check that yours is there.

If you are not using a systemd service, the following still apply:

- The terminal output of all jobs is combined into the output of the server, try looking there.
- The logs of your run work as per usual. Check what is happening there.
