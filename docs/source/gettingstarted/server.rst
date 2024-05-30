.. _server:

Host Server
========================

Cohort Builder is hosted on `sfhvcohortbuilder01` (`10.128.23.76`).

In order to use it, one needs to connect to this server first.
For having a graphical interface, the Remote Desktop Connection software
can be used. This might be helpful to change the configuration files, for instance.
However, since Cohort Builder currently does not support a graphical user interface,
in order to use the :ref:`cli`, on needs to open a terminal in the end.
The recommended way for connceting to this server is via the SSH protocol:

.. code:: bash

    ssh cohortbuilder@sfhvcohortbuilder01

This will ask for the password of the ``cohortbuilder`` user, which may be
communicated to you by other group members.

.. _ssh_with_screen:

---------------------------------------
Managing long processes with screen
---------------------------------------

When connected via SSH, the conenction has to be kept open for the whole duration
of the process, otherwise it will be killed by the Operating System.
In order to avoid this situation, it is advised to use `screen`_.


Here is a list of useful screen commands that you should be familiar with:

========================  ==============================================================================
  Command                      Description
========================  ==============================================================================
  ``$ screen``                 Cretes a new screen
  ``$ screen -ls``             Lists all the sessions
  ``$ screen -S <name>``       Create a new named session
  ``$ screen -R``              Retaches to a detached session if available and create a new one otherwise
  ``$ screen -r <name>``       Retaches to a named deattached session
========================  ==============================================================================

Once in a screen session, the following key-combinations can be used to manage the session:

========================  ==============================================================================
  Command                      Description
========================  ==============================================================================
  ``CTRL+a``, ``CTRL+d``       Detaches the current session
  ``CTRL+a``, ``"``            Show the windows in the current session
  ``CTRL+a``, ``2``            Switches to window #2 in the current session
  ``$ exit``                   Kills the current window/session
========================  ==============================================================================

.. _`screen`: https://www.gnu.org/software/screen/manual/screen.html

More examples are available on `linuxize.com <https://linuxize.com/post/how-to-use-linux-screen/>`_
and `geeksforgeeks.org <https://www.geeksforgeeks.org/screen-command-in-linux-with-examples/>`_.

---------------------------------------
Shared User
---------------------------------------

Currently, all the group members use the same local user on the server which is called
``cohortbuilder``. In the near future, all jobs will run through the ``cb-runner`` user automatically.

.. note::

    Since using the same user might lead to confusions within the group, it is recommended that
    each user creates a separate screen session with his or her name, and manages the
    processes in separate windows inside that session.
    Alternatively, consider using the server mode.
