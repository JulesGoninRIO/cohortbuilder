.. _faqs:

FAQs
=======================================

General
---------------------------------------

No FAQ yet.

Upload
---------------------------------------

- **I expected to see the scans of a patient but I cannot find it in the uploaded files.
  What could be the reason?**

First, make sure that the process has been carried out without any major issues
by checking the logs.
The scans pass several filters before getting uploaded to Discovery.
One possible reason could be that the patient has been filtered out because the consent
form for this patient was not available.
Control the number of patients by reading the output of the process or checking the logs.


Reprocess
---------------------------------------

No FAQ yet.

Build
---------------------------------------
- **What happens when building from a workbook with unsuccessfully processed scans?**

Only the available parts of these scans will be downloaded.
This might, for instance, end up in an empty segmentation folder in the final cohort.
The successfully processed scans are not affected by this.

- **What happens when building from a workbook with scans that are still being processed?**

Those scans will be skipped with a warning in the logs. The rest will be downloaded.

- **I have a downloaded cohort but there are some new scans in the parent
  workbook. I want to get the updates. What should I do?**

Provided that the same configurations file is used, you can get the updates
by deleting the ``.downloaded`` file in your cohort before launching the build
process again.

- **I have a cohort with only biomarkers but now I also want to get the segmentations.
  What should I do?**

You should create another cohort with only the segmentations and then merge the two cohorts.

Server
---------------------------------------
- **How do I use the server mode?**

See the :doc:`corresponding page<../gettingstarted/servermode>` for usage instructions.

- **Why should I use the server mode?**

The server mode manages multiple jobs that are submitted to it, and keeps the number of active threads below 80 at all times. It also allows you to keep runs going in the background, with minimal extra effort, and centralises information about what runs are ongoing.

- **How do I check that my job is properly running?**

Please see :doc:`this page<../gettingstarted/servermode>`, section "checking job status".

.. _fops:

Frequently Occuring Problems
---------------------------------------

- **The output of Cohort Builder is not organized**

If the width of the terminal is not large enough, the progress bars will not have
enough space and will be printed out in multiple lines.
This can be easily avoided by stretching your terminal window before running any Cohort Builder
command.

- **Process gets killed by the operating system**

This problem can be detected by seeing a "Killed" message in the output of the process.
In these situations, the process has not crashed but rather been killed by the operating system.
This issue happens when the process is using more resources than available or allowed.
Most frequently, this is due to an Out of Memory (OOM) or an Out of Storage (OOS) error.
To investigate the reason, check the logs! In almost all cases, you will have information about
when your run stopped, and what the reason was. If there is no information, then it was killed by
the operating system, for the above reasons.

- **Discovery authenticaion fails**

Sometimes, due to updates or internal Discovery problems, the API credentials used by
Cohort Builder for accessing Discovery become invalid.
When this happens, refreshing the token will fail and all queued items will fail as a consequence.
This might happen even in the middle of a Cohort Builder process and without causing crashes.
In this case, you can check the logs to see the corresponding warnings about the failed requests.
To solve this issue, you must contact RetinAI support and ask them to update the credentials
of the Cohort Builder user.

- **Slims server is not accessible**

Sometimes, the address of the SLIMS server is changed and needs to be updated in the :ref:`settings`.
Contact Marine Palaz to get updates about this issue.

- **More than 80 threads working at the same time**

The API client used by Cohort Builder can be temporarily banned by Discovery if too many requests
are sent from it within a timeframe.
To avoid this situation, it is strongly suggested to use the built-in server mode, or communicate with the other members of the group
regarding their currently running Cohort Builder processes.
Either way, the total number of active Cohort Builder threads should be kept below 80.
