.. _settings:

Settings
=======================================

The settings of the software can be changed by changing the fields
of the ``settings.json`` file. In the following, each part of the
file is described.

The fields that are in parenthesis(e.g, ``(password)``) will be read from a *keys* file.
It contains the passwords and credentials and should not be shared
with the others.

.. note::

    Changing settings affects everyone.
    Please consult with the person responsible for Cohort Builder before changing any settings.


general
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*string* ``keys``
    The absolute path to the file containing the access credentials to the
    servers and databases.

*string* ``cache``
    The absolute path to the folder that contains the cache files.

*string* ``cache_large``
    The absolute path to the folder that contains extremely large caches.
    An example is the copy of the scans before uploading them.

*int* ``threads``
    The default number of threads for uploading/downloading files.

*int* ``upload_batch_size``
    The number the files in each batch for upload.

*int* ``reprocess_batch_size``
    The number the files in each batch for `reprocess`.

*string* ``cohorts_dir``
    The relative path to folder for storing the cohorts.

*string* ``configs_dir``
    The relative path to folder containing the :ref:`configuration files <buildconfigs>`.

*float* ``request_timeout``
    The timeout of the requests in seconds.

*int* ``upload_max_attempts``
    The maximum number of attempts for uploading a file to Discovery.

*int* ``download_max_attempts``
    The maximum number of attempts for downloading a file from Discovery.

*int* ``reprocess_max_attempts``
    The maximum number of attempts for reprocessing failed processes when uploading files to Discovery.

*int* ``token_refresh_max_attempts``
    The maximum number of attempts for refreshing the access token of Discovery.

*int* ``discovery_request_max_attempts``
    The maximum number of attempts for sending a request to Discvery.

*int* ``busyhours_start``
    The beginning time (0-24 hour) of busy hours of the image pool servers.

*int* ``busyhours_end``
    The end time (0-24 hour) of busy hours of the image pool servers.

*string* ``urm_patients_db_location``
    The location of an excel spreadsheet which holds information about prior URM patients. This is only used to check for presence of patients who may have visited before 2016, so you shouldn't have to modify it.

*string* ``taxonomy_cnn_location``
    Location of a Pytorch compiled (TorchScript) model, which classifies the modality of downloaded images from Discovery (which removes this information).

*array* ``taxonomy_cnn_input_size``
    Dimensions of image slice which the taxonomy CNN accepts as input.

progress_bar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*int* ``width``
    Total width of the progress bars (charachters).

*int* ``description``
    Width of the description of the progress bars (charachters).


logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*string* ``root``
    The absolute path to the folder of the logs.

*list* [*dict*] ``handlers``
    The list of the logging handlers.
    Refer to `loguru`_'s documentation for the content of each handler.

.. _`loguru`: https://loguru.readthedocs.io/en/stable/index.html


api
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each field contains the settings of a Discovery instance.
The name of the field will be refered to as the name of that Discovery instance.

*string* ``url``
    Access URL to the Discovery instance.

*string* ``url_dataset``
    Access URL to a dataset in the Discovery instance.
    The UUIDs of the workbook, patient, study, and the dataset
    has to be replaced by ``%s``.

*int* ``timeout``
    The expiration time of the file URLs.

*bool* ``anonymize``
    Wether or not to anonymize the files before uploading to this intance.

*string* (``login``)
    The login email of a Discovery user with necessary access.

*string* (``password``)
    The password of the Discovery user.

*string* (``client_authentication``)
    The authentication phrase of the API client.

*string* (``client_password``)
    The password of the API client.


heyex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*string* ``root``
    Mounting point of the image pools.


medisight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*string* ``server``
    Address of host server.

*string* ``database``
    Name of the database.

*string* ``driver``
    Name of the database driver.

*string* ``port``
    Connection port.


slims
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*string* ``url``
    The URL to the consent form of a patient.
    The patient identifier in the address has to be replaced by ``%s``.

*string* (``username``)
    The username of a SLIMS user.

*string* (``password``)
    The password of the SLIMS user.
