import argparse

def get_parser() -> argparse.ArgumentParser:
    """
    Defines and parses the arguments passed by the command line.

    Returns:
        The namespace of the arguments.
    """

    parser = argparse.ArgumentParser(
        prog='cb',
        description='Cohort Builder command-line interface',
        epilog='Check the Cohort Builder documentation for more details.',
        allow_abbrev=False,

        )

    # Define subparsers
    subparsers = parser.add_subparsers(dest='command', required=True)
    setup = subparsers.add_parser('setup',
        help='Functionalities that are related to setting up Cohort Builder.',
        description='Functionalities that are related to setting up Cohort Builder.',
    )
    upload_dir = subparsers.add_parser('upload-dir',
        help='Upload files to Discovery',
        description='Uploads local files in a folder to a certain workbook on Discovery.'
        ' There is no filtering on the files.'
    )
    upload_pids = subparsers.add_parser('upload-pids',
        help='Upload files to Discovery',
        description='Uploads files of a list of patients to a certain workbook on Discovery.'
        ' The files are fetched from the image pools in batches.'
        ' The given patients are filtered based on consent data and consistency of their identifiers.'
        ' Optionally, the files are copied to a local folder before being uploaded.'
    )
    reprocess_workbook = subparsers.add_parser('reprocess-workbook',
        help='Relaunch the failed processes of a workbook.',
        description='Detects the datasets that have not been successfully processed in a workbook,'
        ' relaunches the failed processes, and reports a summary.'
    )
    build = subparsers.add_parser('build',
        help='Build a cohort from one or more workbooks on Discovery.',
        description='Goes through the content of the workbooks on Discovery and creates a subset'
        ' by filtering the patients, the studies, and the datasets. The subset will then be'
        ' downloaded as a cohort in a preset directory.'
        ' The filters and the type of the data can be set in a configuration file.'
    )
    server = subparsers.add_parser('server',
        help='Launch CohortBuilder in server mode. Waits for job requests.',
        description='CohortBuilder can run in server mode, where a client can communicate to a server to send job requests.'
        ' The server runs in the background at all times, and should be configured by the system / IT administrator.'
        ' If running, you can run your other commands as per usual, with the extra --client option, which will send'
        ' your job request to the server for execution!'
        ' Certain sub-commands (-l, -k, etc.) allow you to control / see what is running.'
    )

    # Add upload-dir arguments
    upload_dir.add_argument('--dir', type=str, dest='dir',
                        help='The absoulte path of the directory containing the files.'
    )
    upload_dir.add_argument('--configs', type=str, required=True, dest='configs',
                        help='The name of the configuration file inside the configs folder of the user.'
    )
    upload_dir.add_argument('--configs_dir', type=str, default=None, dest='configs_dir',
                        help='Absoulte path of the parent folder of the configs.'
                            ' If passed, it will override the default path in the settings.'
                            ' *Optional*.'
    )
    upload_dir.add_argument('-i', '--instances', type=str, nargs='+', default=['fhv_research'], dest='instances',
                        help='The name of the Discovery instance(s).'
    )
    upload_dir.add_argument('-p', '--project', type=str, required=True, dest='project',
                        help='The name of the project to look for the workbook(s).'
    )
    upload_dir.add_argument('-w', '--workbooks', type=str, nargs='+', required=True, dest='workbooks',
                        help='The name of the workbook(s).'
    )
    upload_dir.add_argument('--nowait', action='store_true', dest='nowait',
                        help='If passed, the files will be uploaded without waiting for Discovery to process them.'
                        ' This argument should not be passed if the number of the files is high.'
    )
    upload_dir.add_argument('--unblock-pending-files', action='store_true', dest='unblock_pending_files',
                            help='Whether or not to attempt to unblock files (and their acquisitions) which are stuck in the pending state.'
                            ' This works via deleting stuck files before re-uploading them. The alternative is to delete the files,'
                            ' but move them to a separate folder for another upload attempt.'
    )
    upload_dir.add_argument('--do-not-convert-to-dicom', action='store_true', dest='do_not_convert_to_dicom',
                        help='If passed, the files will not be converted to DICOM format before being uploaded.'
                        ' The files will be uploaded as they are, meaning anonymisation (and upload to anonymised instances) will not be possible.'
    )
    upload_dir.add_argument('--strong-anonymisation', action='store_true', dest='strong_anonymisation',
                        help='Perform strong anonymisation, also cloning or removing file UIDs, remove times,'
                        ' and remove information which may identify the treating physician or institution name.'
                        ' The default is to only remove patient IDs, name, make birth date generic, and remove patient sex.'
    )
    upload_dir.add_argument('--client', action='store_true', dest='client',
                        help='Send a job request to a CohortBuilder server, instead of directly running the job.'
    )

    # Add upload-pids arguments
    upload_pids.add_argument('--file', type=str, dest='pidsfile',
                        help='The path of a json file containing the list of the wanted patient identifiers.'
                        ' Compatible with --pids.'
    )
    upload_pids.add_argument('--pids', type=str, nargs='+', dest='pids', default=[],
                        help='The wanted patient identifiers.'
                        ' Compatible with --file.'
    )
    upload_pids.add_argument('--configs', type=str, required=True, dest='configs',
                        help='The name of the configuration file inside the configs folder of the user.'
    )
    upload_pids.add_argument('--configs_dir', type=str, default=None, dest='configs_dir',
                        help='Absoulte path of the parent folder of the configs.'
                            ' If passed, it will override the default path in the settings.'
                            ' *Optional*.'
    )
    upload_pids.add_argument('-i', '--instances', type=str, nargs='+', default=['fhv_research'], dest='instances',
                        help='The name of the Discovery instance(s).'
    )
    upload_pids.add_argument('-p', '--project', type=str, required=True, dest='project',
                        help='The name of the project to look for the workbook(s).'
    )
    upload_pids.add_argument('-w', '--workbooks', type=str, nargs='+', required=True, dest='workbooks',
                        help='The name of the workbook(s).'
    )
    upload_pids.add_argument('--copy', action='store_true', dest='copy',
                        help='If passed, the files will be copied to a local folder before being uploaded.'
                        ' Be sure to have enough space on the disk before passing this option.'
                        ' The files are deleted from the disk once uploaded.'
    )
    upload_pids.add_argument('--nowait', action='store_true', dest='nowait',
                        help='If passed, the files will be uploaded without waiting for Discovery to process them.'
                        ' This argument should not be passed if the number of the files is high.'
    )
    upload_pids.add_argument('--update-metadata', action='store_true', dest='updatemetadata',
                        help='If passed, the the metadata of the new files since the latest update in the image'
                        ' pools will be extracted before looking for related files.'
                        ' This will make the process slower.'
    )
    upload_pids.add_argument('--unblock-pending-files', action='store_true', dest='unblock_pending_files',
                            help='Whether or not to attempt to unblock files (and their acquisitions) which are stuck in the pending state.'
                            ' This works via deleting stuck files before re-uploading them. The alternative is to delete the files,'
                            ' but move them to a separate folder for another upload attempt.'
    )
    upload_pids.add_argument('--do-not-convert-to-dicom', action='store_true', dest='do_not_convert_to_dicom',
                        help='If passed, the files will not be converted to DICOM format before being uploaded.'
                        ' The files will be uploaded as they are, meaning anonymisation (and upload to anonymised instances) will not be possible.'
    )
    upload_pids.add_argument('--strong-anonymisation', action='store_true', dest='strong_anonymisation',
                        help='Perform strong anonymisation, also cloning or removing file UIDs, remove times,'
                        ' and remove information which may identify the treating physician or institution name.'
                        ' The default is to only remove patient IDs, name, make birth date generic, and remove patient sex.'
    )
    upload_pids.add_argument('--client', action='store_true', dest='client',
                    help='Send a job request to a CohortBuilder server, instead of directly running the job.'
    )


    # Add reprocess arguments
    reprocess_workbook.add_argument('--configs', type=str, required=True, dest='configs',
                        help='The name of the configuration file inside the configs folder of the user.'
    )
    reprocess_workbook.add_argument('--configs_dir', type=str, default=None, dest='configs_dir',
                        help='Absoulte path of the parent folder of the configs.'
                            ' If passed, it will override the default path in the settings.'
                            ' *Optional*.'
    )
    reprocess_workbook.add_argument('-i', '--instance', type=str, default='fhv_research', dest='instance',
                        help='The name of the Discovery instance that contains the workbook.'
    )
    reprocess_workbook.add_argument('-p', '--project', type=str, required=True, dest='project',
                        help='The name of the project that contains the workbook.'
    )
    reprocess_workbook.add_argument('-w', '--workbooks', type=str, required=True, dest='workbook',
                        help='The name of the workbook.'
    )
    reprocess_workbook.add_argument('--only-report', action='store_true', dest='onlyreport',
                        help='If passed, the status of the datasets will be reported but'
                        ' the failed processes will not be relaunched.'
    )
    reprocess_workbook.add_argument('--client', action='store_true', dest='client',
                    help='Send a job request to a CohortBuilder server, instead of directly running the job.')

    # Add build arguments
    build.add_argument('--configs', type=str, required=True, dest='configs',
                        help='The name of the configuration file inside the configs folder of the user.'
    )
    build.add_argument('--cohorts_dir', type=str, default=None, dest='cohorts_dir',
                        help='Absolute path of the parent folder of the cohorts.'
                            ' If passed, it will override the default path in the settings.'
                            ' *Optional.*'
    )
    build.add_argument('--configs_dir', type=str, default=None, dest='configs_dir',
                        help='Absoulte path of the parent folder of the configs.'
                            ' If passed, it will override the default path in the settings.'
                            ' *Optional*.'
    )
    build.add_argument('-i', '--instance', type=str, default='fhv_research', dest='instance',
                        help='Discovery instance that includes the source and the target workbooks.'
                            ' *Optional*.'
    )
    build.add_argument('-p', '--project', type=str, required=True, dest='project',
                        help='The name of the project to look for the workbooks.'
                        ' If there exists more than one projects with the provided name,'
                        ' the process will be aborted.'
    )
    build.add_argument('-w', '--workbooks', type=str, nargs='+', required=True, dest='workbooks',
                        help='The name of the workbook(s).'
                        ' Make sure to share the workbook(s) with Cohort Builder before running the code.'
    )
    build.add_argument('--all', action='store_true', dest='all',
                        help='If passed, the datasets that are not completely processed will be downloaded.'
    )
    build.add_argument('--client', action='store_true', dest='client',
                    help='Send a job request to a CohortBuilder server, instead of directly running the job.')
    build.add_argument('--noconfirm-resume', action='store_true', dest='noconfirm_resume',
                    help="Do not ask for a user's confirmation before resuming a build, just resume it. (Default is to ask)"
                    )

    # Add setup subcommands
    setup.add_argument('--client', action='store_true', dest='client',
                    help='Send a job request to a CohortBuilder server, instead of directly running the job.')
    setup_sub = setup.add_mutually_exclusive_group(required=True)
    setup_sub.add_argument('--extract-heyex-metadata', action='store_true', dest='extract_heyex_metadata',
                        help='Subcommand for extracting (or updating) the metadata of dicom files in Heyex image pools.',
    )
    setup_sub.add_argument('--delete-pending-datasets', action='store_true', dest='delete_pending_datasets',
                           help='Interactive mode for listing and deleting datasets which are stuck in the pending state.'
    )
    setup.add_argument('-i', '--instance', type=str, default=None, dest='instance',
                    help='Discovery instance that includes the target workbooks.'
                        ' *Optional*. Ignored in all cases, except for --delete-pending-datasets'
    )
    setup.add_argument('-p', '--project', type=str, default=None, dest='project',
                    help='The name of the project to look for the workbooks.'
                    ' If there exists more than one projects with the provided name,'
                    ' the process will be aborted. Ignored in all cases, except for --delete-pending-datasets'
    )
    setup.add_argument('-w', '--workbook', type=str, default=None, dest='workbook',
                        help='The name of the workbook.'
                        ' Make sure to share the workbook with Cohort Builder before running the code.'
                        ' Ignored in all cases, except for --delete-pending-datasets'
    )
    # setup.add_argument('--list-all', action='store_true', dest='list_all',
    #                    help="This option makes --delete-pending datasets list all datasets, even if they have not failed."
    # )
    setup.add_argument('--all', action='store_true', dest='all',
                        help='Whether or not to clear all the pending tasks, without prompting. *This is dangerous!*'
                        ' Ignored if not using --delete-pending-datasets.'
    )

    server.add_argument('-l', '--list-ongoing-jobs', action='store_true', dest='list_ongoing_jobs',
                        help='Send a query to the server to list ongoing jobs. '
                        'The response will list all current running jobs via their CLI, and associated IDs.'
                        )

    server.add_argument('-k', '--kill-job', type=int, default=0, dest='kill_job',
                        help='Send a signal to the server to terminate an ongoing job with the given ID.\n'
                        'This must be a positive, non-zero integer.'
                        )

    # Add general arguments
    for subparser in [upload_dir, upload_pids, reprocess_workbook, build]: # Do not include server!
        subparser.add_argument('-t', '--threads', type=int, default=0, dest='threads',
                            help='The number of threads that interact with Discovery\'s API.'
                            ' The total number of active threads should be kept below 80.'
        )
    for subparser in [setup, upload_dir, upload_pids, reprocess_workbook, build, server]:
        subparser.add_argument('-u', '--user', type=str, default='', dest='user',
                               help='Run cohortbuilder as this user, instead of the current user.\n'
                               'This can be helpful if running a job for someone else.\n'
                               'This setting affects the outputs of the logs and cohorts.\n'
                               "Make sure that you have write access to that user's log and cohort folders!!"
                               " If you are using the 'cb' or 'cb-dev' aliases, this option won't work, as it is already being used."
        )


    return parser