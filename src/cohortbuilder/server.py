from http.server import BaseHTTPRequestHandler
import requests
import sys
from pathlib import Path
from time import sleep
import subprocess
from multiprocessing import Queue
from queue import Empty
from json import dumps, loads

SERVER_ADDRESS = ('sfhvcohortbuilder01.adi.adies.lan', 11111)
SERVER_ADDRESS_AS_URL = f'http://{SERVER_ADDRESS[0]}:{SERVER_ADDRESS[1]}/'
LIST_TASKS_QUERY = 'TSK@@LIST@@RN'
KILL_JOB_QUERY = 'TSK@@KILL@@%d'
SERVER_TEXT_ENCODING = 'utf-8'

THREAD_LIMIT = 80
# Not very clean, but this is a reliable way to find the location of the main executable.
RUNPY_LOCATION = (Path(__file__).parent.parent.parent / 'run.py').as_posix()

# THIS IS SHARED AMONG THE JOB HANDLER AND REQUEST HANDLER
# tuple of [ID, list of CLI args]
job_comm_queue = Queue() # Queue[tuple[int, list]]
reverse_comm_queue = Queue() # Queue[dict]

def get_threads(x: list) -> int:
    if '--threads' in x: return int(x[x.index('--threads') + 1])
    if '-t' in x: return int(x[x.index('-t') + 1])
    return THREAD_LIMIT # No threads specified? Could be metadata cache processing, treat as worst-case scenario.

def change_num_threads(x: list, change_to: int) -> list:
    idx = None
    if '--threads' in x: idx = x.index('--threads') + 1
    if '-t' in x: idx = x.index('-t') + 1
    if idx is not None: # Otherwise can't do anything, nothing to change!
        x[idx] = str(change_to)
    return x

def remove_end_off_query(string: str) -> str:
    return '@@'.join(string.split('@@')[:-1])

# JOB ID meaning:
# < 0 = special operations: -1 -> list runs
# > 0 = submitted run: eg. {1 : cb build ...}


class CohortBuilderServerRequestHandler(BaseHTTPRequestHandler):
    '''
    This handles the incoming HTTP requests and parses them to the job handler thread.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # These are static attributes, allows for setting up class in main without having it initalise it!
    # (It is initalised at every HTTP call)
    job_communication_queue = None
    reverse_communication_queue = None
    logger = None
    max_assigned_job_id = 1

    def __send_ok_message(self, message: str):
        '''
        Helper to encode and send response with right headers
        '''
        self.send_response(200, message)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
    
    def __send_bad_message(self, message: str):
        '''
        Helper to encode and send error with right headers
        '''
        self.send_error(400, message)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
    
    def __read_incoming_message(self):
        return self.rfile.peek().decode(SERVER_TEXT_ENCODING)

    def do_POST(self):
        message = self.__read_incoming_message()
        if message:
            recieved_cli = message.split('$')
            recieved_cli.remove('--client') # We set the child process to be a "normal" CB run. Otherwise, the server would
                                            # bombard itself with its own requests!
            if recieved_cli[0] != RUNPY_LOCATION: # Only run the process if it's CohortBuilder. This gives more security.
                self.__send_bad_message(f'Invalid command-line: {recieved_cli[0]} is not CohortBuilder!')
                msg = f'Recieved invalid CLI ({recieved_cli}) from {self.client_address}.'
                self.logger.info(msg)
                print(msg)
                return # Stop here!
            
            final_msg = ''

            requested_threads = get_threads(recieved_cli)
            if requested_threads > THREAD_LIMIT:
                recieved_cli = change_num_threads(recieved_cli, THREAD_LIMIT)
                too_many_threads_msg = f"WARNING: Requested number of threads ({requested_threads}) is too high, the maximum is {THREAD_LIMIT}. Your job's thread number has been lowered to this limit. "
                final_msg += too_many_threads_msg

            CohortBuilderServerRequestHandler.job_communication_queue.put((CohortBuilderServerRequestHandler.max_assigned_job_id, recieved_cli))
            client_msg = f'Job successfully submitted! Job ID is: {CohortBuilderServerRequestHandler.max_assigned_job_id}.'
            final_msg += client_msg
            log_msg = f'Recieved and started job with assigned ID {CohortBuilderServerRequestHandler.max_assigned_job_id}, CLI: {recieved_cli}'
            CohortBuilderServerRequestHandler.logger.info(log_msg)
            print(log_msg)
            CohortBuilderServerRequestHandler.max_assigned_job_id += 1
            self.__send_ok_message(final_msg)
        return
    
    def do_GET(self):
        message = self.__read_incoming_message()
        CohortBuilderServerRequestHandler.job_communication_queue.put((-1, []))
        try:
            listed_jobs = CohortBuilderServerRequestHandler.reverse_communication_queue.get(block=True, timeout=2)
        except Empty:
            msg = 'Getting running jobs has timed out! This should not happen, contact the developer.'
            CohortBuilderServerRequestHandler.logger.error(msg)
            print(msg)
            self.__send_bad_message(msg)
            return
    
        if message == LIST_TASKS_QUERY:
            self.__send_ok_message(dumps(listed_jobs))
            return

        if remove_end_off_query(message) == remove_end_off_query(KILL_JOB_QUERY):
            wanted_dead_id = int(message.split('@@')[-1])
            if wanted_dead_id not in listed_jobs.keys():
                self.__send_bad_message(f'Provided job ID does not correspond to a running job! Provided: {wanted_dead_id}, running: {list(listed_jobs.keys())}.')
                return
            # The stop request has passed preliminary checks.
            CohortBuilderServerRequestHandler.job_communication_queue.put((-2, wanted_dead_id))
            self.__send_ok_message("Job stop request successfully processed.")
            return

def server_handle_jobs_forver(job_queue, reverse_queue, logger):

    task_queue = [] # This is not really a queue, more a list. Needs indexing for greedily saturating thread limit.
    running_tasks = []
    current_total_threads = 0

    while True:
        sleep(0.25)

        continue_after_handle_ongoing = False
        try:
            id, recieved_cli = job_queue.get_nowait()
        except Empty:
            continue_after_handle_ongoing = True # Nothing new to handle

        # Check to see if subprocesses have terminated. Report their exit status and remove them.
        tasks_to_del = []
        for i in range(len(running_tasks)):
            exit_i = running_tasks[i][1].poll()
            if exit_i is not None: # The process has terminated.
                msg = f'Job with ID {running_tasks[i][0]} and command-line "{" ".join(running_tasks[i][1].args)}" has ended, with status code {exit_i}.'
                logger.info(msg)
                print(msg)
                tasks_to_del.append(i)
                # Remove thread count from running threads
                current_total_threads -= get_threads(running_tasks[i][1].args)
            elif id == -2 and running_tasks[i][0] == recieved_cli: # If job is ongoing and should be killed
                running_tasks[i][1].terminate() # Kill the job 'nicely'
                msg = f'Job with ID {running_tasks[i][0]} and command-line "{" ".join(running_tasks[i][1].args)}" was terminated on request.'
                logger.info(msg)
                print(msg)
                tasks_to_del.append(i)
                current_total_threads -= get_threads(running_tasks[i][1].args)

        for idx in tasks_to_del[::-1]: # Reverse iterate to keep indexes valid.
            del running_tasks[idx]

        # Launch pending tasks if total n_threads <= THREAD_LIMIT
        tasks_to_del = []
        for i in range(len(task_queue)):
            asked_threads = get_threads(task_queue[i][1])
            if current_total_threads + asked_threads <= THREAD_LIMIT:
                current_total_threads += asked_threads
                tasks_to_del.append(i)
                running_tasks.append((task_queue[i][0], subprocess.Popen(task_queue[i][1])))
        
        for idx in tasks_to_del[::-1]: # Reverse iterate to keep indexes valid.
            del task_queue[idx]

        if continue_after_handle_ongoing:
            continue

        if id == -1:
            reverse_queue.put({task_id: subp.args for task_id, subp in running_tasks})
            continue # Skip to next iter, nothing else to do.

        if id > 0 and recieved_cli:
            if 'build' in recieved_cli and '--noconfirm-resume' not in recieved_cli: # Having to wait for user input defeats the whole
                recieved_cli += ['--noconfirm-resume']                               # purpose of a server (daemon) process
                msg = 'Interactive prompt for resuming a stopped build skipped. Build will resume without asking.'
                logger.warning(msg)
                print(msg)

            task_queue.append((id, recieved_cli)) # Will run at next iteration if OK.

def send_message_and_handle_reply(connection_method, url, msg):
    try:
        response = connection_method(url, data=msg)
    except requests.exceptions.ConnectionError as e:
        print('Problem communicating with server. Is it running?')
        print(e)
        sys.exit(1)

    if response.status_code != 200:
        print('ERROR, job request failed!')
        print('Server response:', response.reason)
        sys.exit(1)
    
    return response.reason

def send_job_to_server():
    cli_args = sys.argv
    cli_args[0] = RUNPY_LOCATION # This is so that the server can "sanity-check", to see if CB send the command to run.
    
    message = '$'.join(cli_args)

    response = send_message_and_handle_reply(requests.post, SERVER_ADDRESS_AS_URL, message)
    
    print('Job sent to server.')
    print('Server response:', response)
    sys.exit(0)

def list_jobs_from_server():
    from pandas import DataFrame

    response = send_message_and_handle_reply(requests.get, SERVER_ADDRESS_AS_URL, LIST_TASKS_QUERY)

    prepared_dict = loads(response)
    prepared_dict = {'Job ID': list(prepared_dict.keys()), 'Job CLI': [' '.join(x) for x in list(prepared_dict.values())]}

    df = DataFrame(prepared_dict, columns=['Job ID', 'Job CLI']) # Convert strings to double quotes, decode to python, then to DF
    print('Currently running jobs are:')
    print(df.to_string(index=False))

    sys.exit(0)

def kill_ongoing_job(job_number):
    response = send_message_and_handle_reply(requests.get, SERVER_ADDRESS_AS_URL, KILL_JOB_QUERY % job_number)

    print('Job kill request sent to server.')
    print('Server response:', response)
    sys.exit(0)