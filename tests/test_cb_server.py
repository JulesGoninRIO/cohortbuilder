import unittest
import subprocess
import requests
from time import sleep
import warnings
import psutil
from os import getpid, devnull
from json import loads

from src.cohortbuilder.server import send_message_and_handle_reply, SERVER_ADDRESS_AS_URL, RUNPY_LOCATION, LIST_TASKS_QUERY, KILL_JOB_QUERY, change_num_threads, THREAD_LIMIT

class pre_test_server_mode(unittest.TestCase):
    def test_0_start_server(self):
        with open(devnull, 'w') as devnullfile:
            subp = subprocess.Popen(args=['./run.py', 'server'], stdout=devnullfile, stderr=devnullfile)
            sleep(8)
            self.assertIsNone(subp.poll(), 'Server is not running, when it should be!')
            subp.terminate()
            sleep(3)

class test_server_mode(unittest.TestCase):
    def __send_message_and_dont_quit(self, connection_method, url, msg):
        response = connection_method(url, data=msg)
        return response

    def setUp(self) -> None:
        self.devnullfile = open(devnull, 'w')
        self.server_subp = subprocess.Popen(
            args=['./run.py', 'server'],
            stdout=self.devnullfile,
            stderr=self.devnullfile
        )
        sleep(8) # Wait a bit for server to properly start up
        self.fake_cli_invocation = [RUNPY_LOCATION, "build", "--configs", "debug-build", "-i", "fhv_jugo", "-p", "cohortbuilder", "-w", "test_intg_small", "--threads", "5", "--noconfirm-resume", "-u", "cohortbuilder", "--client"]

    def tearDown(self) -> None:
        self.server_subp.kill()
        sleep(3) # Let server shut down properly
        self.devnullfile.close()

    def test_1_job_submission(self):
        message = '$'.join(self.fake_cli_invocation)

        response = self.__send_message_and_dont_quit(requests.post, SERVER_ADDRESS_AS_URL, message)

        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 200)

    def test_2_job_listing(self):
        response = self.__send_message_and_dont_quit(requests.get, SERVER_ADDRESS_AS_URL, LIST_TASKS_QUERY)
        prepared_dict = loads(response.reason)
        self.assertFalse(prepared_dict, 'Server just started up, no jobs should be running.')

        message = '$'.join(self.fake_cli_invocation)
        self.__send_message_and_dont_quit(requests.post, SERVER_ADDRESS_AS_URL, message) # Submit job

        response = self.__send_message_and_dont_quit(requests.get, SERVER_ADDRESS_AS_URL, LIST_TASKS_QUERY)
        self.assertIsNotNone(response)
        prepared_dict = loads(response.reason)
        prepared_dict = {'Job ID': list(prepared_dict.keys()), 'Job CLI': [' '.join(x) for x in list(prepared_dict.values())]}

        self.assertEqual(prepared_dict['Job ID'], ['1'])
        cli_no_client = self.fake_cli_invocation
        del_idx = cli_no_client.index('--client')
        del cli_no_client[del_idx]
        self.assertEqual(prepared_dict['Job CLI'], [' '.join(self.fake_cli_invocation)])
    
    def test_3_refuse_bad_jobs(self):
        bad_job_request = self.fake_cli_invocation
        bad_job_request[0] = 'RANDOMEXECUTABLE.sh'

        message = '$'.join(bad_job_request)
        response = self.__send_message_and_dont_quit(requests.post, SERVER_ADDRESS_AS_URL, message)
        self.assertIsNotNone(response)
        self.assertEqual(response.status_code, 400)

        response = self.__send_message_and_dont_quit(requests.get, SERVER_ADDRESS_AS_URL, LIST_TASKS_QUERY)
        self.assertIsNotNone(response)
        prepared_dict = loads(response.reason)
        self.assertFalse(prepared_dict, 'Job should be rejected, and not appear in job queue!')

    def test_4_limit_number_of_threads(self):
        bad_job_request = self.fake_cli_invocation
        bad_job_request = change_num_threads(bad_job_request, 1000)

        message = '$'.join(bad_job_request)
        self.__send_message_and_dont_quit(requests.post, SERVER_ADDRESS_AS_URL, message)
        # This will get accepted, but number of threads should be lowered to the limit.

        response = self.__send_message_and_dont_quit(requests.get, SERVER_ADDRESS_AS_URL, LIST_TASKS_QUERY)
        self.assertIsNotNone(response)
        prepared_dict = loads(response.reason)
        prepared_dict = {'Job ID': list(prepared_dict.keys()), 'Job CLI': [' '.join(x) for x in list(prepared_dict.values())]}

        self.assertEqual(prepared_dict['Job ID'], ['1'], 'Brand new server expected')
        self.assertNotEqual(prepared_dict['Job CLI'], [' '.join(bad_job_request)], 'Job CLI should have been modified to set threads to limit')
        self.assertIn(str(THREAD_LIMIT), prepared_dict['Job CLI'][0], 'Max number of threads should replace requested CLI if number of threads too high!')
    
    def test_5_limit_number_of_threads(self):
        job_request = self.fake_cli_invocation

        message = '$'.join(job_request)
        response = self.__send_message_and_dont_quit(requests.post, SERVER_ADDRESS_AS_URL, message) # Submit a job
        self.assertEqual(response.status_code, 200)

        response = self.__send_message_and_dont_quit(requests.get, SERVER_ADDRESS_AS_URL, LIST_TASKS_QUERY)
        self.assertIsNotNone(response)
        prepared_dict = loads(response.reason)
        prepared_dict = {'Job ID': list(prepared_dict.keys()), 'Job CLI': [' '.join(x) for x in list(prepared_dict.values())]}

        self.assertEqual(prepared_dict['Job ID'], ['1'], 'Brand new server expected')
        self.assertNotEqual(prepared_dict['Job CLI'], [' '.join(job_request)], 'Should be able to find the existing job running')

        kill_request = KILL_JOB_QUERY % 1
        response = self.__send_message_and_dont_quit(requests.get, SERVER_ADDRESS_AS_URL, kill_request)
        self.assertEqual(response.status_code, 200)

        response = self.__send_message_and_dont_quit(requests.get, SERVER_ADDRESS_AS_URL, LIST_TASKS_QUERY)
        self.assertIsNotNone(response)
        prepared_dict = loads(response.reason)
        self.assertFalse(prepared_dict, 'We expect that the job list is empty after killing the only job.')


if __name__ == "__main__":
    unittest.main()
    # Kill all subprocesses (manual clean-up afterwards)
    myPID = psutil.Process(getpid())
    for child in myPID.get_children():
        child.kill()