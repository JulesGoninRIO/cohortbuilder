from unittest.mock import patch
import unittest
import pathlib

from run import get_parser
from src.cohortbuilder.parser import Parser
from src.cohortbuilder.utils.helpers import read_json

class test_general(unittest.TestCase):
    def test_0_reading_cli(self):
        with patch('sys.argv', ["cb", "build", "--configs", "debug-build", "-i", "fhv_jugo", "-p", "cohortbuilder", "-w", "test_intg_all", "--threads", "5", "--noconfirm-resume", "-u", "cohortbuilder"]):
            args = get_parser().parse_args()
        
        settings: dict[str, dict] = read_json('settings.json')

        Parser.store(
            args=args,
            settings=settings
        )

        self.assertEqual(Parser.args.command, 'build')
        self.assertEqual(Parser.args.configs, 'debug-build.json')
        self.assertEqual(Parser.args.instance, 'fhv_jugo')
        self.assertEqual(Parser.args.project, "cohortbuilder")
        self.assertEqual(Parser.args.workbooks, ["test_intg_all"])
        self.assertEqual(Parser.args.threads, 5)
        self.assertEqual(Parser.args.noconfirm_resume, True)
        self.assertEqual(Parser.args.user, "cohortbuilder")

    def test_1_settings_valid(self):
        file_exists_msg = 'Path must be valid (pointing to a file that actually exists).'
        
        settings: dict[str, dict] = read_json('settings.json')

        for key, value in settings.items():
            if key in ['keys', 'cache', 'cache_large'] or '_dir' in key or '_location' in key:
                p = pathlib.Path(value)
                self.assertTrue(p.exists(), file_exists_msg)
        
        self.assertLess(settings['general']['busyhours_start'], settings['general']['busyhours_end'])
        self.assertGreaterEqual(settings['general']['busyhours_start'], 0)
        self.assertLessEqual(settings['general']['busyhours_start'], 24)
        self.assertGreaterEqual(settings['general']['busyhours_end'], 0)
        self.assertLessEqual(settings['general']['busyhours_end'], 24)

        self.assertLessEqual(settings['general']['threads'], 80)

        self.assertTrue(pathlib.Path(settings['heyex']['root']).exists(), file_exists_msg)
        self.assertTrue(pathlib.Path(settings['logging']['root']).exists(), file_exists_msg)
        
    def test_2_no_keys_access(self):
        settings: dict[str, dict] = read_json('settings.json')
        with self.assertRaises(PermissionError) as ex:
            read_json(settings['general']['keys'])
        self.assertEqual(ex.exception.errno, 13) # Permission denied
    
    def test_3_can_run_as_cb_runner(self):
        import subprocess
        from os import getlogin
        try:
            user = subprocess.run(
                args=['sudo -u cb_runner echo $USER'],
                shell=True,
                capture_output=True,
                text=True,
                check=False,
                timeout=2
            ).stdout.strip()
        except subprocess.TimeoutExpired as e:
            raise TimeoutError('Sudo rule is not in place, should not have to enter password!')
        self.assertEqual(user, getlogin()) # We expect that the current user comes back (sudo safely inherits the parent shell)


if __name__ == "__main__":
    unittest.main()