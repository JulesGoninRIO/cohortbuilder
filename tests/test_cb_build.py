import unittest
import argparse
import shutil
import pathlib
from time import sleep
from loguru import logger

from run import Builder
from src.cohortbuilder.parser import Parser
from src.cohortbuilder.utils.helpers import read_json
from tests.utils import get_hash_files, get_hash_tree, timeout

# Need to initalise Parser class here.
args = None
settings = read_json('settings.json')
configs = read_json('template-configs-build.json')
Parser.store(args=args, settings=settings)
Parser.configs = configs

#: Directory for temporary test files
CACHE_DIR = pathlib.Path(Parser.settings['general']['cache']) / 'tmp' / 'test'

#: UUID of the test target workbook
TARGET_WORKBOOK = ('test_target', 'ace23c9a-b14e-4000-a654-25ff14f81c38')

#: Flag for when the hashes should be updated
get_new_hashes = False
TEMP_DIR = pathlib.Path('/tmp')


class test_cb_builder_functionality(unittest.TestCase):
    hash_test_intg_small_everything_tree = 'c669978543283c25592119a84034be78bfce8e33be8818f8b1bad225578dbda9'
    hash_test_intg_small_everything_files = '7c4fc2ed80c1601686324468ff7258bf0f9181a7d671cc94e49cd4bc023edffd'

    hash_test_intg_all_everything_tree = 'e6d03f2e43179b615aa9116fc5152866768ae47891f7afbc1187c3b201f71485'
    hash_test_intg_all_everything_files = '525cb472fd1698cb1dca3361a3c4cf5c9e2887c2c0509f1bb51ac010a81ed5b0'

    hash_test_intg_all_children_tree = '5ab4bf875f24b9c84f19273d2169683ec24cbf9510ada62dd60b16b179c5f32c'
    hash_test_intg_all_children_files = 'd351b914535c6d92ba7faa6b9e7cea788a0e198613f863fee6b2371e94e76d9d'

    hash_test_intg_all_segmentation_tree = 'a9a5ba6a4547a9002e8ec8bfb669044f481d2f2418084afb35ddbb638e7fa942'
    hash_test_intg_all_segmentation_files = 'aefd0ed36973a0fef232c143aa6486cfed9e80763a23023df3a2311696ee1d33'

    hash_test_intg_all_h5_tree = '3bd14a91058fa05786e16bdd8ac98f7d4539ae45d5c05f46b29d52b7b426c25a'
    hash_test_intg_all_h5_files = '1565349c36fc256cb305113b1b08a20f1f0b3bfe1e409faf713bcc361ac89c85'

    hash_test_intg_all_parentfile_tree = '57aa30b931962f741cbb70d6d70a32287a6091dc41def84e4226719559bab864'
    hash_test_intg_all_parentfile_files = 'b4aaacc8ebf260b41e941c31ce41f0369749955492e386872f3ab231017acc1d'

    def build_cohort_for_test(
        self,
        name: str,
        workbook: str,
        settings: dict,
        hash_tree: str,
        hash_files: str,
        configs: str = 'everything.json',
        timelimit: float = None,
        target: bool = False,
    ):
        """Helper function for testing the build functionality of the software
            based on an specific workbook.

        Args:
            name: Name of the cohort.
            workbook: Name of the test workbook.
            settings: Dictionary of settings.
            hash_tree: Expected hash from the get_hash_tree function.
            hash_files: Expected hash from the get_hash_files function.
            configs: Name of the configs file. Defaults to 'everything.json'.
            timelimit: If given, kills the threads and resumes the downloads after this time.
            target: If ``True``, the copying to a target workobook will be tested
        """

        # Create the cache directory
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Create args
        args = argparse.Namespace(
            command='build',
            cohorts_dir=CACHE_DIR,
            configs_dir='tests/configs',
            configs=configs,
            instance='fhv_jugo',
            all=False,
            noconfirm_resume=True
        )

        # Read the keys and update the settings
        if pathlib.Path(settings['general']['keys']).exists():
            keys: dict[str, dict] = read_json(settings['general']['keys'])
            for api_name in settings['api']:
                settings['api'][api_name].update(keys['api'][api_name])
        else:
            raise ValueError('Keys file does not exist.')

        # Initializations
        logger.configure(handlers=[])
        Parser.store(
            args=args,
            settings=settings,
        )

        # Define the test workbook and the cohort path
        def get_configs():
            configs = read_json(Parser.args.configs_dir / Parser.args.configs)
            configs['general']['name'] = name
            configs['purpose'] = args.command
            if target:
                configs['general']['copy_filtered_workbook'] = TARGET_WORKBOOK[1]
            return configs

        # Delete the cohort path if it exists
        cohort = pathlib.Path(Parser.args.cohorts_dir / name)
        if cohort.exists():
            shutil.rmtree(cohort)

        # Clear the target workbook on Discovery
        if target:
            with Builder(
                configs=get_configs(),
                instance=Parser.args.instance,
                project='CohortBuilder',
                workbooks=[workbook],
                noconfirm_resume=Parser.args.noconfirm_resume
            ) as builder:
                target_wb = [w for w in builder.project.get_children() if w.attributes['name'] == TARGET_WORKBOOK[0]][0]
                target_wb.clear()

        # Define the handler to be invoked after the time limit
        def stop_downloads(builder: Builder, freq: float):
            while not builder.project.all:
                sleep(freq)
                builder.downloader.clear()

        # Build the cohort (if set, with time limit)
        with Builder(
            configs=get_configs(),
            instance=Parser.args.instance,
            project='CohortBuilder',
            workbooks=[workbook],
            noconfirm_resume=Parser.args.noconfirm_resume
        ) as builder:
            with timeout(
                time=timelimit,
                handler=stop_downloads,
                handler_kwargs={'builder': builder, 'freq': .5}
            ):
                if builder.create_folders():
                    builder.get_metadata()
                    builder.build()

        if timelimit is not None:
            # Call a second time
            with Builder(
                configs=get_configs(),
                instance=Parser.args.instance,
                project='CohortBuilder',
                workbooks=[workbook],
                noconfirm_resume=Parser.args.noconfirm_resume
            ) as builder:
                if builder.create_folders():
                    builder.get_metadata()
                    builder.build()
            # Call a third time
            with Builder(
                configs=get_configs(),
                instance=Parser.args.instance,
                project='CohortBuilder',
                workbooks=[workbook],
                noconfirm_resume=Parser.args.noconfirm_resume
            ) as builder:
                if builder.create_folders():
                    builder.get_metadata()
                    builder.build()

        if not get_new_hashes:
            # Check the local cohort
            self.assertEqual(get_hash_tree(cohort), hash_tree)
            self.assertEqual(get_hash_files(cohort), hash_files)
            # Remove the local cohort if it passes the test
            shutil.rmtree(cohort)
        else:
            # Write the hashes to a file
            with open(CACHE_DIR / 'hashes.txt', 'a') as f:
                print((name, get_hash_tree(cohort), get_hash_files(cohort)), file=f)

        # Check the target workbook in Discovery
        if target and not get_new_hashes:
            # Configure args and configs
            # Parser.args.configs = 'everything.json'
            configs = get_configs()
            configs['general']['name'] = name + '_target'
            configs['general']['copy_filtered_workbook'] = None
            # Download everything from the target cohort
            cohort = pathlib.Path(Parser.args.cohorts_dir / (name + '_target'))
            if cohort.exists():
                shutil.rmtree(cohort)
            with Builder(
                configs=configs,
                instance=Parser.args.instance,
                project='CohortBuilder',
                workbooks=[TARGET_WORKBOOK[0]],
                noconfirm_resume=Parser.args.noconfirm_resume
            ) as builder:
                if builder.create_folders():
                    builder.get_metadata()
                    builder.build()
            # Change the folder name to match with the original cohort
            _ = (cohort / TARGET_WORKBOOK[0]).rename(cohort / workbook)
            # Compare the hash of the structure tree
            self.assertEqual(get_hash_tree(cohort), hash_tree)
            shutil.rmtree(cohort)

    def test_everything_small(
        self
    ):
        """Tests the main functionality of the software
            by the test_intg_small workbook.
        """

        # Fetch the settings
        settings = read_json('settings.json')
        settings['general']['threads'] = 10


        self.build_cohort_for_test(
            name = 'test_everything_small',
            workbook='test_intg_small',
            settings=settings,
            hash_tree=test_cb_builder_functionality.hash_test_intg_small_everything_tree,
            hash_files=test_cb_builder_functionality.hash_test_intg_small_everything_files,
            target=True,
        )
    
    def test_everything_all(
        self
    ):
        """Tests the main functionality of the software
            by the test_intg_all workbook.
        """

        # Fetch the settings
        settings = read_json('settings.json')
        settings['general']['threads'] = 10

        self.build_cohort_for_test(
            name = 'test_everything_all',
            workbook='test_intg_all',
            settings=settings,
            hash_tree=test_cb_builder_functionality.hash_test_intg_all_everything_tree,
            hash_files=test_cb_builder_functionality.hash_test_intg_all_everything_files,
            target=False,
        )
    
    def test_resume_small(
        self
    ):
        """Tests if the software is handle to resume partially downloaded cohorts
            by the test_intg_small workbook.
        """

        # Skip if the hashes are not updated
        if get_new_hashes: return

        # Fetch the settings
        settings = read_json('settings.json')
        settings['general']['threads'] = 3

        self.build_cohort_for_test(
            name = 'test_resume_small',
            workbook='test_intg_small',
            settings=settings,
            hash_tree=test_cb_builder_functionality.hash_test_intg_small_everything_tree,
            hash_files=test_cb_builder_functionality.hash_test_intg_small_everything_files,
            timelimit=0,
            target=True,
        )
    
    def test_resume_all(
        self
    ):
        """Tests if the software is handle to resume partially downloaded cohorts
            by the test_intg_sll workbook.
        """

        # Skip if the hashes are not updated
        if get_new_hashes: return

        # Fetch the settings
        settings = read_json('settings.json')
        settings['general']['threads'] = 10

        self.build_cohort_for_test(
            name = 'test_resume_all',
            workbook='test_intg_all',
            settings=settings,
            hash_tree=test_cb_builder_functionality.hash_test_intg_all_everything_tree,
            hash_files=test_cb_builder_functionality.hash_test_intg_all_everything_files,
            timelimit=10,
            target=False,
        )

    def test_segmentation_all(
        self
    ):
        """Tests downloading only the segmentations
            by the test_intg_all workbook.
        """

        # Fetch the settings
        settings = read_json('settings.json')
        settings['general']['threads'] = 10

        self.build_cohort_for_test(
            name = 'test_segmentation_all',
            workbook='test_intg_all',
            settings=settings,
            hash_tree=test_cb_builder_functionality.hash_test_intg_all_segmentation_tree,
            hash_files=test_cb_builder_functionality.hash_test_intg_all_segmentation_files,
            configs='segmentation.json',
            target=True,
        )

    def test_children_all(
        self
    ):
        """Tests downloading only the segmentations
            by the test_intg_all workbook.
        """

        # Fetch the settings
        settings = read_json('settings.json')
        settings['general']['threads'] = 10

        self.build_cohort_for_test(
            name = 'test_children_all',
            workbook='test_intg_all',
            settings=settings,
            hash_tree=test_cb_builder_functionality.hash_test_intg_all_children_tree,
            hash_files=test_cb_builder_functionality.hash_test_intg_all_children_files,
            configs='children.json',
            target=True,
        )

    def test_h5_all(
        self
    ):
        """Tests downloading only the h5 files
            by the test_intg_all workbook.
        """

        # Fetch the settings
        settings = read_json('settings.json')
        settings['general']['threads'] = 10

        self.build_cohort_for_test(
            name = 'test_h5_all',
            workbook='test_intg_all',
            settings=settings,
            hash_tree=test_cb_builder_functionality.hash_test_intg_all_h5_tree,
            hash_files=test_cb_builder_functionality.hash_test_intg_all_h5_files,
            configs='h5.json',
            target=True,
        )

    def test_parentfile_all(
        self
    ):
        """Tests downloading only the h5 files
            by the test_intg_all workbook.
        """

        # Fetch the settings
        settings = read_json('settings.json')
        settings['general']['threads'] = 10

        self.build_cohort_for_test(
            name = 'test_parentfile_all',
            workbook='test_intg_all',
            settings=settings,
            hash_tree=test_cb_builder_functionality.hash_test_intg_all_parentfile_tree,
            hash_files=test_cb_builder_functionality.hash_test_intg_all_parentfile_files,
            configs='parentfile.json',
            target=False,
        )


if __name__ == '__main__':
    unittest.main()