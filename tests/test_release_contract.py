import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from lib.models.qirtrack.mllm import prompts

class ReleaseContractTests(unittest.TestCase):
    def test_empty_external_prompt_directory_rejected(self):
        with tempfile.TemporaryDirectory() as d, patch.object(prompts, 'ROOT', Path(d)):
            with self.assertRaises(RuntimeError):
                prompts.require_prompts()

    def test_missing_prompt_rejected(self):
        with self.assertRaises(RuntimeError):
            prompts.prompt_text(prompts.EXPECTED_KEYS[0], required=True)

    def test_unknown_prompt_key_rejected(self):
        with self.assertRaises(KeyError):
            prompts.prompt_text('../not-a-prompt')

    def test_tracker_sources_separate(self):
        root = Path(__file__).resolve().parents[1]
        for name in ['artrackv2', 'ostrack', 'sutrack']:
            self.assertTrue((root/'lib/trackers'/name/'controller.py').is_file())
            self.assertTrue((root/'lib/trackers'/name/'native').is_dir())

    def test_http_is_not_called_without_prompts(self):
        with patch('urllib.request.urlopen') as network:
            with self.assertRaises(RuntimeError):
                prompts.guarded_urlopen('http://invalid.local')
            network.assert_not_called()
