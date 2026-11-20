import os
from pathlib import Path
import tempfile
import unittest

from lib.config.paths import REPOSITORY, ROOT, portable_path, resolve_config_paths


class ConfigPathTests(unittest.TestCase):
    def test_repo_path_resolves_outside_checkout_cwd(self):
        readme = ROOT / 'README.md'
        self.assertEqual(portable_path(readme), 'README.md')
        config = {'path_base': REPOSITORY, 'checkpoint': 'README.md'}
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path.cwd()
            try:
                os.chdir(tmp)
                resolved = resolve_config_paths(config, ('checkpoint',))
                self.assertEqual(Path(resolved['checkpoint']), readme)
                old = resolve_config_paths({'checkpoint': 'old.pt'}, ('checkpoint',))
                self.assertEqual(Path(old['checkpoint']), Path(tmp).resolve() / 'old.pt')
            finally:
                os.chdir(cwd)
        self.assertEqual(config['checkpoint'], 'README.md')

    def test_external_path_and_non_path_values_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            external = Path(tmp) / 'weights.pt'
            config = {
                'path_base': REPOSITORY,
                'checkpoint': portable_path(external),
                'mllm_url': 'http://127.0.0.1:8001/v1/chat/completions',
            }
            self.assertEqual(Path(config['checkpoint']), external.resolve())
            self.assertEqual(resolve_config_paths(config, ('checkpoint',)), config)
