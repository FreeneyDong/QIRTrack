# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.

"""Full 1400-sequence LaSOT main dataset loader for offline diagnostics."""
from pathlib import Path
import numpy as np
from lib.trackers.artrackv2.native.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.trackers.artrackv2.native.test.utils.load_text import load_text

class LaSOTFullDataset(BaseDataset):
    """Enumerate the 70 main LaSOT classes while excluding extension classes."""

    def __init__(self, sequence_names=None):
        super().__init__()
        self.base_path = Path(self.env_settings.lasot_path)
        all_names = self._get_sequence_list()
        if len(all_names) != 1400:
            raise RuntimeError(f'expected 1400 full-LaSOT sequences, found {len(all_names)}')
        if sequence_names is None:
            self.sequence_list = all_names
        else:
            requested = list(sequence_names)
            missing = sorted(set(requested) - set(all_names))
            if missing:
                raise KeyError(f'unknown full-LaSOT sequences: {missing[:10]}')
            self.sequence_list = requested

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(name) for name in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.rsplit('-', 1)[0]
        sequence_dir = self.base_path / class_name / sequence_name
        ground_truth = load_text(str(sequence_dir / 'groundtruth.txt'), delimiter=',', dtype=np.float64).reshape(-1, 4)
        full_occlusion = load_text(str(sequence_dir / 'full_occlusion.txt'), delimiter=',', dtype=np.float64, backend='numpy').reshape(-1)
        out_of_view = load_text(str(sequence_dir / 'out_of_view.txt'), delimiter=',', dtype=np.float64, backend='numpy').reshape(-1)
        valid_ground_truth = np.isfinite(ground_truth).all(axis=1) & (ground_truth[:, 2] > 0) & (ground_truth[:, 3] > 0)
        target_visible = (full_occlusion == 0) & (out_of_view == 0) & valid_ground_truth
        frames_path = sequence_dir / 'img'
        frames = [str(frames_path / f'{frame_number:08d}.jpg') for frame_number in range(1, len(ground_truth) + 1)]
        return Sequence(sequence_name, frames, 'lasot_full', ground_truth, object_class=class_name, target_visible=target_visible)

    def _get_sequence_list(self):
        names = []
        for category_dir in sorted(self.base_path.iterdir()):
            if not category_dir.is_dir() or category_dir.name.startswith('.'):
                continue
            category_sequences = sorted((path.name for path in category_dir.glob(f'{category_dir.name}-*') if path.is_dir() and (path / 'groundtruth.txt').is_file()))
            if len(category_sequences) == 20:
                names.extend(category_sequences)
        return names
