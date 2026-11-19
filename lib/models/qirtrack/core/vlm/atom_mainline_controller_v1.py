"""Opt-in ATOM-to-independent-trial route for the selected frame-ready mainline.

The refined ATOM box is a seed, never an automatically accepted output.  The
existing candidate Q and semantic postcheck decide whether the *trial* box is
written back.  Frozen V3 experiments continue to use their original class.
"""
import math
from .conflict_fused_controller_v3 import ConflictFusedControllerV3

class AtomMainlineControllerV1(ConflictFusedControllerV3):

    def _route(self, row, image, previous, raw, tracker, score):
        super()._route(row, image, previous, raw, tracker, score)
        if row.get('action') != 'high_yes_atom_shadow_no_op':
            return
        atom = row.get('atom_shadow') or {}
        box = atom.get('candidate_box')
        if atom.get('status') != 'ok' or not isinstance(box, (list, tuple)) or len(box) != 4:
            row['atom_mainline'] = {'status': 'no_valid_seed', 'trial_run': False}
            return
        if any((isinstance(x, bool) or not isinstance(x, (int, float)) or (not math.isfinite(x)) for x in box)) or box[2] <= 0 or box[3] <= 0:
            row['atom_mainline'] = {'status': 'no_valid_seed', 'trial_run': False}
            return
        seed = [float(x) for x in box]
        bypass = self.config.uncapped_low_q and self._q_raw < self.config.q_trigger
        row['atom_mainline'] = {'status': 'trial_pending', 'trial_run': True, 'seed_xywh_full': seed, 'seed_is_output': False, 'acceptance': 'existing_relative_not_high_no'}
        accepted = self._postcheck(image, previous, raw, seed, row, 'atom', bypass_budget=bypass)
        row['atom_mainline']['status'] = 'accepted' if accepted else 'rejected'
        row['atom_mainline']['trial_box_xywh_full'] = row.get('candidate_box')
        row['atom_mainline']['candidate_q'] = row.get('candidate_q')
        if not accepted and row['action'] == 'high_yes_atom_shadow_no_op':
            row['action'] = 'atom_rejected_no_op'
