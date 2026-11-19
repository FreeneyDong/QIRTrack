import unittest
from types import SimpleNamespace
from lib.trackers.feedback import apply_accepted_state_writeback
from lib.models.qirtrack.core.vlm.online_semantic_codebook_v1 import guarded_bbox_writeback


class FeedbackTests(unittest.TestCase):
    def test_reject_keeps_native(self):
        tracker = SimpleNamespace(state=[1., 2., 3., 4.])
        row = dict(action='global_rejected', final_box=[5., 6., 3., 4.], candidate_origin='global')
        self.assertFalse(apply_accepted_state_writeback(tracker, row, True))
        self.assertEqual(tracker.state, [1., 2., 3., 4.])

    def test_accept_changes_only_bbox(self):
        dynamic = object()
        tracker = SimpleNamespace(state=[1., 2., 3., 4.], dz_feat=dynamic)
        row = dict(action='global_accepted_output_only', final_box=[5., 6., 3., 4.], candidate_origin='global')
        self.assertTrue(apply_accepted_state_writeback(tracker, row, True))
        self.assertEqual(tracker.state, [5., 6., 3., 4.])
        self.assertIs(tracker.dz_feat, dynamic)

    def test_invalid_accept_raises(self):
        tracker = SimpleNamespace(state=[1., 2., 3., 4.])
        row = dict(action='global_accepted_output_only', final_box=[5., 6., -3., 4.], candidate_origin='global')
        with self.assertRaises(ValueError):
            apply_accepted_state_writeback(tracker, row, True)
        self.assertEqual(tracker.state, [1., 2., 3., 4.])

    def test_disabled_feedback(self):
        tracker = SimpleNamespace(state=[1., 2., 3., 4.])
        row = dict(action='global_accepted_output_only', final_box=[5., 6., 3., 4.], candidate_origin='global')
        self.assertFalse(apply_accepted_state_writeback(tracker, row, False))
        self.assertEqual(tracker.state, [1., 2., 3., 4.])


if __name__ == '__main__':
    unittest.main()
