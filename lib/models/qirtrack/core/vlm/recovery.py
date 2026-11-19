"""R1 varies global-recovery seed/search only, not raw or local search."""
from lib.models.qirtrack.core.vlm.history_evidence import HistoryEvidenceController

class RecoverySearchController(HistoryEvidenceController):

    def __init__(self, *args, trial_search_factor, **kwargs):
        if trial_search_factor not in (3.0, 5.0):
            raise ValueError('Unregistered trial factor')
        super().__init__(*args, **kwargs)
        self.trial_search_factor = float(trial_search_factor)

    def _postcheck(self, image, previous, raw, seed, row, origin, bypass_budget=False):
        callback = self.rerun_and_score
        callback.trial_search_factor = self.trial_search_factor if origin == 'global' else 5.0
        try:
            return super()._postcheck(image, previous, raw, seed, row, origin, bypass_budget)
        finally:
            callback.trial_search_factor = 5.0
