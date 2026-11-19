"""Default fused history controller; V1 decision/proof logic stays unchanged."""
from collections import deque
from .semantic_codebook_controller_v1 import SemanticCodebookController
from .mllm_recovery_search_r1 import RecoverySearchController
from .fused_identity_history_v2 import FusedIdentityCodebook

class FusedHistoryController(SemanticCodebookController):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.memory is not None:
            self.memory = FusedIdentityCodebook.from_existing(self.memory)
        self.log('fused_history_default', strategy='recent4_union_representative4', max_trusted_images=6, quality_gate_unchanged=True, safe_release_required=True, performance_qualified=False)

    def _evidence(self, image, previous, current, past_limit=3, include_trusted=False, search_region=None):
        if self.memory is None:
            return RecoverySearchController._evidence(self, image, previous, current, past_limit, include_trusted, search_region)
        (vector, _) = self._encode(image, current)
        self._active_query = self.memory.query(self._frame, vector)
        refs = [] if self._admission_audit else self.memory.references(self._frame, vector)
        (old_trusted, old_past) = (self.trusted, self.past)
        frozen = self.memory.safe.core.frozen and self.cb_policy.safe_updates
        try:
            self.trusted = deque([self._item(o) for o in refs], maxlen=6)
            self.past = deque([] if self._admission_audit or frozen else [p for p in old_past if p['frame'] not in self.memory.safe.invalid_frames], maxlen=old_past.maxlen)
            limit = 0 if self._admission_audit or frozen else past_limit
            blobs = RecoverySearchController._evidence(self, image, previous, current, limit, bool(refs), search_region)
        finally:
            (self.trusted, self.past) = (old_trusted, old_past)
        self._used_refs.update((o['frame'] for o in refs))
        self.log('retrieve', stage=self._stage, frames=[o['frame'] for o in refs], capacity=6, query=self._active_query, anchor_only=frozen or self._admission_audit, history_policy='fused-v2', not_identity_ground_truth=True)
        return blobs
