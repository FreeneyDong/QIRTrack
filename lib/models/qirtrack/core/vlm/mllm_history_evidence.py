"""Causal history inputs on the frozen combined MLLM route, not new gates."""
from collections import deque
import hashlib
import math
from lib.models.qirtrack.core.vlm.mllm_i1_global_hint import GlobalSearchPriorityController
HISTORY_ARMS = ('recent_trusted', 'anchor_only', 'recent', 'recent_q06', 'trusted', 'recent_q06_trusted', 'countmatched_recent', 'countmatched_recent_trusted')

def select_history(past, trusted, frame, limit, include_trusted, arm):
    if arm not in HISTORY_ARMS:
        raise ValueError(arm)
    window = list(past)[-limit:] if limit else []
    if any((int(row['frame']) >= frame for row in window)):
        raise ValueError('Future history')
    filtered = []
    if arm in {'recent_q06', 'recent_q06_trusted', 'countmatched_recent', 'countmatched_recent_trusted'}:
        for row in window:
            value = float(row['q_raw'])
            if not math.isfinite(value):
                raise ValueError('Missing/invalid frozen PureIoU history Q')
            if value > 0.6:
                filtered.append(row)
    if arm in {'anchor_only', 'trusted'}:
        selected = []
    elif arm.startswith('countmatched_'):
        selected = window[-len(filtered):] if filtered else []
    elif arm in {'recent_q06', 'recent_q06_trusted'}:
        selected = filtered
    else:
        selected = window
    pool = list(trusted) if include_trusted and arm in {'recent_trusted', 'trusted', 'recent_q06_trusted', 'countmatched_recent_trusted'} else []
    if any((int(row['frame']) >= frame for row in pool)):
        raise ValueError('Future trusted evidence')
    return (selected, pool)

class HistoryEvidenceController(GlobalSearchPriorityController):

    def __init__(self, *args, history_arm, **kwargs):
        if history_arm not in HISTORY_ARMS:
            raise ValueError(history_arm)
        super().__init__(*args, **kwargs)
        self.history_arm = history_arm
        self.evidence_policy += ':history=' + history_arm

    def _evidence(self, image, previous, current, past_limit=3, include_trusted=False, search_region=None):
        (past, pool) = select_history(self.past, self.trusted, self._frame, past_limit, include_trusted, self.history_arm)
        (original_past, original_trusted) = (self.past, self.trusted)
        try:
            self.past = deque(past, maxlen=original_past.maxlen)
            self.trusted = deque(pool, maxlen=original_trusted.maxlen)
            blobs = super()._evidence(image, previous, current, len(past), bool(pool), search_region)
        finally:
            (self.past, self.trusted) = (original_past, original_trusted)
        key = tuple((hashlib.sha256(blob).hexdigest() for blob in blobs))
        self._evidence_stages[key] = 'temporal_verifier' if include_trusted or past_limit > 3 else 'periodic_audit' if self._q_raw >= self.config.q_trusted else 'verifier'
        return blobs
