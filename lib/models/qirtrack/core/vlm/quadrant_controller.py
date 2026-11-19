"""Actual semantic conflicts route independently of uncalibrated codebook cosine."""
from lib.models.qirtrack.mllm.prompts import prompt_text
import math
from .history_controller import FusedHistoryController
from .online_codebook import semantic_label

class ConflictFusedControllerV3(FusedHistoryController):

    def step(self, frame, image, previous, raw, q_raw, tracker):
        if type(frame) is not int or frame <= 0 or (not math.isfinite(q_raw)):
            raise ValueError('Invalid online frame/quality')
        if self.cb_policy.route_enabled or self.cb_policy.rank_enabled:
            raise ValueError(prompt_text('conflict_fused_controller_v3_5dd73f60dc'))
        (self._frame, self._q_raw, self._current_shape) = (frame, float(q_raw), image.shape)
        self._current_image = image
        self._vectors = {}
        self._used_refs = set()
        self._alarm_this_frame = False
        self._manifests.clear()
        self._evidence_stages.clear()
        self._recent_request_ids = []
        self._stage = None
        self._raw_query = self._query(image, raw)
        row = dict(frame=frame, q_raw=q_raw, raw_box=list(raw), final_box=list(raw), candidate_box=None, triggered=False, action='no_op', variant=self.config.variant, requests_before=self.requests)
        low = q_raw < self.config.q_trigger
        periodic = not low and frame % self.config.shadow_every == 0
        audit = None
        if periodic:
            audit = self._score(image, previous, raw, 'periodic_audit', anchor_only=True)
            row['periodic_audit'] = audit
            self._alarm(self._raw_query, audit)
        actual_no = bool(audit and audit.get('status') == 'ok' and audit.get('evidence_request_id') and (semantic_label(audit, self.config.p_high) == 'high_no'))
        highq_count = getattr(self, '_highq_triggers', 0)
        blocked_highq = frame - self.last_trigger < self.config.cooldown_frames or highq_count >= self.config.max_trigger_events or self.budgeted_requests >= self.config.max_mllm_requests
        if low or actual_no:
            blocked = not low and blocked_highq or (low and (not self.config.uncapped_low_q) and (frame - self.last_trigger < self.config.cooldown_frames or self.triggers >= self.config.max_trigger_events))
            if blocked:
                row['action'] = 'highq_conflict_budget_or_cooldown' if not low else 'cooldown_or_trigger_budget'
            else:
                self.triggers += 1
                self.last_trigger = frame
                row['triggered'] = True
                if not low:
                    self._highq_triggers = highq_count + 1
                score = audit if audit is not None else self._score(image, previous, raw, 'verifier')
                row['verifier'] = score
                self._alarm(self._raw_query, score)
                self._route(row, image, previous, raw, tracker, score)
        if q_raw >= self.config.q_trusted:
            self._admission(row, image, raw)
        elif audit is not None:
            row['admission_ineligible'] = 'Q below unchanged trusted threshold'
        frozen = bool(self.memory and self.memory.safe.core.frozen)
        row['evidence_route'] = dict(version='actual-semantic-conflict-v3', quality='low' if low else 'high', semantic=semantic_label(row.get('verifier', audit or {}), self.config.p_high), semantic_observed=bool(row.get('verifier', audit or {}).get('evidence_request_id')), highq_actual_no=actual_no, audit_opportunity=periodic, cosine_used_for_acceptance_or_trigger=False, identity_error_is_hypothesis=True)
        row['codebook'] = dict(raw_query=self._raw_query, policy=__import__('dataclasses').asdict(self.cb_policy), updates_frozen=frozen, logical_mllm_requests=self.requests - row['requests_before'], writeback_guard=dict(blocked=self.cb_policy.guard_bbox_writeback and self.cb_policy.safe_updates and frozen, reason='unresolved_memory_conflict' if frozen else None))
        row['requests_after'] = self.requests
        row['evidence_request_ids'] = self._recent_request_ids[:]
        self._last_row = row
        self.log('decision', action=row['action'], triggered=row['triggered'], codebook=row['codebook'], evidence_route=row['evidence_route'])
        self._current_image = None
        return row
