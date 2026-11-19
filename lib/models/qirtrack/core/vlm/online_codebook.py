"""Online identity prototype memory. No dataset, GT, tracker or model calls."""
from lib.models.qirtrack.mllm.prompts import prompt_text
import copy, math
from dataclasses import dataclass, asdict
import numpy as np
from .protected_history import SafeIdentityHistoryV2
from .unified_memory import Policy, digest

@dataclass(frozen=True)
class SemanticCodebookPolicy:
    memory_enabled: bool = True
    route_enabled: bool = True
    rank_enabled: bool = True
    safe_updates: bool = True
    id_high: float = 0.9
    id_low: float = 0.5
    merge_cosine: float = 0.95
    history_slots: int = 4
    candidate_top_k: int = 4
    confirmations: int = 2
    min_gap: int = 90
    max_active: int = 8
    max_pending: int = 8
    support_per_mode: int = 16
    max_reviews_per_alarm: int = 2
    resume_confirmations: int = 2
    guard_bbox_writeback: bool = True

    def __post_init__(self):
        flags = ('memory_enabled', 'route_enabled', 'rank_enabled', 'safe_updates', 'guard_bbox_writeback')
        if any((type(getattr(self, n)) is not bool for n in flags)):
            raise ValueError('Boolean switches required')
        if not all((type(x) in (int, float) and math.isfinite(x) for x in (self.id_low, self.id_high, self.merge_cosine))):
            raise ValueError('Finite similarity thresholds')
        if not -1 <= self.id_low < self.id_high <= 1 or not -1 <= self.merge_cosine <= 1:
            raise ValueError('Threshold order')
        for n in ('history_slots', 'candidate_top_k', 'confirmations', 'min_gap', 'max_active', 'max_pending', 'support_per_mode', 'max_reviews_per_alarm', 'resume_confirmations'):
            if type(getattr(self, n)) is not int or getattr(self, n) < 1:
                raise ValueError(n)
        if self.history_slots > 4 or self.confirmations < 2 or self.resume_confirmations < 2:
            raise ValueError('Unsafe capacity/confirmation')
        if self.support_per_mode < self.confirmations:
            raise ValueError('Insufficient mode support')
        if not self.memory_enabled and (self.route_enabled or self.rank_enabled):
            raise ValueError('Routing/ranking needs memory')

def unit(x):
    z = np.asarray(x, dtype=np.float64)
    if z.shape != (512,) or not np.isfinite(z).all() or np.linalg.norm(z) < 1e-12:
        raise ValueError('Invalid CLIP512 vector')
    return z / np.linalg.norm(z)

def semantic_label(score, high=0.7):
    if not isinstance(score, dict) or score.get('status') != 'ok':
        return 'uncertain'
    p = score.get('conditional_probs_uncalibrated', {})
    (y, n) = (p.get('YES'), p.get('NO'))
    if not all((type(v) in (int, float) and math.isfinite(v) and (0 <= v <= 1) for v in (y, n))):
        return 'uncertain'
    if abs(y + n - 1) > 0.0001:
        return 'uncertain'
    return 'high_yes' if y >= high else 'high_no' if n >= high else 'uncertain'

class OnlineIdentityCodebook:
    schema = 'online-identity-codebook-v1'

    def __init__(self, sequence, anchor, contract, policy):
        self.policy = policy
        self.safe = SafeIdentityHistoryV2(sequence, anchor, contract, Policy(policy.merge_cosine, recent=4, active=policy.max_active, pending=policy.max_pending, support=policy.support_per_mode, confirmations=policy.confirmations, min_gap=policy.min_gap))
        self.events = []
        self.resume_proofs = []
        self.last_frame = 0

    def log(self, action, **kw):
        r = dict(event=action, frame=self.last_frame, **kw)
        self.events.append(copy.deepcopy(r))
        return r

    def query(self, frame, vector):
        if type(frame) is not int or frame <= 0 or frame < self.last_frame:
            raise ValueError('Noncausal query')
        self.last_frame = frame
        z = None if vector is None else unit(vector)
        view = self.safe.read(frame, z, top_k=self.policy.max_active) if self.policy.safe_updates else self.safe.core.read(frame, z, top_k=self.policy.max_active)
        if not self.policy.memory_enabled or z is None:
            return dict(label='unavailable', score=None, anchor_similarity=None, matches=[], history_frames=[], frozen=self.safe.core.frozen, source='codebook_cosine_not_probability')
        candidates = []
        for p in view['prototypes']:
            candidates.append(dict(p, kind='prototype'))
        if not self.policy.safe_updates:
            candidates += [dict(record=k, frame=self.safe.core.records[k]['frame'], similarity=float(z @ unit(self.safe.core.records[k]['vector'])), kind='recent', mode=None) for k in self.safe.core.recent]
        candidates.sort(key=lambda r: (-r['similarity'], r['frame'], r['record']))
        sa = float(z @ unit(self.safe.core.anchor['vector']))
        score = max([sa] + [r['similarity'] for r in candidates])
        frozen = self.safe.core.frozen
        label = 'high' if score >= self.policy.id_high else 'low' if score < self.policy.id_low else 'uncertain'
        if frozen and self.policy.safe_updates:
            label = 'uncertain'
        return dict(label=label, score=score, anchor_similarity=sa, matches=candidates, history_frames=[r['frame'] for r in candidates], frozen=frozen, source='codebook_cosine_not_probability')

    def references(self, frame, vector, budget=None):
        q = self.query(frame, vector)
        n = self.policy.history_slots if budget is None else budget
        if not 0 <= n <= 4:
            raise ValueError('History capacity')
        if self.safe.core.frozen and self.policy.safe_updates:
            return []
        if self.policy.safe_updates:
            view = self.safe.read(frame, top_k=self.policy.max_active)
            rows = {r['frame']: r for r in view['recent']}
            for p in q['matches']:
                rows[p['frame']] = self.safe.core.records[p['record']]
        else:
            rows = {r['frame']: r for r in self.safe.core.records.values()}
        if not rows or not n:
            return []
        chosen = [max(rows)]
        for p in q['matches']:
            if len(chosen) >= n:
                break
            if p['frame'] not in chosen:
                chosen.append(p['frame'])
        for t in sorted(rows, reverse=True):
            if len(chosen) >= n:
                break
            if t not in chosen:
                chosen.append(t)
        if any((t >= frame for t in chosen)):
            raise ValueError('Current/future history')
        return [copy.deepcopy(rows[t]) for t in sorted(chosen)]

    def rank(self, frame, candidates):
        if not candidates or candidates[0]['id'] != 'raw' or len({x['id'] for x in candidates}) != len(candidates):
            raise ValueError('Unique candidates, raw first')
        rows = []
        for (position, c) in enumerate(candidates):
            q = self.query(frame, c.get('vector'))
            rows.append(dict(id=c['id'], position=position, score=q['score']))
        if not self.policy.rank_enabled or not self.policy.memory_enabled or self.safe.core.frozen:
            selected = list(range(len(candidates)))
        else:
            rest = sorted(rows[1:], key=lambda r: (r['score'] is None, -(r['score'] or 0), r['position']))
            selected = [0] + [r['position'] for r in rest[:max(0, self.policy.candidate_top_k - 1)]]
        return dict(rows=rows, selected_positions=selected, retained_ids=[candidates[i]['id'] for i in selected], dropped_ids=[c['id'] for (i, c) in enumerate(candidates) if i not in selected], raw_fallback_preserved=True)

    def freeze(self, reason):
        if not self.policy.safe_updates:
            return self.log('alarm_diagnostic_only', reason=reason)
        c = self.safe.core
        c.frozen = True
        c._event('online_conflict_freeze', reason=reason)
        self.safe.releases[c.version] = sorted(self.safe.released)
        self.safe.releases = {v: r for (v, r) in self.safe.releases.items() if v in c.snapshots}
        self.resume_proofs = []
        return self.log('updates_frozen', reason=reason)

    def reject_support(self, frames, reason, request_ids):
        if not self.policy.safe_updates:
            return self.log('rejected_support_diagnostic_only', frames=frames)
        if not request_ids or any((not r for r in request_ids)):
            raise ValueError('Actual external review proof required')
        event = self.safe.invalidate(frames, reason)
        invalid = set(self.safe.invalid_frames)
        core = self.safe.core
        clean = []
        for (v, s) in core.snapshots.items():
            if all((o['frame'] not in invalid and (not set(o.get('proof_references', [])) & invalid) for o in s['records'].values())):
                clean.append(v)
        version = max(clean, default=0)
        self.safe.rollback(version)
        self.resume_proofs = []
        return self.log('invalidated_and_rolled_back', invalid_frames=event['frames'], restored_version=version, review_request_ids=list(request_ids), bbox_tracker_rollback=False)

    def admit(self, observation, evidence):
        if evidence.get('source') != 'actual_mllm' or semantic_label(evidence.get('score')) != 'high_yes':
            return self.log('admission_blocked_no_actual_high_yes')
        if observation['frame'] % 90 or evidence.get('stage') != 'periodic_audit' or evidence.get('q_raw', 0) < 0.8:
            return self.log('admission_blocked_quality_or_schedule')
        resumed = False
        if self.safe.core.frozen and self.policy.safe_updates:
            actual_aux = evidence.get('actual_auxiliary_image_frames', evidence.get('auxiliary_raw_frames', []))
            if evidence.get('trusted_reference_frames') or actual_aux:
                return self.log('resume_blocked_non_anchor_only_proof')
            proof = dict(frame=observation['frame'], request=evidence['request_id'])
            if any((p['request'] == proof['request'] for p in self.resume_proofs)):
                raise ValueError('Reused resume proof')
            if not self.resume_proofs or proof['frame'] - self.resume_proofs[-1]['frame'] >= self.policy.min_gap:
                self.resume_proofs.append(proof)
            if len(self.resume_proofs) < self.policy.resume_confirmations:
                return self.log('admission_blocked_waiting_clean_resume', confirmations=len(self.resume_proofs))
            self.safe.resume(prompt_text('online_semantic_codebook_v1_bd6bc24c26'))
            self.log('clean_proof_resume', proofs=self.resume_proofs[:])
            self.resume_proofs = []
            resumed = True
        evidence = copy.deepcopy(evidence)
        if resumed:
            evidence['prior_trajectory_dependency_frames'] = evidence.get('trajectory_dependency_frames', [])[:]
            evidence['trajectory_dependency_frames'] = []
            evidence['auxiliary_raw_frames'] = evidence.get('actual_auxiliary_image_frames', [])[:]
            self.log('new_anchor_verified_evidence_epoch', prior_trajectory_dependencies=evidence['prior_trajectory_dependency_frames'], native_tracker_reset=False, old_dependency_ledger_preserved=True)
        e = dict(evidence, frame=observation['frame'], crop_sha256=observation['crop_sha256'], quality_model=self.safe.core.contract.quality_model, trusted_added=True, p_yes=evidence['score']['conditional_probs_uncalibrated']['YES'])
        r = self.safe.consider(observation, e)
        self.log('admission', result=r, request_id=e['request_id'])
        return dict(r, trajectory_evidence_epoch_reset=resumed)

    def export(self):
        body = dict(schema=self.schema, policy=asdict(self.policy), safe=self.safe.export(), events=self.events, resume_proofs=self.resume_proofs, last_frame=self.last_frame)
        return dict(body=body, sha256=digest(body))

    @classmethod
    def restore(cls, payload, contract):
        b = payload['body']
        if payload.get('sha256') != digest(b) or b.get('schema') != cls.schema:
            raise ValueError('Corrupt online memory')
        p = SemanticCodebookPolicy(**b['policy'])
        obj = cls.__new__(cls)
        obj.policy = p
        obj.safe = SafeIdentityHistoryV2.restore(b['safe'], contract)
        obj.events = copy.deepcopy(b['events'])
        obj.resume_proofs = copy.deepcopy(b['resume_proofs'])
        obj.last_frame = b['last_frame']
        if type(obj.last_frame) is not int or obj.last_frame < obj.safe.core.clock:
            raise ValueError('Invalid online clock')
        return obj

def guarded_bbox_writeback(original, tracker, row, enabled):
    guard = row.get('codebook', {}).get('writeback_guard', {})
    allow = bool(enabled) and (not guard.get('blocked', False))
    result = original(tracker, row, allow)
    row['state_writeback']['codebook_guard'] = copy.deepcopy(guard)
    return result
