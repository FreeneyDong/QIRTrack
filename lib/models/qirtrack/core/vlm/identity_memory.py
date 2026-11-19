"""Causal recent-FIFO + representative union; never an identity probability."""
from lib.models.qirtrack.mllm.prompts import prompt_text
import copy
import math
import numpy as np
from .online_codebook import OnlineIdentityCodebook, unit
from .unified_memory import digest

def fused_references(observations, anchor_vector, frame, representative_slots=4, recent_slots=4, max_slots=6):
    """Return at most six *unique* references, with no future/anchor observations.

    Admission/quarantine are the caller's responsibility. The safe adapter below
    supplies released records only; similarity never supplies admission proof.
    No new semantic near-duplicate threshold is introduced.
    """
    if type(frame) is not int or frame <= 0 or representative_slots != 4 or (recent_slots != 4) or (max_slots not in (4, 6)):
        raise ValueError(prompt_text('fused_identity_history_v2_cf4813bfdc'))
    rows = {}
    for observation in observations:
        o = copy.deepcopy(observation)
        t = o.get('frame')
        if type(t) is not int or not 0 < t < frame:
            raise ValueError(prompt_text('fused_identity_history_v2_5b4773e9fe'))
        if not isinstance(o.get('sequence'), str) or not o['sequence']:
            raise ValueError('Explicit sequence identity required')
        if 'bbox_xywh_full' in o:
            b = o['bbox_xywh_full']
            if len(b) != 4 or not all((math.isfinite(float(x)) for x in b)) or b[2] <= 0 or (b[3] <= 0):
                raise ValueError('Invalid reference bbox')
        o['vector'] = unit(o['vector']).tolist()
        if t in rows and digest(rows[t]) != digest(o):
            raise ValueError(prompt_text('fused_identity_history_v2_3da00bac41'))
        rows[t] = o
    if len({o['sequence'] for o in rows.values()}) > 1:
        raise ValueError('Cross-sequence history forbidden')
    pool = sorted(rows)
    keep = pool[-2:]
    refs = [unit(anchor_vector)] + [unit(rows[t]['vector']) for t in keep]
    while len(keep) < min(4, len(pool)):
        candidates = [t for t in pool if t not in keep]
        pick = max(candidates, key=lambda t: (min((1.0 - float(unit(rows[t]['vector']) @ r) for r in refs)), -t))
        keep.append(pick)
        refs.append(unit(rows[pick]['vector']))
    fifo = pool[-recent_slots:]
    selected = sorted(set(keep) | set(fifo))
    if len(selected) > 6:
        raise AssertionError(prompt_text('fused_identity_history_v2_bf27d52f40'))
    (unique, seen) = ([], set())
    for t in sorted(selected, reverse=True):
        o = rows[t]
        key = o.get('reference_crop_sha256') or o.get('crop_sha256')
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append(t)
    if max_slots == 4 and len(unique) > 4:
        recent = sorted(unique)[-3:]
        older = [t for t in unique if t not in recent]
        view = max(older, key=lambda t: (min((1.0 - float(unit(rows[t]['vector']) @ r) for r in [unit(anchor_vector)] + [unit(rows[j]['vector']) for j in recent])), -t))
        unique = recent + [view]
    chosen = sorted(unique)
    assert len(chosen) <= max_slots
    result = [copy.deepcopy(rows[t]) for t in chosen]
    audit = dict(schema='fused-history-selection-v2', representative_frames=sorted(keep), fifo_frames=fifo, selected_frames=chosen, max_slots=max_slots, exact_duplicates_removed=len(selected) - len(unique) if max_slots == 6 else None, approximate_identity_dedup=False)
    return (result, audit)

class FusedIdentityCodebook(OnlineIdentityCodebook):
    """Keep V1's real-proofs/pending/release/freeze/rollback; replace read only."""
    schema = 'online-identity-codebook-fused-v2'

    @classmethod
    def from_existing(cls, memory):
        obj = cls.__new__(cls)
        obj.__dict__.update(memory.__dict__)
        obj.fused_slots = 6
        return obj

    def references(self, frame, vector, budget=None):
        self.query(frame, vector)
        n = self.fused_slots if budget is None else budget
        if n not in (0, 4, 6):
            raise ValueError(prompt_text('fused_identity_history_v2_d1128dfee8'))
        if not n or (self.safe.core.frozen and self.policy.safe_updates):
            self.log('fused_retrieve', selected_frames=[], reason='frozen_or_zero_budget')
            return []
        if self.policy.safe_updates:
            eligible = {k for group in self.safe._eligible().values() for k in group}
        else:
            eligible = set(self.safe.core.records)
        records = [self.safe.core.records[k] for k in eligible if self.safe.core.records[k]['frame'] not in self.safe.invalid_frames]
        (rows, audit) = fused_references(records, self.safe.core.anchor['vector'], frame, max_slots=n)
        self.log('fused_retrieve', **audit, safe_released_only=self.policy.safe_updates)
        return rows

    def export(self):
        body = super().export()['body']
        body['fused_slots'] = self.fused_slots
        return dict(body=body, sha256=digest(body))

    @classmethod
    def restore(cls, payload, contract):
        body = copy.deepcopy(payload['body'])
        if payload.get('sha256') != digest(body) or body.get('schema') != cls.schema:
            raise ValueError('Corrupt fused memory snapshot')
        slots = body.pop('fused_slots')
        if slots != 6:
            raise ValueError('Unsupported fused image contract')
        body['schema'] = OnlineIdentityCodebook.schema
        memory = OnlineIdentityCodebook.restore(dict(body=body, sha256=digest(body)), contract)
        return cls.from_existing(memory)
