"""Shared CB0 observation store. Shadow only; no tracker/model/API dependencies."""
from lib.models.qirtrack.mllm.prompts import prompt_text
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
import numpy as np

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class Contract:
    encoder: str
    preprocessing: str
    source_sha256: str
    quality_model: str
    dimension: int = 512

@dataclass(frozen=True)
class Policy:
    merge_cosine: float
    recent: int = 4
    active: int = 8
    pending: int = 8
    support: int = 16
    confirmations: int = 2
    min_gap: int = 90
    snapshots: int = 32

class UnifiedIdentityMemory:
    schema = 'unified-identity-memory-shadow-v1'

    def __init__(self, sequence, anchor, contract, policy):
        if not sequence or not all((contract.encoder, contract.preprocessing, contract.quality_model)):
            raise ValueError('Explicit sequence/encoder/preprocessing/quality contract required')
        if len(contract.source_sha256) != 64 or any((c not in '0123456789abcdef' for c in contract.source_sha256)) or contract.dimension != 512:
            raise ValueError(prompt_text('unified_identity_memory_v1_9b20ae3e06'))
        if not math.isfinite(policy.merge_cosine) or not -1 <= policy.merge_cosine <= 1 or min(policy.recent, policy.active, policy.pending, policy.min_gap) < 1 or (policy.confirmations < 2) or (policy.support < policy.confirmations) or (policy.snapshots < 2):
            raise ValueError('Invalid bounded policy')
        (self.sequence, self.contract, self.policy) = (sequence, contract, policy)
        self._anchor = self._validate(anchor)
        if anchor['frame'] != 0:
            raise ValueError('Immutable first-frame anchor required')
        (self.records, self.recent, self.active, self.pending) = ({}, [], {}, {})
        (self.clock, self.last_update, self.serial, self.version) = (0, 0, 0, 0)
        (self.seen_requests, self.frozen, self.events) = (set(), False, [])
        self.snapshots = {0: self._state()}

    def _validate(self, observation):
        o = copy.deepcopy(observation)
        if o.get('sequence') != self.sequence or type(o.get('frame')) is not int or o['frame'] < 0 or (o.get('contract') != asdict(self.contract)):
            raise ValueError('Sequence/frame/encoder/preprocessing mismatch')
        sha = o.get('crop_sha256', '')
        if len(sha) != 64 or any((c not in '0123456789abcdef' for c in sha)) or (not o.get('crop_path')):
            raise ValueError('Actual crop provenance required')
        x = np.asarray(o.get('vector'), dtype=np.float64)
        if x.shape != (512,) or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-12:
            raise ValueError('Finite nonzero CLIP512 vector required')
        o['vector'] = (x / np.linalg.norm(x)).tolist()
        return o

    @property
    def anchor(self):
        return copy.deepcopy(self._anchor)

    def _state(self):
        return copy.deepcopy(dict(records=self.records, recent=self.recent, active=self.active, pending=self.pending))

    def _prune(self):
        live = set(self.recent)
        for ids in list(self.active.values()) + list(self.pending.values()):
            live.update(ids)
        self.records = {k: v for (k, v) in self.records.items() if k in live}

    def _event(self, action, **extra):
        self._prune()
        self.version += 1
        row = dict(version=self.version, frame=self.clock, action=action, shadow_only=True, tracker_action=None, **extra)
        self.events.append(row)
        self.snapshots[self.version] = self._state()
        for old in sorted(self.snapshots)[1:-self.policy.snapshots]:
            del self.snapshots[old]
        return copy.deepcopy(row)

    def _medoid(self, ids):
        ids = sorted(ids, key=lambda k: (self.records[k]['frame'], k))
        x = np.asarray([self.records[k]['vector'] for k in ids])
        return ids[int((x @ x.T).sum(axis=1).argmax())]

    def read(self, frame, vector=None, top_k=4):
        if type(frame) is not int or frame < self.clock or frame <= self.last_update or (top_k < 1):
            raise ValueError(prompt_text('unified_identity_memory_v1_38b004c13a'))
        if vector is not None:
            z = np.asarray(vector, dtype=np.float64)
            if z.shape != (512,) or not np.isfinite(z).all() or np.linalg.norm(z) < 1e-12:
                raise ValueError('Invalid query vector')
            z = z / np.linalg.norm(z)
        self.clock = frame
        matches = []
        for (mode, ids) in self.active.items():
            k = self._medoid(ids)
            matches.append(dict(mode=mode, record=k, frame=self.records[k]['frame'], similarity=None if vector is None else float(z @ self.records[k]['vector'])))
        matches.sort(key=lambda x: (0 if x['similarity'] is None else -x['similarity'], x['mode']))
        return dict(recent=[copy.deepcopy(self.records[k]) for k in self.recent], prototypes=matches[:top_k], anchor_similarity=None if vector is None else float(z @ self._anchor['vector']), updates_frozen=self.frozen, identity_decision=None, tracker_action=None, shadow_only=True, version=self.version)

    def consider(self, observation, evidence):
        o = self._validate(observation)
        frame = o['frame']
        if frame < self.clock or frame <= self.last_update:
            raise ValueError('Noncausal/duplicate update')
        request = evidence.get('request_id')
        if not request or request in self.seen_requests or evidence.get('frame') != frame or (evidence.get('crop_sha256') != o['crop_sha256']) or (evidence.get('quality_model') != self.contract.quality_model):
            raise ValueError('Mismatched/reused external proof')
        (q, p) = (evidence.get('q_raw'), evidence.get('p_yes'))
        if not all((isinstance(x, (int, float)) and math.isfinite(x) and (0 <= x <= 1) for x in [q, p])):
            raise ValueError('Invalid external scores')
        self.clock = self.last_update = frame
        self.seen_requests.add(request)
        if evidence.get('identity_conflict'):
            self.frozen = True
            return self._event('conflict_freeze')
        if self.frozen:
            return self._event('blocked_frozen')
        if o.get('kind') != 'raw' or frame % 90 or evidence.get('stage') != 'periodic_audit' or (evidence.get('trusted_added') is not True) or (q < 0.8) or (p < 0.7):
            return self._event('blocked_untrusted')
        key = digest([self.sequence, frame, o['crop_sha256'], request])
        o['evidence'] = copy.deepcopy(evidence)
        self.records[key] = o
        self.recent = (self.recent + [key])[-self.policy.recent:]
        pools = dict(self.active, **self.pending)
        ranked = [(float(np.dot(o['vector'], self.records[self._medoid(ids)]['vector'])), mode) for (mode, ids) in pools.items()]
        (score, mode) = max(ranked, default=(-2, None))
        if score >= self.policy.merge_cosine:
            pool = self.active if mode in self.active else self.pending
            ids = pool[mode]
            if frame - max((self.records[k]['frame'] for k in ids)) < self.policy.min_gap:
                return self._event('recent_only_temporal_duplicate', record=key)
            pool[mode] = (ids + [key])[-self.policy.support:]
            if pool is self.pending and len(pool[mode]) >= self.policy.confirmations:
                evicted = None
                if len(self.active) == self.policy.active:
                    evicted = min(self.active, key=lambda m: (max((self.records[k]['frame'] for k in self.active[m])), m))
                    del self.active[evicted]
                self.active[mode] = self.pending.pop(mode)
                return self._event('activated', mode=mode, record=key, evicted=evicted)
            return self._event('support_added', mode=mode, record=key)
        evicted = None
        if len(self.pending) == self.policy.pending:
            evicted = min(self.pending, key=lambda m: (max((self.records[k]['frame'] for k in self.pending[m])), m))
            del self.pending[evicted]
        self.serial += 1
        mode = 'mode_%06d' % self.serial
        self.pending[mode] = [key]
        return self._event('pending_created', mode=mode, record=key, evicted=evicted)

    def rollback(self, version):
        if version not in self.snapshots:
            raise ValueError('Unknown/pruned version')
        for (key, value) in copy.deepcopy(self.snapshots[version]).items():
            setattr(self, key, value)
        self.frozen = True
        return self._event('rollback_frozen', restored_version=version)

    def resume(self, reason):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Explicit reviewed reason required')
        self.frozen = False
        return self._event('reviewed_resume', reason=reason)

    def export(self):
        body = dict(schema=self.schema, sequence=self.sequence, anchor=self.anchor, contract=asdict(self.contract), policy=asdict(self.policy), state=self._state(), clock=self.clock, last_update=self.last_update, serial=self.serial, version=self.version, seen_requests=sorted(self.seen_requests), frozen=self.frozen, events=copy.deepcopy(self.events), snapshots={str(k): copy.deepcopy(v) for (k, v) in self.snapshots.items()}, shadow_only=True)
        return dict(body=body, sha256=digest(body))

    def _check_state(self, state):
        if len(state['recent']) > self.policy.recent or len(set(state['recent'])) != len(state['recent']) or len(state['active']) > self.policy.active or (len(state['pending']) > self.policy.pending) or set(state['active']) & set(state['pending']):
            raise ValueError('Invalid persisted capacities/references')
        for pool in [state['active'], state['pending']]:
            for (mode, ids) in pool.items():
                if not ids or len(ids) > self.policy.support or len(set(ids)) != len(ids) or (mode not in self.active and mode not in self.pending and (not mode.startswith('mode_'))):
                    raise ValueError('Invalid persisted prototype')
        for ids in state['active'].values():
            if len(ids) < self.policy.confirmations:
                raise ValueError('Unconfirmed active prototype')
        for (k, o) in state['records'].items():
            self._validate(o)
            proof = o.get('evidence', {})
            if not 0 < o['frame'] <= self.last_update or o['frame'] % 90 or o.get('kind') != 'raw' or (proof.get('request_id') not in self.seen_requests) or (proof.get('frame') != o['frame']) or (proof.get('crop_sha256') != o['crop_sha256']) or (proof.get('quality_model') != self.contract.quality_model) or (proof.get('trusted_added') is not True) or (proof.get('stage') != 'periodic_audit') or (not 0.8 <= proof.get('q_raw', -1) <= 1) or (not 0.7 <= proof.get('p_yes', -1) <= 1) or (abs(np.linalg.norm(o['vector']) - 1) > 1e-08) or (k != digest([self.sequence, o['frame'], o['crop_sha256'], proof.get('request_id')])):
                raise ValueError('Invalid persisted record/proof')
        refs = state['recent'] + [k for ids in list(state['active'].values()) + list(state['pending'].values()) for k in ids]
        if set(refs) != set(state['records']):
            raise ValueError('Dangling/orphan restored references')

    @classmethod
    def restore(cls, payload, expected_contract):
        b = payload['body']
        if payload['sha256'] != digest(b) or b['schema'] != cls.schema or b['contract'] != asdict(expected_contract):
            raise ValueError('Corrupt/incompatible persisted memory')
        obj = cls(b['sequence'], b['anchor'], expected_contract, Policy(**b['policy']))
        obj._anchor = copy.deepcopy(b['anchor'])
        for (k, v) in b['state'].items():
            setattr(obj, k, copy.deepcopy(v))
        for k in ['clock', 'last_update', 'serial', 'version', 'frozen', 'events']:
            setattr(obj, k, copy.deepcopy(b[k]))
        obj.seen_requests = set(b['seen_requests'])
        obj.snapshots = {int(k): copy.deepcopy(v) for (k, v) in b['snapshots'].items()}
        if type(obj.clock) is not int or type(obj.last_update) is not int or (not 0 <= obj.last_update <= obj.clock) or (len(obj.seen_requests) != len(b['seen_requests'])) or (len(obj.snapshots) > obj.policy.snapshots + 1) or (0 not in obj.snapshots) or (obj.version not in obj.snapshots):
            raise ValueError('Invalid persisted clock/ledger/versions')
        for state in [obj._state()] + list(obj.snapshots.values()):
            obj._check_state(state)
        return obj
