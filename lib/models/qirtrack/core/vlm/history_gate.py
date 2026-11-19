"""CB4 safe read facade; isolated shadow, never changes tracker decisions."""
from lib.models.qirtrack.mllm.prompts import prompt_text
import copy
from dataclasses import asdict
from .unified_memory import UnifiedIdentityMemory, digest
import numpy as np

class SafeIdentityHistory:
    schema = 'safe-identity-history-shadow-v1'

    def __init__(self, sequence, anchor, contract, policy):
        self.core = UnifiedIdentityMemory(sequence, anchor, contract, policy)
        self.released = set()
        self.crop_ledger = {anchor['crop_sha256']}
        self.invalid_frames = set()
        self.audit = []
        self.releases = {0: []}

    def _eligible(self):
        out = {}
        for (mode, ids) in self.core.active.items():
            safe = [k for k in ids if k in self.released and self.core.records[k]['frame'] not in self.invalid_frames]
            if len(safe) >= self.core.policy.confirmations:
                out[mode] = safe
        return out

    def read(self, frame, vector=None, top_k=4):
        view = self.core.read(frame, vector, top_k)
        modes = {} if self.core.frozen else self._eligible()
        ids = {k for group in modes.values() for k in group}
        recent = sorted(ids, key=lambda k: self.core.records[k]['frame'])[-self.core.policy.recent:]
        z = None if vector is None else np.asarray(vector, dtype=float)
        if z is not None:
            z = z / np.linalg.norm(z)
        prototypes = []
        for (mode, group) in modes.items():
            k = self.core._medoid(group)
            prototypes.append(dict(mode=mode, record=k, frame=self.core.records[k]['frame'], similarity=None if z is None else float(z @ self.core.records[k]['vector'])))
        prototypes.sort(key=lambda x: (0 if x['similarity'] is None else -x['similarity'], x['mode']))
        return dict(view, recent=[copy.deepcopy(self.core.records[k]) for k in recent], prototypes=prototypes[:top_k], anchor=self.core.anchor, quarantine_count=len(set(self.core.records) - ids), safe_history_only=True, anchor_only=not ids)

    def consider(self, observation, evidence):
        o = self.core._validate(observation)
        e = copy.deepcopy(evidence)
        refs = e.get('trusted_reference_frames')
        aux = e.get('auxiliary_raw_frames')
        if not isinstance(refs, list) or not isinstance(aux, list) or any((type(t) is not int or not 0 <= t < o['frame'] for t in refs + aux)) or (len(refs) != len(set(refs))):
            raise ValueError('Explicit causal reference roles required')
        allowed = {0} | {self.core.records[k]['frame'] for ids in self._eligible().values() for k in ids}
        reason = None
        if o['crop_sha256'] in self.crop_ledger:
            reason = 'blocked_exact_crop_duplicate'
        elif set(refs) - allowed:
            reason = 'blocked_unreleased_reference'
        elif set(refs + aux) & self.invalid_frames:
            reason = 'blocked_invalidated_reference'
        if reason:
            e['trusted_added'] = False
        o['proof_references'] = sorted(set(refs + aux))
        event = self.core.consider(o, e)
        if reason and event['action'] == 'blocked_untrusted':
            event = dict(event, action=reason)
        inserted = event.get('record')
        if inserted is not None:
            self.crop_ledger.add(o['crop_sha256'])
            for (mode, ids) in self.core.active.items():
                candidates = [k for k in ids if k not in self.released and self.core.records[k]['frame'] not in self.invalid_frames]
                if len(candidates) >= self.core.policy.confirmations:
                    self.released.update(candidates)
        self.released.intersection_update(self.core.records)
        self.releases[self.core.version] = sorted(self.released)
        self.releases = {v: r for (v, r) in self.releases.items() if v in self.core.snapshots}
        result = dict(event, safe_released=len(self.released), auxiliary_raw_used=bool(aux), independent_identity_proof=False)
        self.audit.append(copy.deepcopy(result))
        return result

    def invalidate(self, frames, reason):
        if not frames or any((type(t) is not int or t <= 0 for t in frames)) or (not isinstance(reason, str)) or (not reason.strip()):
            raise ValueError(prompt_text('safe_identity_history_v1_c024c83ff2'))
        invalid = self.invalid_frames | set(frames)
        changed = True
        while changed:
            before = len(invalid)
            for o in self.core.records.values():
                if set(o.get('proof_references', [])) & invalid:
                    invalid.add(o['frame'])
            changed = len(invalid) != before
        self.invalid_frames = invalid
        self.released = {k for k in self.released if self.core.records[k]['frame'] not in invalid}
        self.core.frozen = True
        event = self.core._event('dependency_invalidation_frozen', frames=sorted(invalid), reason=reason)
        self.releases[self.core.version] = sorted(self.released)
        self.releases = {v: r for (v, r) in self.releases.items() if v in self.core.snapshots}
        self.audit.append(event)
        return copy.deepcopy(event)

    def rollback(self, version):
        if version not in self.releases:
            raise ValueError('Unknown/pruned safe snapshot')
        released = set(self.releases[version])
        event = self.core.rollback(version)
        self.released = {k for k in released if self.core.records[k]['frame'] not in self.invalid_frames}
        self.releases[self.core.version] = sorted(self.released)
        self.releases = {v: r for (v, r) in self.releases.items() if v in self.core.snapshots}
        self.audit.append(event)
        return event

    def resume(self, reason):
        event = self.core.resume(reason)
        self.releases[self.core.version] = sorted(self.released)
        self.releases = {v: r for (v, r) in self.releases.items() if v in self.core.snapshots}
        self.audit.append(event)
        return event

    def export(self):
        body = dict(schema=self.schema, core=self.core.export(), released=sorted(self.released), crop_ledger=sorted(self.crop_ledger), invalid_frames=sorted(self.invalid_frames), releases={str(v): r for (v, r) in self.releases.items()}, audit=copy.deepcopy(self.audit))
        return dict(body=body, sha256=digest(body))

    @classmethod
    def restore(cls, payload, contract):
        b = payload['body']
        if digest(b) != payload['sha256'] or b['schema'] != cls.schema:
            raise ValueError('Corrupt safe memory')
        core = UnifiedIdentityMemory.restore(b['core'], contract)
        obj = cls.__new__(cls)
        obj.core = core
        (obj.released, obj.crop_ledger) = (set(b['released']), set(b['crop_ledger']))
        obj.invalid_frames = set(b['invalid_frames'])
        obj.releases = {int(v): r for (v, r) in b['releases'].items()}
        obj.audit = copy.deepcopy(b['audit'])
        if not obj.released <= set(core.records) or set(obj.releases) != set(core.snapshots) or obj.releases[core.version] != sorted(obj.released) or (core.anchor['crop_sha256'] not in obj.crop_ledger) or any((type(t) is not int or t <= 0 for t in obj.invalid_frames)):
            raise ValueError('Broken safe state/ledger')
        for (version, keys) in obj.releases.items():
            if len(keys) != len(set(keys)) or not set(keys) <= set(core.snapshots[version]['records']):
                raise ValueError('Broken safe snapshot')
        if any((o['crop_sha256'] not in obj.crop_ledger for o in core.records.values())):
            raise ValueError('Missing crop ledger')
        return obj

def archived_admission_proof(request, observation, contract, q_raw, p_yes):
    """Bind saved actual input roles; not a new model response or anchor-only audit."""
    t = observation['frame']
    (domain, sequence) = observation['sequence'].split('__', 1)
    if request.get('domain') != domain or request.get('sequence') != sequence or request.get('frame') != t or (request.get('stage') != 'periodic_audit') or (request.get('status') != 'response') or (request.get('input_contract_validated') is not True) or (request.get('frame_source', {}).get('frame_index_zero_based') != t):
        raise ValueError('Actual admission request provenance mismatch')
    images = request['ordered_images']
    if not any((x['role'] == 'ANCHOR_TARGET' and x['frame'] == 0 for x in images)):
        raise ValueError('Immutable anchor missing')
    current = [x for x in images if x['role'] == 'CURRENT_RAW_FULL']
    if len(current) != 1 or current[0]['frame'] != t:
        raise ValueError('Ambiguous current image')
    return dict(frame=t, crop_sha256=observation['crop_sha256'], quality_model=contract.quality_model, request_id=request['request_id'], q_raw=q_raw, p_yes=p_yes, stage='periodic_audit', trusted_added=True, trusted_reference_frames=[x['frame'] for x in images if x['role'] == 'TRUSTED'], auxiliary_raw_frames=[x['frame'] for x in images if x['role'] == 'RECENT_RAW'])
