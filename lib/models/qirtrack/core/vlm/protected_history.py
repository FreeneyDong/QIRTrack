"""V2 retains scalar proof dependencies across evidence pruning and rollback."""
from lib.models.qirtrack.mllm.prompts import prompt_text
import copy
from .history_gate import SafeIdentityHistory
from .unified_memory import digest

class SafeIdentityHistoryV2(SafeIdentityHistory):
    schema = 'safe-identity-history-shadow-v2-dependency-ledger'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dependency_ledger = {}

    def consider(self, observation, evidence):
        result = super().consider(observation, evidence)
        key = result.get('record')
        if key is not None:
            o = self.core.records[key]
            self.dependency_ledger[o['frame']] = list(o.get('proof_references', []))
        return result

    def invalidate(self, frames, reason):
        if not frames or any((type(t) is not int or t <= 0 for t in frames)) or (not isinstance(reason, str)) or (not reason.strip()):
            raise ValueError(prompt_text('safe_identity_history_v2_3e6d2193b8'))
        invalid = set(frames) | self.invalid_frames
        while True:
            following = {t for (t, refs) in self.dependency_ledger.items() if set(refs) & invalid}
            new = invalid | following
            if new == invalid:
                break
            invalid = new
        return super().invalidate(sorted(invalid), reason)

    def export(self):
        body = super().export()['body']
        body['dependency_ledger'] = {str(t): refs for (t, refs) in sorted(self.dependency_ledger.items())}
        return dict(body=body, sha256=digest(body))

    @classmethod
    def restore(cls, payload, contract):
        body = copy.deepcopy(payload['body'])
        if digest(body) != payload['sha256'] or body.get('schema') != cls.schema:
            raise ValueError(prompt_text('safe_identity_history_v2_b66d8e10a5'))
        ledger = body.pop('dependency_ledger', None)
        if not isinstance(ledger, dict):
            raise ValueError('Missing dependency ledger')
        body['schema'] = SafeIdentityHistory.schema
        original = SafeIdentityHistory.restore(dict(body=body, sha256=digest(body)), contract)
        obj = cls.__new__(cls)
        obj.__dict__.update(original.__dict__)
        obj.dependency_ledger = {}
        for (key, refs) in ledger.items():
            try:
                t = int(key)
            except (ValueError, TypeError):
                raise ValueError('Invalid dependency frame')
            if str(t) != key or t <= 0 or t > obj.core.last_update or (not isinstance(refs, list)) or any((type(x) is not int or not 0 <= x < t for x in refs)) or (refs != sorted(set(refs))):
                raise ValueError('Invalid dependency ledger references')
            obj.dependency_ledger[t] = refs
        for state in [obj.core._state()] + list(obj.core.snapshots.values()):
            for o in state['records'].values():
                if obj.dependency_ledger.get(o['frame']) != o.get('proof_references', []):
                    raise ValueError('Missing/mismatched dependency proof')
        if any((set(refs) & obj.invalid_frames and t not in obj.invalid_frames for (t, refs) in obj.dependency_ledger.items())):
            raise ValueError('Incomplete invalidation closure')
        return obj
