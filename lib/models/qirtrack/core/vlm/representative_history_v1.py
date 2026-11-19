"""Fixed-capacity history selection. No identity/IoU decision, training or veto."""
import json, math, copy
import numpy as np

class RepresentativeHistory:

    def __init__(self, anchor, capacity=4, recent=2):
        self.anchor = self.unit(anchor)
        self.capacity = capacity
        self.recent = recent
        if capacity != 4 or recent != 2:
            raise ValueError('Frozen 4/2 capacity contract')
        self.entries = {}
        self.last_frame = 0
        self.requests = set()

    @staticmethod
    def unit(v):
        x = np.asarray(v, dtype=np.float64)
        if x.shape != (512,) or not np.isfinite(x).all() or np.linalg.norm(x) < 1e-12:
            raise ValueError('CLIP 512 finite vector required')
        return x / np.linalg.norm(x)

    def query(self, frame):
        if frame <= self.last_frame:
            raise ValueError('Query before same-frame admission')
        result = sorted(self.entries)
        assert len(result) <= 4 and all((t < frame for t in result))
        return result

    def add(self, frame, embedding, request, q, p_yes):
        if frame <= self.last_frame or not request or request in self.requests:
            raise ValueError('Duplicate/noncausal evidence')
        if not all((math.isfinite(v) for v in [q, p_yes])) or not 0.8 <= q <= 1 or (not 0.7 <= p_yes <= 1):
            raise ValueError('Original W1 admission proof required')
        v = self.unit(embedding)
        self.last_frame = frame
        self.requests.add(request)
        self.entries[frame] = dict(vector=v.tolist(), request=request, q=float(q), p_yes=float(p_yes))
        pool = sorted(self.entries)
        if len(pool) > self.capacity:
            keep = pool[-self.recent:]
            refs = [self.anchor] + [self.unit(self.entries[t]['vector']) for t in keep]
            while len(keep) < self.capacity:
                candidates = [t for t in pool if t not in keep]
                pick = max(candidates, key=lambda t: (min((1.0 - float(self.unit(self.entries[t]['vector']) @ r) for r in refs)), -t))
                keep.append(pick)
                refs.append(self.unit(self.entries[pick]['vector']))
            self.entries = {t: self.entries[t] for t in sorted(keep)}
        assert set(pool[-min(2, len(pool)):]) <= set(self.entries)
        return sorted(self.entries)

    def export(self):
        return dict(schema='representative-history-4-recent2-v1', anchor=self.anchor.tolist(), capacity=4, recent=2, entries=copy.deepcopy(self.entries), last_frame=self.last_frame, requests=sorted(self.requests))

    @classmethod
    def restore(cls, state):
        if state['schema'] != 'representative-history-4-recent2-v1':
            raise ValueError('Wrong scheme')
        obj = cls(state['anchor'], state['capacity'], state['recent'])
        if abs(np.linalg.norm(state['anchor']) - 1.0) > 1e-08:
            raise ValueError('Unnormalised anchor snapshot')
        obj.anchor = np.asarray(state['anchor'], dtype=np.float64)
        obj.entries = {int(k): v for (k, v) in state['entries'].items()}
        obj.last_frame = int(state['last_frame'])
        obj.requests = set(state['requests'])
        if len(obj.entries) > 4 or any((t <= 0 or t > obj.last_frame for t in obj.entries)):
            raise ValueError('Invalid restored time/capacity')
        for v in obj.entries.values():
            obj.unit(v['vector'])
        return obj

def tests():
    rng = np.random.default_rng(19)
    anchor = rng.normal(size=512)
    b = RepresentativeHistory(anchor)
    for i in range(1, 15):
        assert all((t < i * 90 for t in b.query(i * 90)))
        b.add(i * 90, rng.normal(size=512), str(i), 0.9, 0.8)
        assert len(b.entries) == min(i, 4)
        restored = RepresentativeHistory.restore(json.loads(json.dumps(b.export())))
        assert restored.export() == b.export()
    for (frame, req) in [(1260, 'new'), (1350, '14')]:
        try:
            b.add(frame, anchor, req, 0.9, 0.8)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted duplicate')
    for v in [np.zeros(512), np.ones(511), np.full(512, np.nan)]:
        try:
            RepresentativeHistory(v)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid vector')
    print('REPRESENTATIVE_HISTORY_TESTS_PASS')
