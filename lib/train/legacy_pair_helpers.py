"""Sampling and checked input loading preserved from the diagnostic trainer."""
from pathlib import Path
import hashlib,collections,random
UPDATES=4000
BATCH=8
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda : f.read(4 * 1024 ** 2), b''):
            h.update(b)
    return h.hexdigest()

class Cache:

    def __init__(self, torch):
        self.torch = torch
        self.items = collections.OrderedDict()
        self.size = 0
        self.verified = set()

    def get(self, ref):
        path = ref['file']
        if path in self.items:
            self.items.move_to_end(path)
            return self.items[path]
        if path not in self.verified:
            assert sha(path) == ref['sha256']
            self.verified.add(path)
        x = self.torch.load(path, map_location='cpu')
        assert len(x) == 14
        assert x['template_tokens'].shape == x['search_tokens'].shape == (16, 16, 768)
        assert x['coord_logits'].shape == (16, 4, 800) and x['search_tokens_full'].shape == (256, 768)
        assert all((self.torch.isfinite(v).all() for v in x.values() if v.is_floating_point()))
        n = sum((v.numel() * v.element_size() for v in x.values()))
        self.items[path] = (x, n)
        self.size += n
        while self.size > 6 * 1024 ** 3:
            (_, (_, removed)) = self.items.popitem(last=False)
            self.size -= removed
        return (x, n)

def batch(cache, refs, torch):
    xs = [cache.get(r)[0] for r in refs]
    return {k: torch.stack([x[k] for x in xs]).cuda() for k in xs[0]}

def schedule(groups, seed):
    rng = random.Random(seed)
    order = []
    cursor = 0
    draws = []
    queues = {}
    source_counts = collections.Counter()

    def pick(indices, key):
        if not queues.get(key):
            queues[key] = rng.sample(indices, len(indices))
        return queues[key].pop()
    for step in range(UPDATES):
        selected = []
        while len(selected) < BATCH:
            if cursor >= len(order):
                order = rng.sample(list(range(len(groups))), len(groups))
                cursor = 0
            g = order[cursor]
            cursor += 1
            if g not in selected:
                selected.append(g)
        drawn = []
        for (slot, g) in enumerate(selected):
            group = groups[g]
            global_slot = (step * BATCH + slot) % 4 == 0
            candidates = group['global_'] if global_slot and group['global_'] else group['local']
            if not candidates:
                candidates = list(range(len(group['pairs'])))
            source = 'global' if candidates is group['global_'] else 'local'
            quality = {}
            for i in candidates:
                p = group['pairs'][i]
                q = min(4, int(p['raw_iou'] / 0.2))
                gain = 0 if p['true_gain'] < -0.05 else 2 if p['true_gain'] > 0.05 else 1
                quality.setdefault((q, gain), []).append(i)
            desired = ((step * BATCH + slot) % 5, (step * BATCH + slot) // 5 % 3)
            layer = desired if desired in quality else rng.choice(sorted(quality))
            i = pick(quality[layer], (g, source, *layer))
            drawn.append((g, i))
            source_counts[source] += 1
            source_counts['raw_iou_bin%d_gain_layer%d' % layer] += 1
        draws.append(drawn)
    return (draws, dict(source_counts))
