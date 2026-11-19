import collections
class Column:

    def __init__(self, np):
        self.items = collections.deque(maxlen=256)
        self.count = 0
        self.np = np

    def append(self, x):
        self.items.append(x)
        self.count += 1

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self[key[0]][:, key[1]]
        offset = self.count - len(self.items)
        if self.np.isscalar(key):
            assert offset <= int(key) < self.count, 'Causal history index outside retained window'
            return self.items[int(key) - offset]
        assert all((offset <= int(i) < self.count for i in key)), 'Causal history index outside retained window'
        return self.np.asarray([self.items[int(i) - offset] for i in key])
