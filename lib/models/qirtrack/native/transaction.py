"""Controller-facing isolated trial bridge over the already-qualified native adapter."""
import copy

class NativeTrackerTransaction:

    def __init__(self, adapter):
        self.adapter = adapter

    def native(self, image, frame):
        self.pre = self.adapter.snapshot()
        (self.raw_obs, self.raw_arrays) = self.adapter.step(image, frame, full=True)
        self.post = self.adapter.snapshot()
        self.frame = frame
        self.image = image
        return (self.raw_obs, self.raw_arrays)

    def trial(self, seed, search_factor):
        ad = self.adapter
        try:
            ad.restore(self.pre)
            ad.t.state = list(map(float, seed))
            ad.t.params.search_factor = float(search_factor)
            (obs, arrays) = ad.step(self.image, self.frame, full=True)
            return dict(observation=obs, arrays=arrays, state=ad.snapshot(), seed=list(seed))
        finally:
            ad.restore(self.post)
            assert ad.digest(ad.snapshot()) == ad.digest(self.post), 'Trial contaminated native fallback'

    def feedback(self, box, accepted, guard):
        ad = self.adapter
        ad.restore(self.post)
        if accepted and guard:
            ad.t.state = list(map(float, box))
        else:
            assert ad.digest(ad.snapshot()) == ad.digest(self.post)
        return dict(policy='bbox_only', applied=bool(accepted and guard), dynamic_templates_unchanged=True)
