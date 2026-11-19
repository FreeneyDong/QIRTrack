from lib.models.qirtrack.native.online_quality import NativeOnlineQuality
from lib.models.qirtrack.native.native_tracker_transaction_v1 import NativeTrackerTransaction

class NativeRuntime:
    def __init__(self,adapter,kind,predictor):
        self.ad=adapter;self.t=adapter.t;self.transaction=NativeTrackerTransaction(adapter);self.quality=NativeOnlineQuality(kind,predictor);self.past=[]
    def initialize(self,image,box):self.ad.initialize(image,box)
    def native_step(self,image,frame):
        previous=list(self.t.state);o,a=self.transaction.native(image,frame);self.current=(o,a)
        q=self.quality.predict(o,a,self.past)['Q']
        return previous,o['bbox_xywh'],q,o['score_peak'],o['spatial_mass_entropy']
    def trial(self,seed,factor):
        c=self.transaction.trial(seed,factor);o,a=c['observation'],c['arrays']
        return o['bbox_xywh'],self.quality.predict(o,a,self.past,raw_reference=self.current[1])['Q']
    def assert_fallback(self):assert self.ad.digest(self.ad.snapshot())==self.ad.digest(self.transaction.post)
    def advance(self):self.past=(self.past+[self.current])[-16:]
    def close(self):self.ad.close()
