# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.

import torch.nn as nn

def is_multi_gpu(net):
    return isinstance(net, (MultiGPU, nn.parallel.distributed.DistributedDataParallel))

class MultiGPU(nn.parallel.distributed.DistributedDataParallel):

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except:
            pass
        return getattr(self.module, item)
