# Nested MAE provenance: see external/licenses/MAE_CC_BY_NC_4.0.txt and docs/THIRD_PARTY.md.
# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math

class CosineEMA:

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_epoch = 0

    def adjust(self, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        self.last_epoch = epoch
        cfg = self.cfg
        ema_decay = cfg.TRAIN.EMA_SCHEDULER.MAX - (cfg.TRAIN.EMA_SCHEDULER.MAX - cfg.TRAIN.EMA_SCHEDULER.MIN) * 0.5 * (1.0 + math.cos(math.pi * epoch / cfg.TRAIN.EPOCH))
        return ema_decay

class ConstantEMA:

    def __init__(self, cfg):
        self.cfg = cfg

    def adjust(self, epoch):
        ema_decay = self.cfg.TRAIN.EMA_DECAY
        return ema_decay
