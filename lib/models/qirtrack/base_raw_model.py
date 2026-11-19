"""Base-seq raw-IoU encoder, a prerequisite to the full S9 adapter.

No Large checkpoint is loaded. The tracker is frozen and only cached Base
observations enter this model. Scalar confidence is fit-standardised rather
than erased by LayerNorm(1).
"""
from lib.config.paths import configured_path
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
WORK = Path(configured_path('QIR_PATH_07A5EE1A93'))
from lib.models.qirtrack.core.trigger_e2e.model import TriggerE2E
from lib.models.qirtrack.core.trigger_e2e.losses import IoUDistributionLoss
CONFIG = dict(token_dim=768, tracker_score_dim=1, d_model=128, heads=4, history_length=16, temporal_layers=2, iou_bins=50, use_temporal=True, temporal_fusion='unified_cross_attention', transition_auxiliary_outputs=0, distribution_gate_mode='standard')

class ScalarStandardisation(nn.Module):

    def __init__(self, mean, std):
        super().__init__()
        assert np.isfinite(mean) and np.isfinite(std) and (std > 0)
        self.register_buffer('mean', torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer('std', torch.tensor(max(float(std), 1e-06), dtype=torch.float32))

    def forward(self, x):
        return (x.float() - self.mean) / self.std

class BaseSeqRawU1(TriggerE2E):

    def __init__(self, score_mean, score_std):
        super().__init__(**CONFIG)
        self.frame_encoder.score_encoder[0] = ScalarStandardisation(score_mean, score_std)

def load_raw_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu')
    assert checkpoint['model_type'] == 'BaseSeqRawU1' and checkpoint['model_config'] == CONFIG
    model = BaseSeqRawU1(checkpoint['score_mean'], checkpoint['score_std'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return (model.to(device).eval(), checkpoint)

def make_sample(native, t):
    assert int(native['frame'][t - 1]) == t
    indices = np.arange(t - 16, t)
    mask = indices >= 0
    indices = np.clip(indices, 0, t - 1)
    value = dict(template_tokens=np.asarray(native['anchor_template_tokens'][indices]), search_tokens=np.asarray(native['search_tokens'][indices]), coord_logits=np.asarray(native['coord_logits'][indices]), tracker_score=np.asarray(native['tracker_score'][indices]), pred_bbox=np.asarray(native['pred_bbox'][indices]), history_mask=mask)
    assert value['template_tokens'].shape == value['search_tokens'].shape == (16, 16, 768)
    assert value['coord_logits'].shape == (16, 4, 800) and value['tracker_score'].shape == (16, 1)
    assert value['pred_bbox'].shape == (16, 4) and mask[-1]
    assert all((np.isfinite(v).all() for v in value.values()))
    return {k: torch.from_numpy(np.array(v, copy=True)) for (k, v) in value.items()}

def absolute_metrics(q, y):
    (q, y) = (np.asarray(q, dtype=float), np.asarray(y, dtype=float))
    error = q - y
    result = dict(n=len(y), mae=float(abs(error).mean()), rmse=float(np.sqrt((error ** 2).mean())), bias=float(error.mean()), overestimate_ge02=int((error >= 0.2).sum()), underestimate_ge02=int((error <= -0.2).sum()), true_bad_under04=int((y < 0.4).sum()), highQ_ge06_bad=int(((q >= 0.6) & (y < 0.4)).sum()), thresholds={})
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.675, 0.7, 0.8, 0.9]:
        trigger = q < threshold
        bad = y < 0.4
        tp = int((trigger & bad).sum())
        fp = int((trigger & ~bad).sum())
        fn = int((~trigger & bad).sum())
        tn = int((~trigger & ~bad).sum())
        result['thresholds'][str(threshold)] = dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=tp / (tp + fn) if tp + fn else None, false_trigger_rate=fp / (fp + tn) if fp + tn else None)
    return result
