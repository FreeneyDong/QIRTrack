"""OSTrack/SUTrack-specific evidence adapter, reusing existing BQP layers.

Preparation code, not a trained model. No AR coordinate logits are invented.
The legacy coord_logits argument carries native response maps ONLY inside the
reused frame-encoder ABI; model_type and tracker identity must be checked.
"""
from lib.config.paths import configured_path
import sys
from pathlib import Path
import torch
from torch import nn
WORK = Path(configured_path('QIR_PATH_07A5EE1A93'))
from lib.models.qirtrack.core.trigger_e2e.model import TriggerE2E, causal_box_geometry

class NativeResponseEncoder(nn.Module):

    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(256, width), nn.LayerNorm(width))

    def forward(self, response):
        assert response.ndim == 4 and response.shape[1:] == (1, 24, 24), 'Expected native score map, not four coordinate distributions'
        assert torch.isfinite(response).all()
        return self.net(response.float())

class NativeResponseU1(TriggerE2E):

    def __init__(self, tracker_kind):
        assert tracker_kind in ('ostrack', 'sutrack')
        super().__init__(token_dim=768, tracker_score_dim=3, d_model=128, heads=4, history_length=16, temporal_layers=2, iou_bins=50, use_temporal=True, temporal_fusion='unified_cross_attention')
        self.tracker_kind = tracker_kind
        self.frame_encoder.coordinate_encoder = NativeResponseEncoder(128)

    def encode_frames(self, batch):
        assert batch['coord_logits'].ndim == 5 and batch['coord_logits'].shape[2:] == (1, 24, 24)
        (b, k) = batch['history_mask'].shape
        assert k == 16
        geom = causal_box_geometry(batch['pred_bbox'], batch['history_mask'])

        def flat(x):
            return x.reshape(b * k, *x.shape[2:])
        return self.frame_encoder(flat(batch['template_tokens']), flat(batch['search_tokens']), flat(batch['coord_logits']), flat(batch['tracker_score']), flat(geom)).reshape(b, k, -1)

def make_existing_bqp(raw_model, template_mode='recent'):
    from lib.models.qirtrack.core.localisation.searchmap_candidate import PrPoolSearchMapCandidateIoU
    assert isinstance(raw_model, NativeResponseU1)
    return PrPoolSearchMapCandidateIoU(raw_model, template_mode=template_mode)

def validate_weight_role(checkpoint, kind, role, tracker_sha):
    assert checkpoint['model_type'] == 'NativeResponseU1'
    assert checkpoint['tracker_kind'] == kind
    assert checkpoint['training_role'] == role
    assert checkpoint['tracker_checkpoint_sha256'] == tracker_sha
    assert not (role.endswith('_train') and 'test' in checkpoint.get('training_role', ''))

def checked_history_responses(indices, valid, response_by_frame, current_response):
    """Restore real dense response history by frame keys; never zero-fill real frames."""
    assert len(indices) == len(valid) == 16
    current = int(indices[-1])
    assert bool(valid[-1])
    output = torch.zeros(16, 1, 24, 24)
    past = [int(i) for (i, m) in zip(indices[:-1], valid[:-1]) if m]
    assert all((i < current for i in past)) and past == sorted(set(past))
    for (k, (i, m)) in enumerate(zip(indices, valid)):
        if not m:
            continue
        value = current_response if k == 15 else response_by_frame[int(i)]
        value = torch.as_tensor(value, dtype=torch.float32)
        assert value.shape == (24, 24) and torch.isfinite(value).all()
        output[k, 0] = value
    return output
