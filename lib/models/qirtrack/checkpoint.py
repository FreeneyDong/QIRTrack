import torch
from .base_raw_model import BaseSeqRawU1
from .core.localisation.searchmap_candidate import PrPoolSearchMapCandidateIoU

def load_legacy_diagnostic(path, device='cpu'):
    # Only load checkpoints from trusted sources (torch pickle format).
    c = torch.load(path, map_location='cpu')
    if c.get('variant_id') != 'artrackv2_base256_seq/lasot_test430_diagnostic/S9_recent_v1':
        raise ValueError('Wrong predictor identity; no cross-tracker fallback')
    if c.get('tracker_checkpoint_sha256') != '6c2654caa850fce22ccde730e1040b72becb4364055cd044dea22322f677c604':
        raise ValueError('Wrong native tracker checkpoint')
    model = PrPoolSearchMapCandidateIoU(BaseSeqRawU1(c['score_mean'], c['score_std']), template_mode='recent')
    model.load_state_dict(c['model'], strict=True)
    return model.to(device).eval(), c
