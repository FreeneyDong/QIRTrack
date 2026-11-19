from hashlib import sha256

import torch
from .base_raw_model import BaseSeqRawU1
from .core.localisation.roi_quality import PrPoolSearchMapCandidateIoU

def load_legacy_diagnostic(path, device='cpu'):
    # Only load checkpoints from trusted sources (torch pickle format).
    c = torch.load(path, map_location='cpu')
    # Check the exact legacy checkpoint family tag without changing the stored metadata.
    tag = c.get('variant_id')
    if not isinstance(tag, str) or sha256(tag.encode('utf-8')).hexdigest() != '9a2d3d782b3a13451fb5dd3c48fb5e855d337d151d37fd63afb7603ff93871f2':
        raise ValueError('Wrong predictor identity; no cross-tracker fallback')
    if c.get('tracker_checkpoint_sha256') != '6c2654caa850fce22ccde730e1040b72becb4364055cd044dea22322f677c604':
        raise ValueError('Wrong native tracker checkpoint')
    model = PrPoolSearchMapCandidateIoU(BaseSeqRawU1(c['score_mean'], c['score_std']), template_mode='recent')
    model.load_state_dict(c['model'], strict=True)
    return model.to(device).eval(), c
