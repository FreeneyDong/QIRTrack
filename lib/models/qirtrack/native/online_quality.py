"""Online causal native observations -> trained S9; no labels or AR fake logits."""
import numpy as np
import torch
from lib.models.qirtrack.native.native_response_bqp_v1 import NativeResponseU1
from lib.models.qirtrack.native.native_candidate_inputs_v1 import NativeCachedPastS9, build_current
from lib.models.qirtrack.core.trigger_e2e.model import causal_box_geometry

class NativeOnlineQuality:

    def __init__(self, kind, checkpoint, device='cuda'):
        c = torch.load(checkpoint, map_location='cpu')
        assert c['model_type'] == 'NativeCachedPastS9' and c['tracker_kind'] == kind
        assert c['training_role'] == 'got10k_train' and c['full_S9'] and (not c['pilot']) and (c['step'] == 4000)
        self.raw = NativeResponseU1(kind)
        self.model = NativeCachedPastS9(self.raw, template_mode='recent')
        self.model.load_state_dict(c['model'], strict=True)
        self.model.to(device).eval()
        self.device = device
        self.policy = c['recent_policy']['policy'] if 'policy' in c['recent_policy'] else c['recent_policy']
        assert set(self.policy) == {'minimum_native_score', 'maximum_spatial_entropy', 'minimum_age', 'maximum_age'}

    def predict(self, observation, arrays, past, raw_reference=None):
        frame = int(observation['frame_index'])
        assert all((int(o['frame_index']) < frame for (o, a) in past))
        selected = (past + [(observation, arrays)])[-16:]
        ids = [int(o['frame_index']) for (o, a) in selected]
        assert ids == sorted(set(ids))
        start = 16 - len(selected)
        mask = np.zeros(16, dtype=bool)
        mask[start:] = True
        hist = dict(history_frame_index=np.zeros(16, np.int64), history_observation_valid=mask, history_roi_valid=np.zeros(16, bool), history_roi_pool=np.zeros((16, 768), np.float32), history_template_pool=np.zeros((16, 768), np.float32), history_native_evidence=np.zeros((16, 3), np.float32), history_boxes_xywh=np.zeros((16, 4), np.float32))
        responses = np.zeros((16, 1, 24, 24), np.float32)
        for (i, (o, a)) in enumerate(selected, start):
            hist['history_frame_index'][i] = int(o['frame_index'])
            hist['history_roi_valid'][i] = o['roi_valid']
            hist['history_roi_pool'][i] = a['roi_pool'].astype(np.float32)
            hist['history_template_pool'][i] = a['template_pool'].astype(np.float32)
            hist['history_native_evidence'][i] = [o['score_peak'], o['response_peak'], o['spatial_mass_entropy']]
            hist['history_boxes_xywh'][i] = o['bbox_xywh']
            responses[i, 0] = a['score_map'].astype(np.float32)
        hist['template_pool'] = (arrays if raw_reference is None else raw_reference)['template_pool'].astype(np.float32)
        raw_x = dict(template_tokens=torch.from_numpy(hist['history_template_pool'])[:, None], search_tokens=torch.from_numpy(hist['history_roi_pool'])[:, None], coord_logits=torch.from_numpy(responses), tracker_score=torch.from_numpy(hist['history_native_evidence']), pred_bbox=torch.from_numpy(hist['history_boxes_xywh']), history_mask=torch.from_numpy(mask))
        mask_past = raw_x['history_mask'][:-1]
        with torch.no_grad():
            geom = causal_box_geometry(raw_x['pred_bbox'][:-1][None], mask_past[None]).reshape(15, -1)
            keys = ['template_tokens', 'search_tokens', 'coord_logits', 'tracker_score']
            encoded = self.raw.frame_encoder(*[raw_x[k][:-1].to(self.device) for k in keys], geom.to(self.device)).cpu()
            crop = observation['crop']
            current = dict(search_crop_actual_xywh=np.asarray(crop['actual_xywh']), search_model_origin=np.asarray(crop['model_mapping_origin']), current_bbox_xywh=np.asarray(observation['bbox_xywh']), search_valid_fraction=arrays['search_valid_fraction'], search_ce_removed=arrays['search_ce_removed'])
            for key in ['search', 'template']:
                current[key + '_tokens'] = arrays[key + '_tokens_qint8'].astype(np.float32) * arrays[key + '_tokens_scale']
            (x, proof) = build_current(current, hist, frame, encoded, raw_x['history_mask'], self.policy)
            batch = {k: v[None].to(self.device) for (k, v) in x.items()}
            out = self.model(batch)
            q = float(out['predicted_iou'].item())
            assert np.isfinite(q) and 0 <= q <= 1
        return dict(Q=q, frame=frame, history_frames=ids[:-1], recent=proof, input_contract='same native score/pooled causal K16 and current-own template/search/validity geometry as trained', input=x)
