from functools import partial as _qir_partial
from lib.trackers.checkpoint_io import load_checkpoint as _qir_load_checkpoint
_qir_torch_load = _qir_partial(_qir_load_checkpoint, kind='artrackv2')
# QIRTrack integration modification (2026-11-18): namespaced imports and portable resource paths.
# Upstream licence and attribution remain applicable; see this tracker directory LICENSE.

import math
from pathlib import Path
from lib.trackers.artrackv2.native.models.artrackv2 import build_artrackv2
from lib.trackers.artrackv2.native.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F
import numpy as np
from lib.trackers.artrackv2.native.test.tracker.vis_utils import gen_visualization
from lib.trackers.artrackv2.native.test.utils.hann import hann2d
from lib.trackers.artrackv2.native.train.data.processing_utils import sample_target
import cv2
import os
from lib.trackers.artrackv2.native.test.tracker.data_utils import Preprocessor
from lib.trackers.artrackv2.native.utils.box_ops import clip_box
from lib.trackers.artrackv2.native.utils.ce_utils import generate_mask_cond
import random

def pool_square_tokens(tokens, output_side):
    """Average-pool ``[B,N,C]`` square token grids without learning parameters."""
    if tokens.ndim != 3:
        raise ValueError(f'expected [B,N,C] tokens, got {tuple(tokens.shape)}')
    (batch, token_count, channels) = tokens.shape
    input_side = math.isqrt(token_count)
    if input_side * input_side != token_count:
        raise ValueError(f'token count {token_count} is not a square grid')
    if not 1 <= output_side <= input_side:
        raise ValueError(f'output side must be in [1,{input_side}], got {output_side}')
    grid = tokens.transpose(1, 2).reshape(batch, channels, input_side, input_side)
    pooled = F.adaptive_avg_pool2d(grid, (output_side, output_side))
    return pooled.flatten(2).transpose(1, 2)

class RandomErasing(object):

    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.33, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.EPSILON:
            return img
        for attempt in range(100):
            print(img.size())
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[1]
                return img
        return img

class ARTrackV2(BaseTracker):

    def __init__(self, params, dataset_name):
        super(ARTrackV2, self).__init__(params)
        network = build_artrackv2(params.cfg, training=False)
        checkpoint = _qir_torch_load(self.params.checkpoint, map_location='cpu')['net']
        if any((key.startswith('score_decoder.') for key in checkpoint)):
            checkpoint = {key.replace('score_decoder.', 'score_mlp.', 1) if key.startswith('score_decoder.') else key: value for (key, value) in checkpoint.items()}
        network.load_state_dict(checkpoint, strict=True)
        self.cfg = params.cfg
        self.bins = self.cfg.MODEL.BINS
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.update_ = False
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        self.erase = RandomErasing()
        if self.debug:
            if not self.use_visdom:
                self.save_dir = 'debug'
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                self._init_visdom(None, 1)
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.last_diagnostics = None
        self.last_uncertainty = None
        self.last_vlm_recovery = None
        self.uncertainty_gate = None
        self.vlm_recovery_controller = None
        if os.environ.get('ARTRACK_UNCERTAINTY_GATE_ENABLE', '0') == '1':
            raise RuntimeError('Legacy ensemble gate is not exported; use the QIRTrack BQP controller')
        if os.environ.get('ARTRACK_VLM_ROUTER_ENABLE', '0') == '1':
            if self.uncertainty_gate is None:
                raise RuntimeError('ARTRACK_VLM_ROUTER_ENABLE requires ARTRACK_UNCERTAINTY_GATE_ENABLE=1')
            from lib.models.qirtrack.core.vlm.artrackv2_vlm_recovery import VLMRecoveryController
            self.vlm_recovery_controller = VLMRecoveryController()

    def _set_tracking_template(self, image, bbox):
        (z_patch_arr, _, z_amask_arr) = sample_target(image, bbox, self.params.template_factor, output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template
            self.z_dict2 = template
        self.box_mask_z = None
        self.state = list(bbox)

    def initialize(self, image, info: dict):
        self._set_tracking_template(image, info['init_bbox'])
        self.frame_id = 0
        self.last_diagnostics = None
        self.last_uncertainty = None
        self.last_vlm_recovery = None
        if self.uncertainty_gate is not None:
            self.uncertainty_gate.reset()
        if self.vlm_recovery_controller is not None:
            sequence_name = info.get('sequence_name', 'sequence')
            self.vlm_recovery_controller.reset(image, info['init_bbox'], sequence_name)
            self.vlm_recovery_controller.set_template(self.z_patch_arr)
        if self.save_all_boxes:
            'save all predicted boxes'
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {'all_boxes': all_boxes_save}

    def track(self, image, info: dict=None):
        (H, W, _) = image.shape
        self.frame_id += 1
        previous_state_exact = tuple(self.state)
        previous_state = np.asarray(previous_state_exact, dtype=np.float32)
        diagnostic_profile = os.environ.get('ARTRACK_DIAGNOSTIC_PROFILE', 'full')
        (x_patch_arr, resize_factor, x_amask_arr) = sample_target(image, self.state, self.params.search_factor, output_sz=self.params.search_size)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            x_dict = search
            if self.update_:
                template = torch.concat([self.z_dict1.tensors.unsqueeze(0), self.z_dict2.unsqueeze(0)], dim=0)
            else:
                template = torch.concat([self.z_dict1.tensors.unsqueeze(0), self.z_dict2.tensors.unsqueeze(0)], dim=0)
            out_dict = self.network.forward(template=template, search=x_dict.tensors, ce_template_mask=self.box_mask_z)
        valid_coord_bins = self.bins * self.cfg.MODEL.RANGE
        coord_logits = out_dict['feat'][:4, 0, :valid_coord_bins]
        coord_probs = torch.softmax(coord_logits.float(), dim=-1)
        (coord_top2_prob, coord_top2_index) = coord_probs.topk(k=2, dim=-1)
        coord_log_probs = torch.log_softmax(coord_logits.float(), dim=-1)
        coord_entropy = -(coord_probs * coord_log_probs).sum(dim=-1)
        coord_entropy_normalized = coord_entropy / math.log(valid_coord_bins)
        coord_bins = torch.arange(valid_coord_bins, device=coord_probs.device, dtype=coord_probs.dtype)
        coord_mean_bin = (coord_probs * coord_bins).sum(dim=-1)
        coord_variance_bin = (coord_probs * (coord_bins - coord_mean_bin.unsqueeze(-1)).square()).sum(dim=-1)
        self.last_diagnostics = {'score_original': out_dict['score'].detach().float().cpu().numpy(), 'score_command': out_dict['score_command'].detach().float().cpu().numpy(), 'score_command_raw': out_dict['score_command_raw'].detach().float().cpu().numpy(), 'coord_argmax_index': out_dict['seqs'][0, :4].detach().cpu().numpy(), 'coord_argmax_prob_full_vocab': out_dict['class'][0, :4].detach().float().cpu().numpy(), 'coord_top2_index': coord_top2_index.detach().cpu().numpy(), 'coord_top2_prob': coord_top2_prob.detach().float().cpu().numpy(), 'coord_entropy': coord_entropy.detach().cpu().numpy(), 'coord_entropy_normalized': coord_entropy_normalized.detach().cpu().numpy(), 'coord_mean_bin': coord_mean_bin.detach().cpu().numpy(), 'coord_variance_bin': coord_variance_bin.detach().cpu().numpy(), 'score_feature_original': out_dict['score_feat'][0].detach().half().cpu().numpy(), 'score_command_feat_raw': out_dict['score_command_feat_raw'][0].detach().half().cpu().numpy(), 'score_command_feat': out_dict['score_command_feat'][0].detach().half().cpu().numpy()}
        if diagnostic_profile in {'full', 'logits', 'e2e', 'e2e_spatial'}:
            self.last_diagnostics['coord_logits'] = coord_logits.detach().half().cpu().numpy()
        if diagnostic_profile == 'full':
            self.last_diagnostics['coord_token_feat'] = out_dict['coord_token_feat'][0].detach().half().cpu().numpy()
        if diagnostic_profile in {'e2e', 'e2e_spatial'}:
            template_side = int(os.environ.get('ARTRACK_E2E_TEMPLATE_POOL_SIDE', '4'))
            search_side = int(os.environ.get('ARTRACK_E2E_SEARCH_POOL_SIDE', '6'))
            self.last_diagnostics['template_tokens'] = pool_square_tokens(out_dict['template_initial_tokens'], template_side)[0].detach().half().cpu().numpy()
            self.last_diagnostics['search_tokens'] = pool_square_tokens(out_dict['search_tokens'], search_side)[0].detach().half().cpu().numpy()
            if diagnostic_profile == 'e2e_spatial':
                from lib.models.qirtrack.core.localisation.spatial_provenance import compute_search_crop_metadata, pooled_valid_fraction, validate_resize_factor
                full_search_side = math.isqrt(out_dict['search_tokens'].shape[1])
                spatial = compute_search_crop_metadata(image.shape, previous_state_exact, self.params.search_factor, self.params.search_size, full_search_side, search_side)
                validate_resize_factor(spatial, resize_factor)
                spatial['search_valid_fraction'] = pooled_valid_fraction(x_amask_arr, search_side)
                self.last_diagnostics.update(spatial)
        pred_boxes = out_dict['seqs'][:, 0:4] / (self.bins - 1) - 0.5
        pred_boxes = pred_boxes.view(-1, 4).mean(dim=0)
        pred_new = pred_boxes
        pred_new[2] = pred_boxes[2] - pred_boxes[0]
        pred_new[3] = pred_boxes[3] - pred_boxes[1]
        pred_new[0] = pred_boxes[0] + pred_boxes[2] / 2
        pred_new[1] = pred_boxes[1] + pred_boxes[3] / 2
        pred_boxes = (pred_new * self.params.search_size / resize_factor).tolist()
        self.state = clip_box(self.map_box_back(pred_boxes, resize_factor), H, W, margin=10)
        if self.uncertainty_gate is not None:
            current_state = np.asarray(self.state, dtype=np.float32)
            previous_center = previous_state[:2] + 0.5 * previous_state[2:]
            current_center = current_state[:2] + 0.5 * current_state[2:]
            previous_diag = math.sqrt(max(float(previous_state[2] * previous_state[3]), 1e-12))
            displacement = float(np.linalg.norm(current_center - previous_center) / previous_diag)
            width_ratio = max(float(current_state[2]), 1e-12) / max(float(previous_state[2]), 1e-12)
            height_ratio = max(float(current_state[3]), 1e-12) / max(float(previous_state[3]), 1e-12)
            scale_change = math.sqrt(math.log(width_ratio) ** 2 + math.log(height_ratio) ** 2)
            self.last_uncertainty = self.uncertainty_gate.step(self.last_diagnostics, self.frame_id, displacement, scale_change)
        if self.vlm_recovery_controller is not None:
            self.last_vlm_recovery = self.vlm_recovery_controller.step(self.last_uncertainty, self.frame_id, image, self.state)
            call_result = self.last_vlm_recovery.get('call', {})
            config = self.vlm_recovery_controller.config
            if config.apply_recovery and call_result.get('status') == 'completed' and call_result.get('target_present') and (call_result.get('bbox_xywh_pixels') is not None):
                recovered = clip_box(call_result['bbox_xywh_pixels'], H, W, margin=10)
                self._set_tracking_template(image, recovered)
                if self.uncertainty_gate is not None:
                    self.uncertainty_gate.reset()
                self.vlm_recovery_controller.accept_recovery(image, recovered)
                self.last_vlm_recovery['recovery_applied'] = True
        if self.debug:
            if not self.use_visdom:
                (x1, y1, w, h) = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, '%04d.jpg' % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')
                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break
        if self.save_all_boxes:
            'save all predictions'
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {'target_bbox': self.state, 'all_boxes': all_boxes_save}
        else:
            return {'target_bbox': self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        (cx_prev, cy_prev) = (self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3])
        (cx, cy, w, h) = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        (cx_prev, cy_prev) = (self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3])
        (cx, cy, w, h) = pred_box.unbind(-1)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        (conv_features, enc_attn_weights, dec_attn_weights) = ([], [], [])
        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(lambda self, input, output: enc_attn_weights.append(output[1]))
        self.enc_attn_weights = enc_attn_weights

def get_tracker_class():
    return ARTrackV2
