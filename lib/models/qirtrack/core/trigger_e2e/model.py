"""Cross-spatial and causal-temporal Trigger-E2E quality model."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from lib.models.qirtrack.core.trigger_e2e.distribution import iou_support_points

class CoordinateDistributionEncoder(nn.Module):
    """Learn directly from four complete coordinate distributions."""

    def __init__(self, d_model: int, heads: int, coordinates: int=4) -> None:
        super().__init__()
        self.coordinates = coordinates
        self.convolution = nn.Sequential(nn.Conv1d(1, 16, kernel_size=9, stride=4, padding=4), nn.GELU(), nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4), nn.GELU(), nn.AdaptiveAvgPool1d(4))
        self.projection = nn.Linear(32 * 4, d_model)
        self.coordinate_embedding = nn.Parameter(torch.empty(1, coordinates, d_model))
        nn.init.trunc_normal_(self.coordinate_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=4 * d_model, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.coordinate_attention = nn.TransformerEncoder(layer, num_layers=1)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 3 or logits.shape[1] != self.coordinates:
            raise ValueError(f'coord logits must be [B,{self.coordinates},V], got {tuple(logits.shape)}')
        probability = torch.softmax(logits.float(), dim=-1)
        (batch, coordinates, vocabulary) = probability.shape
        encoded = self.convolution(probability.reshape(batch * coordinates, 1, vocabulary))
        encoded = self.projection(encoded.flatten(1)).reshape(batch, coordinates, -1)
        encoded = encoded + self.coordinate_embedding
        encoded = self.coordinate_attention(encoded)
        return self.output_norm(encoded.mean(dim=1))

def causal_box_geometry(pred_bbox: torch.Tensor, history_mask: torch.Tensor | None=None) -> torch.Tensor:
    """Convert absolute xywh boxes into tracker-independent causal changes."""
    if pred_bbox.ndim != 3 or pred_bbox.shape[-1] != 4:
        raise ValueError(f'pred_bbox must be [B,K,4], got {tuple(pred_bbox.shape)}')
    if history_mask is not None and history_mask.shape != pred_bbox.shape[:2]:
        raise ValueError('history_mask must match pred_bbox [B,K]')
    box = pred_bbox.float()
    previous = torch.cat([box[:, :1], box[:, :-1]], dim=1)
    centre = box[..., :2] + 0.5 * box[..., 2:]
    previous_centre = previous[..., :2] + 0.5 * previous[..., 2:]
    previous_scale = torch.sqrt(previous[..., 2:].prod(dim=-1).clamp_min(1e-06))
    displacement = (centre - previous_centre) / previous_scale.unsqueeze(-1)
    log_size_ratio = torch.log(box[..., 2:].clamp_min(1e-06) / previous[..., 2:].clamp_min(1e-06))
    log_aspect = torch.log(box[..., 2].clamp_min(1e-06) / box[..., 3].clamp_min(1e-06)).unsqueeze(-1)
    log_area_change = log_size_ratio.sum(dim=-1, keepdim=True)
    geometry = torch.cat([displacement, log_size_ratio, log_aspect, log_area_change], dim=-1)
    if history_mask is not None:
        valid = history_mask.bool()
        previous_valid = torch.cat([valid[:, :1], valid[:, :-1]], dim=1)
        transition_valid = valid & previous_valid
        geometry[..., :4] *= transition_valid.unsqueeze(-1)
        geometry[..., 5:] *= transition_valid.unsqueeze(-1)
    return geometry

class SpatialFrameEncoder(nn.Module):
    """Template-condition search tokens, then pool them with a quality query."""

    def __init__(self, token_dim: int, d_model: int, heads: int, tracker_score_dim: int) -> None:
        super().__init__()
        self.tracker_score_dim = tracker_score_dim
        self.template_projection = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, d_model))
        self.search_projection = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, d_model))
        self.search_self_attention = nn.MultiheadAttention(d_model, heads, dropout=0.1, batch_first=True)
        self.search_self_norm = nn.LayerNorm(d_model)
        self.template_cross_attention = nn.MultiheadAttention(d_model, heads, dropout=0.1, batch_first=True)
        self.template_cross_norm = nn.LayerNorm(d_model)
        self.quality_query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.quality_query, std=0.02)
        self.quality_attention = nn.MultiheadAttention(d_model, heads, dropout=0.1, batch_first=True)
        self.coordinate_encoder = CoordinateDistributionEncoder(d_model, heads)
        if tracker_score_dim > 0:
            self.score_encoder = nn.Sequential(nn.LayerNorm(tracker_score_dim), nn.Linear(tracker_score_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        else:
            self.score_encoder = None
            self.missing_score = nn.Parameter(torch.zeros(1, d_model))
        self.geometry_encoder = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.fusion = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, template_tokens: torch.Tensor, search_tokens: torch.Tensor, coord_logits: torch.Tensor, tracker_score: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        template = self.template_projection(template_tokens.float())
        search = self.search_projection(search_tokens.float())
        (self_context, _) = self.search_self_attention(search, search, search, need_weights=False)
        search = self.search_self_norm(search + self_context)
        (matched_template, _) = self.template_cross_attention(search, template, template, need_weights=False)
        search = self.template_cross_norm(search + matched_template)
        query = self.quality_query.expand(search.shape[0], -1, -1)
        (quality, _) = self.quality_attention(query, search, search, need_weights=False)
        quality = quality[:, 0]
        coordinate = self.coordinate_encoder(coord_logits)
        if self.score_encoder is None:
            score = self.missing_score.expand(search.shape[0], -1)
        else:
            if tracker_score.shape[-1] != self.tracker_score_dim:
                raise ValueError(f'tracker score dim {tracker_score.shape[-1]} != {self.tracker_score_dim}')
            score = self.score_encoder(tracker_score.float())
        geometry_embedding = self.geometry_encoder(geometry.float())
        return self.fusion(torch.cat([quality, coordinate, score, geometry_embedding], dim=-1))

class UnifiedTemporalBlock(nn.Module):
    """Update one current-frame query from causal frame memory.

    The residual path always carries the current observation. History enters
    through attention inside the same representation; there is no second IoU
    expert and no scalar router deciding between two predictions.
    """

    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, heads, dropout=0.1, batch_first=True)
        self.attention_dropout = nn.Dropout(0.1)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(4 * d_model, d_model), nn.Dropout(0.1))
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, current: torch.Tensor, causal_memory: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        if current.ndim != 2 or causal_memory.ndim != 3:
            raise ValueError('current/memory must be [B,D] and [B,K,D]')
        if history_mask.shape != causal_memory.shape[:2]:
            raise ValueError('history_mask must match causal memory [B,K]')
        query = self.query_norm(current).unsqueeze(1)
        memory = self.memory_norm(causal_memory)
        (context, _) = self.cross_attention(query, memory, memory, key_padding_mask=~history_mask.bool(), need_weights=False)
        updated = current + self.attention_dropout(context[:, 0])
        updated = updated + self.ffn(self.ffn_norm(updated))
        return self.output_norm(updated)

class TriggerE2E(nn.Module):
    """Predict an ordered IoU distribution from current and past tracker tokens."""

    def __init__(self, token_dim: int, tracker_score_dim: int, d_model: int=128, heads: int=4, history_length: int=16, temporal_layers: int=2, iou_bins: int=50, use_temporal: bool=True, temporal_fusion: str='replace', transition_auxiliary_outputs: int=0, distribution_gate_mode: str='standard') -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError('d_model must be divisible by heads')
        if history_length < 1 or iou_bins < 2:
            raise ValueError('history_length must be positive and iou_bins >= 2')
        if temporal_fusion not in {'replace', 'gated_residual', 'dual_distribution', 'unified_cross_attention'}:
            raise ValueError(f'unknown temporal fusion: {temporal_fusion}')
        if not use_temporal and temporal_fusion != 'replace':
            raise ValueError('temporal fusion requires use_temporal=True')
        if transition_auxiliary_outputs < 0:
            raise ValueError('transition_auxiliary_outputs must be non-negative')
        if transition_auxiliary_outputs and (not use_temporal):
            raise ValueError('transition auxiliary head requires use_temporal=True')
        if distribution_gate_mode not in {'standard', 'capacity_control', 'innovation'}:
            raise ValueError(f'unknown distribution gate mode: {distribution_gate_mode}')
        if distribution_gate_mode != 'standard' and temporal_fusion != 'dual_distribution':
            raise ValueError('A8 gate modes require dual-distribution temporal fusion')
        self.history_length = history_length
        self.iou_bins = iou_bins
        self.use_temporal = use_temporal
        self.temporal_fusion = temporal_fusion
        self.transition_auxiliary_outputs = transition_auxiliary_outputs
        self.distribution_gate_mode = distribution_gate_mode
        self.frame_encoder = SpatialFrameEncoder(token_dim, d_model, heads, tracker_score_dim)
        self.temporal_position = nn.Parameter(torch.empty(1, history_length, d_model))
        self.sequence_start = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        temporal_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=4 * d_model, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        self.temporal_norm = nn.LayerNorm(d_model)
        if temporal_fusion == 'unified_cross_attention':
            self.unified_temporal = UnifiedTemporalBlock(d_model, heads)
        else:
            self.unified_temporal = None
        if temporal_fusion == 'gated_residual':
            self.temporal_residual = nn.Linear(d_model, d_model)
            self.temporal_gate = nn.Linear(2 * d_model, d_model)
            nn.init.zeros_(self.temporal_residual.weight)
            nn.init.zeros_(self.temporal_residual.bias)
            nn.init.zeros_(self.temporal_gate.weight)
            nn.init.constant_(self.temporal_gate.bias, -2.0)
        else:
            self.temporal_residual = None
            self.temporal_gate = None
        self.iou_head = nn.Linear(d_model, iou_bins)
        if temporal_fusion == 'dual_distribution':
            self.temporal_iou_head = nn.Linear(d_model, iou_bins)
            self.distribution_gate = nn.Linear(2 * d_model, 1)
            nn.init.zeros_(self.distribution_gate.weight)
            nn.init.constant_(self.distribution_gate.bias, -2.0)
            if distribution_gate_mode in {'capacity_control', 'innovation'}:
                routing_dim = 4 * d_model + 4
                self.innovation_gate = nn.Sequential(nn.LayerNorm(routing_dim), nn.Linear(routing_dim, d_model), nn.GELU(), nn.Linear(d_model, 1))
                nn.init.zeros_(self.innovation_gate[-1].weight)
                nn.init.zeros_(self.innovation_gate[-1].bias)
            else:
                self.innovation_gate = None
        else:
            self.temporal_iou_head = None
            self.distribution_gate = None
            self.innovation_gate = None
        if transition_auxiliary_outputs:
            self.transition_auxiliary_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, transition_auxiliary_outputs))
        else:
            self.transition_auxiliary_head = None
        support = iou_support_points(iou_bins)
        self.register_buffer('iou_bin_centres', support, persistent=True)

    def encode_frames(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        template = batch['template_tokens']
        search = batch['search_tokens']
        coord = batch['coord_logits']
        score = batch['tracker_score']
        bbox = batch['pred_bbox']
        if template.ndim != 4 or search.ndim != 4 or coord.ndim != 4:
            raise ValueError('token and coordinate inputs must include [B,K,...] dimensions')
        (batch_size, history) = template.shape[:2]
        if history != self.history_length:
            raise ValueError(f'history length {history} != configured {self.history_length}')
        geometry = causal_box_geometry(bbox, batch['history_mask'])
        flatten = lambda value: value.reshape(batch_size * history, *value.shape[2:])
        frame = self.frame_encoder(flatten(template), flatten(search), flatten(coord), flatten(score), flatten(geometry))
        return frame.reshape(batch_size, history, -1)

    def encode_temporal(self, frame_tokens: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        if history_mask.shape != frame_tokens.shape[:2]:
            raise ValueError('history_mask must have shape [B,K]')
        valid = history_mask.bool()
        values = frame_tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        padding_mask = ~valid
        has_padding = padding_mask.any(dim=1)
        if torch.any(has_padding):
            values = values.clone()
            values[has_padding, 0] = self.sequence_start[0, 0]
            padding_mask = padding_mask.clone()
            padding_mask[has_padding, 0] = False
        values = values + self.temporal_position[:, :values.shape[1]]
        causal_mask = torch.triu(torch.ones(values.shape[1], values.shape[1], dtype=torch.bool, device=values.device), diagonal=1)
        encoded = self.temporal_encoder(values, mask=causal_mask, src_key_padding_mask=padding_mask)
        return self.temporal_norm(encoded)

    def routing_features(self, current_token: torch.Tensor, temporal_token: torch.Tensor, current_probability: torch.Tensor, temporal_probability: torch.Tensor) -> torch.Tensor:
        """Build capacity-matched A8 gate evidence without future information."""
        delta = current_token - temporal_token
        absolute_delta = delta.abs()
        cosine = F.cosine_similarity(current_token, temporal_token, dim=-1).unsqueeze(-1)
        current_mean = (current_probability * self.iou_bin_centres).sum(dim=-1)
        temporal_mean = (temporal_probability * self.iou_bin_centres).sum(dim=-1)
        mean_gap = (temporal_mean - current_mean).unsqueeze(-1)
        total_variation = (0.5 * (current_probability - temporal_probability).abs().sum(dim=-1)).unsqueeze(-1)
        midpoint = 0.5 * (current_probability + temporal_probability)
        epsilon = 1e-08
        js_divergence = 0.5 * ((current_probability * (current_probability.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log())).sum(dim=-1) + (temporal_probability * (temporal_probability.clamp_min(epsilon).log() - midpoint.clamp_min(epsilon).log())).sum(dim=-1))
        scalars = torch.cat([cosine, mean_gap, total_variation, js_divergence.unsqueeze(-1)], dim=-1)
        if self.distribution_gate_mode == 'capacity_control':
            delta = torch.zeros_like(delta)
            absolute_delta = torch.zeros_like(absolute_delta)
            scalars = torch.zeros_like(scalars)
        return torch.cat([current_token, temporal_token, delta, absolute_delta, scalars], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frame_tokens = self.encode_frames(batch)
        current_frame = frame_tokens[:, -1]
        gate = None
        transition_delta = None
        current_logits = self.iou_head(current_frame)
        current_probability = torch.softmax(current_logits, dim=-1)
        if self.use_temporal:
            temporal_tokens = self.encode_temporal(frame_tokens, batch['history_mask'])
            temporal_current = temporal_tokens[:, -1]
            if self.training and self.transition_auxiliary_head is not None:
                transition_delta = self.transition_auxiliary_head(temporal_current)
            if self.temporal_fusion == 'gated_residual':
                if self.temporal_residual is None or self.temporal_gate is None:
                    raise RuntimeError('gated residual modules are not initialized')
                residual = self.temporal_residual(temporal_current)
                gate = torch.sigmoid(self.temporal_gate(torch.cat([current_frame, temporal_current], dim=-1)))
                current = current_frame + gate * residual
            elif self.temporal_fusion == 'dual_distribution':
                current = current_frame
            elif self.temporal_fusion == 'unified_cross_attention':
                if self.unified_temporal is None:
                    raise RuntimeError('unified temporal block is not initialized')
                current = self.unified_temporal(current_frame, temporal_tokens, batch['history_mask'])
            else:
                current = temporal_current
        else:
            temporal_tokens = frame_tokens
            current = current_frame
        if self.temporal_fusion == 'dual_distribution':
            if self.temporal_iou_head is None or self.distribution_gate is None:
                raise RuntimeError('dual-distribution modules are not initialized')
            temporal_logits = self.temporal_iou_head(temporal_current)
            temporal_probability = torch.softmax(temporal_logits, dim=-1)
            if self.distribution_gate_mode == 'standard':
                gate_logits = self.distribution_gate(torch.cat([current_frame, temporal_current], dim=-1))
            else:
                if self.innovation_gate is None:
                    raise RuntimeError('A8 routing gate is not initialized')
                gate_logits = self.innovation_gate(self.routing_features(current_frame, temporal_current, current_probability, temporal_probability))
            gate = torch.sigmoid(gate_logits)
            probability = (1.0 - gate) * current_probability + gate * temporal_probability
            logits = probability.clamp_min(1e-08).log()
        else:
            temporal_logits = None
            logits = self.iou_head(current)
            probability = torch.softmax(logits, dim=-1)
        mean_iou = (probability * self.iou_bin_centres).sum(dim=-1)
        return {'iou_logits': logits, 'iou_probability': probability, 'predicted_iou': mean_iou, 'frame_tokens': frame_tokens, 'temporal_tokens': temporal_tokens, 'current_token': current, 'temporal_gate': gate, 'current_iou_logits': current_logits, 'current_iou_probability': current_probability, 'temporal_iou_logits': temporal_logits, 'transition_delta': transition_delta}
