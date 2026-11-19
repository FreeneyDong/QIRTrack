"""C1-Context data and model definitions.

The arbitrary-box path consumes candidate-specific context tokens and never
coordinate logits from the tracker's raw box decoder.  Frozen U1 frame tokens
are used only for strictly previous frames.
"""
from __future__ import annotations
import copy
import json
import math
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from lib.models.qirtrack.core.localisation.candidate_data import candidate_relative_geometry
from lib.models.qirtrack.core.trigger_e2e.data import discover_sequence_files, train_sequence_names
from lib.models.qirtrack.core.trigger_e2e.model import causal_box_geometry

class CandidateContextDataset(Dataset[dict[str, torch.Tensor]]):

    def __init__(self, context_root: Path, base_feature_root: Path, train_manifest: Path, evaluation_manifest: Path, history_length: int=16, cache_sequences: int=2, split: str='fit', evaluation_domain: str | None=None, recent_template_root: Path | None=None) -> None:
        fit_names = set(train_sequence_names(train_manifest, evaluation_manifest))
        all_context_paths = discover_sequence_files(context_root)
        if split == 'fit':
            unknown = sorted(set(all_context_paths) - fit_names)
            if unknown:
                raise ValueError(f'context root contains non-fit sequences: {unknown[:10]}')
            context_paths = all_context_paths
        elif split == 'development':
            if evaluation_domain is None:
                raise ValueError('evaluation_domain is required for development')
            manifest = json.loads(evaluation_manifest.read_text(encoding='utf-8'))
            allowed = set(manifest['evaluation_domains'][evaluation_domain])
            overlap = sorted(allowed & fit_names)
            if overlap:
                raise ValueError(f'development/fit overlap: {overlap[:10]}')
            missing_context = sorted(allowed - set(all_context_paths))
            if missing_context:
                raise FileNotFoundError(f'missing development context features: {missing_context[:10]}')
            context_paths = {name: all_context_paths[name] for name in allowed}
        else:
            raise ValueError("split must be 'fit' or 'development'")
        base_paths = discover_sequence_files(base_feature_root)
        missing = sorted(set(context_paths) - set(base_paths))
        if missing:
            raise FileNotFoundError(f'missing base features: {missing[:10]}')
        self.names = sorted(context_paths)
        if not self.names:
            raise FileNotFoundError(f'no context features in {context_root}')
        self.context_paths = context_paths
        self.base_paths = base_paths
        self.recent_template_paths = None
        if recent_template_root is not None:
            recent_paths = discover_sequence_files(recent_template_root)
            missing_recent = sorted(set(context_paths) - set(recent_paths))
            if missing_recent:
                raise FileNotFoundError(f'missing recent-template features: {missing_recent[:10]}')
            self.recent_template_paths = recent_paths
        self.history_length = history_length
        self.return_metadata = split == 'development'
        self.sequence_indices = {name: index for (index, name) in enumerate(self.names)}
        self.index: list[tuple[str, int]] = []
        for name in self.names:
            with np.load(context_paths[name], allow_pickle=False) as data:
                required = {'frame', 'candidate_bbox', 'candidate_iou', 'candidate_context_tokens', 'candidate_valid_fraction'}
                absent = sorted(required - set(data.files))
                if absent:
                    raise ValueError(f'{context_paths[name]} missing {absent}')
                if str(np.asarray(data['representation_mode']).item()) != 'candidate_context_backbone':
                    raise ValueError(f'{context_paths[name]} has wrong representation mode')
                self.index.extend(((name, i) for i in range(len(data['frame']))))
        self._cache: OrderedDict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray] | None]] = OrderedDict()
        self.cache_sequences = cache_sequences

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, name: str):
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        with np.load(self.context_paths[name], allow_pickle=False) as source:
            context = {key: np.asarray(source[key]) for key in source.files}
        base_fields = ('frame', 'template_tokens', 'search_tokens', 'coord_logits', 'tracker_score', 'pred_bbox')
        with np.load(self.base_paths[name], allow_pickle=False) as source:
            base = {key: np.asarray(source[key]) for key in base_fields}
        recent = None
        if self.recent_template_paths is not None:
            required_recent = ('frame', 'recent_template_tokens', 'recent_template_age', 'recent_template_valid', 'recent_template_similarity')
            with np.load(self.recent_template_paths[name], allow_pickle=False) as source:
                absent = sorted(set(required_recent) - set(source.files))
                if absent:
                    raise ValueError(f'{self.recent_template_paths[name]} missing {absent}')
                recent = {key: np.asarray(source[key]) for key in required_recent}
            recent_frame = np.asarray(recent['frame'])
            context_frame = np.asarray(context['frame'])
            if len(np.unique(recent_frame)) != len(recent_frame):
                raise ValueError(f'{name}: duplicate recent-template frames')
            missing_frame = sorted(set(context_frame.tolist()) - set(recent_frame.tolist()))
            if missing_frame:
                raise ValueError(f'{name}: recent-template features missing context frames {missing_frame[:10]}')
        self._cache[name] = (context, base, recent)
        while len(self._cache) > self.cache_sequences:
            self._cache.popitem(last=False)
        return (context, base, recent)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        (name, local) = self.index[item]
        (context, base, recent) = self._load(name)
        frame = int(context['frame'][local])
        matches = np.flatnonzero(base['frame'] == frame)
        if len(matches) != 1:
            raise ValueError(f'{name}: frame {frame} not unique in base features')
        current = int(matches[0])
        raw_indices = np.arange(current - self.history_length + 1, current + 1)
        history_mask = raw_indices >= 0
        indices = np.clip(raw_indices, 0, current)
        template = base['template_tokens']
        if template.ndim == 2:
            template = np.broadcast_to(template[None], (self.history_length, *template.shape))
        else:
            template = template[indices]
        candidates = context['candidate_bbox'][local]
        raw = base['pred_bbox'][current]
        result = {'template_tokens': torch.from_numpy(np.array(template, copy=True)), 'search_tokens': torch.from_numpy(np.array(base['search_tokens'][indices], copy=True)), 'coord_logits': torch.from_numpy(np.array(base['coord_logits'][indices], copy=True)), 'tracker_score': torch.from_numpy(np.array(base['tracker_score'][indices], copy=True)), 'pred_bbox': torch.from_numpy(np.array(base['pred_bbox'][indices], copy=True)), 'history_mask': torch.from_numpy(history_mask), 'candidate_context_tokens': torch.from_numpy(np.array(context['candidate_context_tokens'][local], copy=True)), 'candidate_geometry': torch.from_numpy(candidate_relative_geometry(candidates, raw)), 'candidate_valid_fraction': torch.from_numpy(np.array(context['candidate_valid_fraction'][local], copy=True)), 'candidate_iou': torch.from_numpy(np.array(context['candidate_iou'][local], copy=True))}
        if recent is not None:
            recent_matches = np.flatnonzero(recent['frame'] == frame)
            if len(recent_matches) != 1:
                raise ValueError(f'{name}: frame {frame} not unique in recent features')
            recent_local = int(recent_matches[0])
            result.update({'recent_template_tokens': torch.from_numpy(np.array(recent['recent_template_tokens'][recent_local], copy=True)), 'recent_template_age': torch.tensor(float(recent['recent_template_age'][recent_local]), dtype=torch.float32), 'recent_template_valid': torch.tensor(bool(recent['recent_template_valid'][recent_local]), dtype=torch.bool), 'recent_template_similarity': torch.tensor(float(recent['recent_template_similarity'][recent_local]), dtype=torch.float32)})
        if self.return_metadata:
            result['sequence_index'] = torch.tensor(self.sequence_indices[name], dtype=torch.int64)
            result['frame'] = torch.tensor(frame, dtype=torch.int64)
            result['candidate_bbox'] = torch.from_numpy(np.array(candidates, copy=True))
        return result

class CandidateContextIoU(nn.Module):
    """Candidate-aligned current evidence plus frozen, strictly past U1 memory."""

    def __init__(self, base_model: nn.Module, heads: int=4, use_history: bool=True, template_mode: str='static', recent_similarity_threshold: float=-1.0, recent_max_age: int=128) -> None:
        super().__init__()
        self.use_history = bool(use_history)
        if template_mode not in {'static', 'recent'}:
            raise ValueError("template_mode must be 'static' or 'recent'")
        self.template_mode = template_mode
        self.recent_similarity_threshold = float(recent_similarity_threshold)
        self.recent_max_age = int(recent_max_age)
        self.base_model = base_model
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        d_model = int(base_model.iou_head.in_features)
        encoder = base_model.frame_encoder
        self.context_projection = copy.deepcopy(encoder.search_projection)
        self.context_self_attention = nn.MultiheadAttention(d_model, heads, 0.1, batch_first=True)
        self.context_self_norm = nn.LayerNorm(d_model)
        self.template_attention = nn.MultiheadAttention(d_model, heads, 0.1, batch_first=True)
        self.template_norm = nn.LayerNorm(d_model)
        if self.template_mode == 'recent':
            self.recent_source_embedding = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.trunc_normal_(self.recent_source_embedding, std=0.02)
            self.recent_age_encoder = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.context_attention = nn.MultiheadAttention(d_model, heads, 0.1, batch_first=True)
        self.context_norm = nn.LayerNorm(d_model)
        self.geometry_encoder = nn.Sequential(nn.LayerNorm(5), nn.Linear(5, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.history_attention = nn.MultiheadAttention(d_model, heads, 0.1, batch_first=True)
        self.history_norm = nn.LayerNorm(d_model)
        self.fusion = nn.Sequential(nn.LayerNorm(3 * d_model), nn.Linear(3 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * d_model, d_model), nn.LayerNorm(d_model))
        self.iou_head = copy.deepcopy(base_model.iou_head)
        self.register_buffer('iou_bin_centres', base_model.iou_bin_centres.detach().clone())

    def train(self, mode: bool=True):
        super().train(mode)
        self.base_model.eval()
        return self

    def adapter_state_dict(self):
        return {k: v for (k, v) in self.state_dict().items() if not k.startswith('base_model.') and k != 'iou_bin_centres'}

    def _encode_strict_past(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the frozen frame encoder without even reading current raw evidence."""
        history = batch['template_tokens'].shape[1] - 1
        if history < 1:
            raise ValueError('C1-Context requires at least one history slot')
        mask = batch['history_mask'][:, :-1]
        geometry = causal_box_geometry(batch['pred_bbox'][:, :-1], mask)
        batch_size = geometry.shape[0]
        flatten = lambda value: value[:, :-1].reshape(batch_size * history, *value.shape[2:])
        frame = self.base_model.frame_encoder(flatten(batch['template_tokens']), flatten(batch['search_tokens']), flatten(batch['coord_logits']), flatten(batch['tracker_score']), geometry.reshape(batch_size * history, -1))
        return frame.reshape(batch_size, history, -1)

    def forward(self, batch: dict[str, torch.Tensor]):
        tokens = batch['candidate_context_tokens'].float()
        if tokens.ndim != 4:
            raise ValueError('candidate_context_tokens must be [B,N,T,C]')
        (batch_size, count, token_count, _) = tokens.shape
        flat = tokens.reshape(batch_size * count, token_count, -1)
        context = self.context_projection(flat)
        (update, _) = self.context_self_attention(context, context, context, need_weights=False)
        context = self.context_self_norm(context + update)
        template = self.base_model.frame_encoder.template_projection(batch['template_tokens'][:, -1].float())
        template_mask = None
        if self.template_mode == 'recent':
            required = {'recent_template_tokens', 'recent_template_age', 'recent_template_valid', 'recent_template_similarity'}
            absent = sorted(required - set(batch))
            if absent:
                raise ValueError(f'recent template mode missing {absent}')
            recent = self.base_model.frame_encoder.template_projection(batch['recent_template_tokens'].float())
            age = torch.log1p(batch['recent_template_age'].float()).clamp_min(0.0)
            age = age / math.log1p(max(self.recent_max_age, 1))
            recent = recent + self.recent_source_embedding + self.recent_age_encoder(age[:, None])[:, None, :]
            accepted = batch['recent_template_valid'].bool() & (batch['recent_template_similarity'].float() >= self.recent_similarity_threshold)
            initial_count = template.shape[1]
            recent_count = recent.shape[1]
            template = torch.cat([template, recent], dim=1)
            template_mask = torch.cat([torch.zeros(batch_size, initial_count, dtype=torch.bool, device=template.device), (~accepted)[:, None].expand(-1, recent_count)], dim=1)
        template = template[:, None].expand(-1, count, -1, -1).reshape(batch_size * count, template.shape[1], -1)
        if template_mask is not None:
            template_mask = template_mask[:, None].expand(-1, count, -1).reshape(batch_size * count, template_mask.shape[1])
        (update, _) = self.template_attention(context, template, template, key_padding_mask=template_mask, need_weights=False)
        context = self.template_norm(context + update)
        query = self.query.expand(batch_size * count, -1, -1)
        (update, _) = self.context_attention(query, context, context, need_weights=False)
        current = self.context_norm(query + update)
        valid = batch['candidate_valid_fraction'].float().reshape(batch_size * count, 1)
        geometry = self.geometry_encoder(torch.cat([batch['candidate_geometry'].float().reshape(batch_size * count, 4), valid], dim=-1))
        current = current[:, 0] + geometry
        if self.use_history:
            with torch.no_grad():
                previous_tokens = self._encode_strict_past(batch)
            previous_mask = batch['history_mask'][:, :-1].bool()
            memory = previous_tokens[:, None].expand(-1, count, -1, -1).reshape(batch_size * count, previous_tokens.shape[1], -1)
            memory_mask = previous_mask[:, None].expand(-1, count, -1).reshape(batch_size * count, previous_mask.shape[1])
            empty = ~memory_mask.any(dim=1)
            if torch.any(empty):
                memory = memory.clone()
                memory_mask = memory_mask.clone()
                memory[empty, 0] = 0.0
                memory_mask[empty, 0] = True
            (history, _) = self.history_attention(current[:, None], memory, memory, key_padding_mask=~memory_mask, need_weights=False)
            history = self.history_norm(current + history[:, 0])
        else:
            history = self.history_norm(current)
        fused = self.fusion(torch.cat([current, history, geometry], dim=-1))
        logits = self.iou_head(fused).reshape(batch_size, count, -1)
        probability = torch.softmax(logits, dim=-1)
        return {'iou_logits': logits, 'predicted_iou': (probability * self.iou_bin_centres).sum(dim=-1)}
