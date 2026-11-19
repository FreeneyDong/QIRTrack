"""SearchMap-24 candidate IoU data contract and model.

The current candidate is represented by differentiable ROI samples from the
single canonical 24x24 ARTrack search map.  Frozen U1 history remains the exact
6x6 representation used during Trigger-E2E training.
"""
from __future__ import annotations
import json
import math
import hashlib
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from lib.models.qirtrack.core.localisation.candidate_data import candidate_relative_geometry
from lib.models.qirtrack.core.localisation.context_candidate import CandidateContextIoU
from lib.models.qirtrack.core.localisation.spatial_provenance import xywh_image_to_search_normalized
from lib.models.qirtrack.core.trigger_e2e.data import discover_sequence_files, train_sequence_names
SEARCHMAP_CACHE_SCHEMA = 'PrPoolSearchMapPackedCache-v1'
SEARCHMAP_MMAP_CACHE_SCHEMA = 'PrPoolSearchMapMMapCache-v1'

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

class PackedSearchMapCandidateDataset(Dataset[dict[str, torch.Tensor]]):
    """Tensor-only SearchMap cache; values are identical to the NPZ dataset."""

    def __init__(self, cache_root: Path, train_manifest: Path, evaluation_manifest: Path, cache_sequences: int=4) -> None:
        index_path = cache_root / 'index.json'
        metadata = json.loads(index_path.read_text(encoding='utf-8'))
        if metadata.get('schema') != SEARCHMAP_CACHE_SCHEMA:
            raise ValueError('unsupported SearchMap packed-cache schema')
        if metadata.get('complete_fit_cache') is not True:
            raise ValueError('incomplete SearchMap cache cannot train a model')
        for (key, path) in (('train_manifest_sha256', train_manifest), ('evaluation_manifest_sha256', evaluation_manifest)):
            if metadata.get(key) != _sha256(path):
                raise ValueError(f'SearchMap packed cache {key} mismatch')
        fit_names = set(train_sequence_names(train_manifest, evaluation_manifest))
        entries = metadata['sequences']
        self.names = [str(entry['name']) for entry in entries]
        if len(self.names) != len(set(self.names)):
            raise ValueError('duplicate SearchMap cache identities')
        unknown = sorted(set(self.names) - fit_names)
        if unknown:
            raise ValueError(f'SearchMap cache contains non-fit identities: {unknown[:10]}')
        self.paths = {str(entry['name']): cache_root / str(entry['file']) for entry in entries}
        self.index = [(str(entry['name']), local) for entry in entries for local in range(int(entry['length']))]
        self.cache_sequences = int(cache_sequences)
        self._cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, name: str) -> dict[str, torch.Tensor]:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        payload = torch.load(self.paths[name], map_location='cpu')
        if payload.get('schema') != SEARCHMAP_CACHE_SCHEMA:
            raise ValueError(f'{name}: invalid packed SearchMap payload')
        samples = payload['samples']
        self._cache[name] = samples
        while len(self._cache) > self.cache_sequences:
            self._cache.popitem(last=False)
        return samples

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        (name, local) = self.index[item]
        return {key: value[local] for (key, value) in self._load(name).items()}

class MMapSearchMapCandidateDataset(Dataset[dict[str, torch.Tensor]]):
    """Read-only, field-sharded cache with the packed dataset's exact values.

    Each field is one uncompressed ``.npy`` array whose first dimension follows
    the packed cache's global frame order.  ``numpy.load(..., mmap_mode="r")``
    leaves pages in the operating-system page cache, so independent seed runs do
    not deserialize every sequence into a private 20+ GiB Python process.
    """

    def __init__(self, cache_root: Path, train_manifest: Path, evaluation_manifest: Path) -> None:
        index_path = cache_root / 'index.json'
        metadata = json.loads(index_path.read_text(encoding='utf-8'))
        if metadata.get('schema') != SEARCHMAP_MMAP_CACHE_SCHEMA:
            raise ValueError('unsupported SearchMap mmap-cache schema')
        if metadata.get('complete_fit_cache') is not True:
            raise ValueError('incomplete SearchMap mmap cache cannot train a model')
        for (key, path) in (('train_manifest_sha256', train_manifest), ('evaluation_manifest_sha256', evaluation_manifest)):
            if metadata.get(key) != _sha256(path):
                raise ValueError(f'SearchMap mmap cache {key} mismatch')
        entries = metadata.get('sequences', [])
        self.names = [str(entry['name']) for entry in entries]
        if len(self.names) != len(set(self.names)):
            raise ValueError('duplicate SearchMap mmap-cache identities')
        fit_names = set(train_sequence_names(train_manifest, evaluation_manifest))
        unknown = sorted(set(self.names) - fit_names)
        if unknown:
            raise ValueError(f'SearchMap mmap cache contains non-fit identities: {unknown[:10]}')
        self.index = [(str(entry['name']), local) for entry in entries for local in range(int(entry['length']))]
        if len(self.index) != int(metadata.get('length', -1)):
            raise ValueError('SearchMap mmap-cache global length mismatch')
        field_metadata = metadata.get('fields')
        if not isinstance(field_metadata, dict) or not field_metadata:
            raise ValueError('SearchMap mmap cache has no fields')
        self.arrays: dict[str, np.memmap] = {}
        for (key, description) in field_metadata.items():
            path = cache_root / str(description['file'])
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            expected_shape = tuple((int(value) for value in description['shape']))
            if array.shape != expected_shape:
                raise ValueError(f'{key}: mmap-cache shape mismatch')
            if str(array.dtype) != str(description['dtype']):
                raise ValueError(f'{key}: mmap-cache dtype mismatch')
            if array.shape[0] != len(self.index):
                raise ValueError(f'{key}: mmap-cache field is not frame-aligned')
            self.arrays[str(key)] = array
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        return {key: torch.from_numpy(np.array(array[item], copy=True)) for (key, array) in self.arrays.items()}

def downsample_search_tokens(tokens: np.ndarray, output_side: int=6) -> np.ndarray:
    """Average-pool square token maps without mixing channels."""
    tokens = np.asarray(tokens)
    side = int(round(tokens.shape[-2] ** 0.5))
    if side * side != tokens.shape[-2] or side % output_side:
        raise ValueError('token grid must be square and divisible by output_side')
    factor = side // output_side
    shaped = tokens.reshape(*tokens.shape[:-2], output_side, factor, output_side, factor, tokens.shape[-1])
    pooled = shaped.mean(axis=(-4, -2), dtype=np.float32)
    return pooled.reshape(*tokens.shape[:-2], output_side * output_side, tokens.shape[-1]).astype(tokens.dtype)

class SearchMapCandidateDataset(Dataset[dict[str, torch.Tensor]]):

    def __init__(self, searchmap_root: Path, candidate_root: Path, base_feature_root: Path, train_manifest: Path, evaluation_manifest: Path, history_length: int=16, cache_sequences: int=2, split: str='fit', evaluation_domain: str | None=None, recent_template_root: Path | None=None) -> None:
        fit_names = set(train_sequence_names(train_manifest, evaluation_manifest))
        maps = discover_sequence_files(searchmap_root)
        if split == 'fit':
            unknown = sorted(set(maps) - fit_names)
            if unknown:
                raise ValueError(f'search-map root contains non-fit sequences: {unknown[:10]}')
            names = set(maps)
        elif split == 'fit_holdout':
            if evaluation_domain is None:
                raise ValueError('evaluation_domain is required for fit_holdout')
            unknown = sorted(set(maps) - fit_names)
            if unknown:
                raise ValueError(f'fit-holdout root contains non-fit sequences: {unknown[:10]}')
            manifest = json.loads(train_manifest.read_text(encoding='utf-8'))
            sequence_domains = manifest['sequence_domains']
            names = {name for name in maps if sequence_domains.get(name) == evaluation_domain}
            if not names:
                raise FileNotFoundError(f'no {evaluation_domain} fit-holdout search maps')
        elif split == 'development':
            if evaluation_domain is None:
                raise ValueError('evaluation_domain is required for development')
            manifest = json.loads(evaluation_manifest.read_text(encoding='utf-8'))
            names = set(manifest['evaluation_domains'][evaluation_domain])
            if names & fit_names:
                raise ValueError('development/fit overlap')
        else:
            raise ValueError('split must be fit, fit_holdout, or development')
        candidates = discover_sequence_files(candidate_root)
        base = discover_sequence_files(base_feature_root)
        missing = sorted(names - set(maps) | names - set(candidates) | names - set(base))
        if missing:
            raise FileNotFoundError(f'missing SearchMap inputs: {missing[:10]}')
        self.names = sorted(names)
        self.maps = maps
        self.candidates = candidates
        self.base = base
        self.recent_template_paths = None
        if recent_template_root is not None:
            recent_paths = discover_sequence_files(recent_template_root)
            missing_recent = sorted(names - set(recent_paths))
            if missing_recent:
                raise FileNotFoundError(f'missing recent-template features: {missing_recent[:10]}')
            self.recent_template_paths = recent_paths
        self.history_length = history_length
        self.return_metadata = split != 'fit'
        self.sequence_indices = {name: index for (index, name) in enumerate(self.names)}
        self.index: list[tuple[str, int]] = []
        for name in self.names:
            with np.load(maps[name], allow_pickle=False) as source:
                if str(np.asarray(source['representation_mode']).item()) != 'canonical_searchmap24':
                    raise ValueError(f'{name}: wrong search-map representation')
                if np.asarray(source['search_tokens_full']).shape[1] != 24 * 24:
                    raise ValueError(f'{name}: expected 24x24 tokens')
                length = len(source['frame'])
            self.index.extend(((name, local) for local in range(length)))
        self.cache_sequences = cache_sequences
        self._cache: OrderedDict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray] | None]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, name: str):
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        with np.load(self.maps[name], allow_pickle=False) as source:
            searchmap = {key: np.asarray(source[key]) for key in source.files}
        with np.load(self.candidates[name], allow_pickle=False) as source:
            candidate = {key: np.asarray(source[key]) for key in ('frame', 'candidate_bbox', 'candidate_iou')}
        fields = ('frame', 'template_tokens', 'search_tokens', 'coord_logits', 'tracker_score', 'pred_bbox')
        with np.load(self.base[name], allow_pickle=False) as source:
            base = {key: np.asarray(source[key]) for key in fields}
        if not np.array_equal(searchmap['frame'], candidate['frame']):
            raise ValueError(f'{name}: search-map/candidate frame mismatch')
        recent = None
        if self.recent_template_paths is not None:
            required_recent = ('frame', 'recent_template_tokens', 'recent_template_age', 'recent_template_valid', 'recent_template_similarity')
            with np.load(self.recent_template_paths[name], allow_pickle=False) as source:
                absent = sorted(set(required_recent) - set(source.files))
                if absent:
                    raise ValueError(f'{self.recent_template_paths[name]} missing {absent}')
                recent = {key: np.asarray(source[key]) for key in required_recent}
            missing_frame = sorted(set(searchmap['frame'].tolist()) - set(recent['frame'].tolist()))
            if missing_frame:
                raise ValueError(f'{name}: recent-template frames missing {missing_frame[:10]}')
        self._cache[name] = (searchmap, candidate, base, recent)
        while len(self._cache) > self.cache_sequences:
            self._cache.popitem(last=False)
        return self._cache[name]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        (name, local) = self.index[item]
        (searchmap, candidate, base, recent) = self._load(name)
        frame = int(searchmap['frame'][local])
        matches = np.flatnonzero(base['frame'] == frame)
        if len(matches) != 1:
            raise ValueError(f'{name}: frame {frame} is not unique')
        current = int(matches[0])
        raw_indices = np.arange(current - self.history_length + 1, current + 1)
        history_mask = raw_indices >= 0
        indices = np.clip(raw_indices, 0, current)
        template = base['template_tokens']
        template = np.broadcast_to(template[None], (self.history_length, *template.shape)) if template.ndim == 2 else template[indices]
        boxes = candidate['candidate_bbox'][local]
        raw_box = base['pred_bbox'][current]
        crop = searchmap['search_crop_xywh_image'][local]
        search_boxes = np.stack([xywh_image_to_search_normalized(box, crop) for box in boxes], axis=0).astype(np.float32)
        result = {'template_tokens': torch.from_numpy(np.array(template, copy=True)), 'search_tokens': torch.from_numpy(np.array(base['search_tokens'][indices], copy=True)), 'coord_logits': torch.from_numpy(np.array(base['coord_logits'][indices], copy=True)), 'tracker_score': torch.from_numpy(np.array(base['tracker_score'][indices], copy=True)), 'pred_bbox': torch.from_numpy(np.array(base['pred_bbox'][indices], copy=True)), 'history_mask': torch.from_numpy(history_mask), 'search_tokens_full': torch.from_numpy(np.array(searchmap['search_tokens_full'][local], copy=True)), 'search_valid_fraction_full': torch.from_numpy(np.array(searchmap['search_valid_fraction_full'][local], copy=True)), 'candidate_search_xywh': torch.from_numpy(search_boxes), 'candidate_geometry': torch.from_numpy(candidate_relative_geometry(boxes, raw_box)), 'candidate_iou': torch.from_numpy(np.array(candidate['candidate_iou'][local], copy=True))}
        if recent is not None:
            recent_matches = np.flatnonzero(recent['frame'] == frame)
            if len(recent_matches) != 1:
                raise ValueError(f'{name}: frame {frame} not unique in recent features')
            recent_local = int(recent_matches[0])
            result.update(recent_template_tokens=torch.from_numpy(np.array(recent['recent_template_tokens'][recent_local], copy=True)), recent_template_age=torch.tensor(float(recent['recent_template_age'][recent_local]), dtype=torch.float32), recent_template_valid=torch.tensor(bool(recent['recent_template_valid'][recent_local]), dtype=torch.bool), recent_template_similarity=torch.tensor(float(recent['recent_template_similarity'][recent_local]), dtype=torch.float32))
        if self.return_metadata:
            result.update(sequence_index=torch.tensor(self.sequence_indices[name], dtype=torch.int64), frame=torch.tensor(frame, dtype=torch.int64), candidate_bbox=torch.from_numpy(np.array(boxes, copy=True)))
        return result

class SearchMapCandidateIoU(CandidateContextIoU):
    """Candidate quality from interior, expanded context, and boundary evidence."""

    def __init__(self, base_model: nn.Module, heads: int=4, context_factor: float=1.75, template_mode: str='static', recent_similarity_threshold: float=-1.0, recent_max_age: int=128) -> None:
        super().__init__(base_model, heads=heads, template_mode=template_mode, recent_similarity_threshold=recent_similarity_threshold, recent_max_age=recent_max_age)
        if context_factor <= 1.0:
            raise ValueError('context_factor must exceed one')
        self.context_factor = context_factor
        d_model = int(base_model.iou_head.in_features)
        self.region_embedding = nn.Parameter(torch.empty(3, d_model))
        nn.init.trunc_normal_(self.region_embedding, std=0.02)

    @staticmethod
    def _sample(feature_map: torch.Tensor, validity_map: torch.Tensor, boxes: torch.Tensor, output_side: int) -> tuple[torch.Tensor, torch.Tensor]:
        (batch, candidates) = boxes.shape[:2]
        fractions = (torch.arange(output_side, device=boxes.device, dtype=torch.float32) + 0.5) / output_side
        x = boxes[..., 0, None] + boxes[..., 2, None] * fractions
        y = boxes[..., 1, None] + boxes[..., 3, None] * fractions
        grid = 2.0 * torch.stack([x[:, :, None, :].expand(-1, -1, output_side, -1), y[:, :, :, None].expand(-1, -1, -1, output_side)], dim=-1) - 1.0
        expanded_features = feature_map[:, None].expand(-1, candidates, -1, -1, -1)
        sampled = F.grid_sample(expanded_features.reshape(batch * candidates, feature_map.shape[1], feature_map.shape[2], feature_map.shape[3]), grid.reshape(batch * candidates, output_side, output_side, 2), mode='bilinear', padding_mode='zeros', align_corners=False).flatten(2).transpose(1, 2).reshape(batch, candidates, output_side * output_side, -1)
        expanded_validity = validity_map[:, None, None].expand(-1, candidates, -1, -1, -1)
        validity = F.grid_sample(expanded_validity.reshape(batch * candidates, 1, validity_map.shape[1], validity_map.shape[2]), grid.reshape(batch * candidates, output_side, output_side, 2), mode='bilinear', padding_mode='zeros', align_corners=False).reshape(batch, candidates, output_side * output_side)
        return (sampled * validity.unsqueeze(-1), validity)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        boxes = batch['candidate_search_xywh'].float()
        full = batch['search_tokens_full'].float()
        side = int(round(full.shape[1] ** 0.5))
        if side * side != full.shape[1]:
            raise ValueError('search_tokens_full must be a square map')
        feature_map = full.transpose(1, 2).reshape(full.shape[0], full.shape[2], side, side)
        validity_map = batch['search_valid_fraction_full'].float()
        (interior, interior_valid) = self._sample(feature_map, validity_map, boxes, 4)
        centre = boxes[..., :2] + 0.5 * boxes[..., 2:]
        context_size = boxes[..., 2:] * self.context_factor
        context_boxes = torch.cat([centre - 0.5 * context_size, context_size], dim=-1)
        (context, context_valid) = self._sample(feature_map, validity_map, context_boxes, 6)
        axis = (torch.arange(6, device=boxes.device, dtype=torch.float32) + 0.5) / 6
        inner_half = 0.5 / self.context_factor
        ring_mask = ((axis - 0.5).abs()[:, None] > inner_half) | ((axis - 0.5).abs()[None, :] > inner_half)
        ring = context * ring_mask.reshape(1, 1, 36, 1)
        tokens = torch.cat([interior, context, ring], dim=2)
        (batch_size, count, token_count, _) = tokens.shape
        flat = tokens.reshape(batch_size * count, token_count, -1)
        encoded = self.context_projection(flat)
        region_ids = torch.cat([torch.zeros(interior.shape[2], dtype=torch.long, device=boxes.device), torch.ones(context.shape[2], dtype=torch.long, device=boxes.device), torch.full((ring.shape[2],), 2, dtype=torch.long, device=boxes.device)])
        encoded = encoded + self.region_embedding[region_ids][None]
        (update, _) = self.context_self_attention(encoded, encoded, encoded, need_weights=False)
        encoded = self.context_self_norm(encoded + update)
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
            template_mask = torch.cat([torch.zeros(batch_size, template.shape[1], dtype=torch.bool, device=template.device), (~accepted)[:, None].expand(-1, recent.shape[1])], dim=1)
            template = torch.cat([template, recent], dim=1)
        template = template[:, None].expand(-1, count, -1, -1).reshape(batch_size * count, template.shape[1], -1)
        if template_mask is not None:
            template_mask = template_mask[:, None].expand(-1, count, -1).reshape(batch_size * count, template_mask.shape[1])
        (update, _) = self.template_attention(encoded, template, template, key_padding_mask=template_mask, need_weights=False)
        encoded = self.template_norm(encoded + update)
        query = self.query.expand(batch_size * count, -1, -1)
        (update, _) = self.context_attention(query, encoded, encoded, need_weights=False)
        current = self.context_norm(query + update)[:, 0]
        valid = torch.cat([interior_valid, context_valid], dim=-1).mean(dim=-1, keepdim=True)
        geometry = self.geometry_encoder(torch.cat([batch['candidate_geometry'].float(), valid], dim=-1).reshape(batch_size * count, 5))
        current = current + geometry
        with torch.no_grad():
            previous = self._encode_strict_past(batch)
        previous_mask = batch['history_mask'][:, :-1].bool()
        memory = previous[:, None].expand(-1, count, -1, -1).reshape(batch_size * count, previous.shape[1], -1)
        memory_mask = previous_mask[:, None].expand(-1, count, -1).reshape(batch_size * count, previous_mask.shape[1])
        empty = ~memory_mask.any(dim=1)
        if torch.any(empty):
            memory = memory.clone()
            memory_mask = memory_mask.clone()
            memory[empty, 0] = 0.0
            memory_mask[empty, 0] = True
        (history, _) = self.history_attention(current[:, None], memory, memory, key_padding_mask=~memory_mask, need_weights=False)
        history = self.history_norm(current + history[:, 0])
        fused = self.fusion(torch.cat([current, history, geometry], dim=-1))
        logits = self.iou_head(fused).reshape(batch_size, count, -1)
        probability = torch.softmax(logits, dim=-1)
        return {'iou_logits': logits, 'iou_probability': probability, 'predicted_iou': (probability * self.iou_bin_centres).sum(dim=-1), 'candidate_valid_fraction': valid.reshape(batch_size, count), 'candidate_embedding': fused.reshape(batch_size, count, -1)}

class PrPoolSearchMapCandidateIoU(SearchMapCandidateIoU):
    """SearchMap candidate predictor with exact differentiable Precise RoI Pooling.

    Candidate boxes stay in the canonical search-crop coordinate system.  The
    extension integrates the bilinearly interpolated shared feature map over
    every output bin and exposes gradients with respect to all four RoI edges.
    Coordinates outside the search map retain the extension's zero-padding
    semantics; the separately pooled validity map records how much evidence is
    actually inside the crop.
    """

    @staticmethod
    def _sample(feature_map: torch.Tensor, validity_map: torch.Tensor, boxes: torch.Tensor, output_side: int) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_map.device.type != 'cuda':
            raise RuntimeError('Precise RoI Pooling requires CUDA')
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError('boxes must have shape [batch, candidates, 4]')
        (batch, candidates) = boxes.shape[:2]
        (height, width) = feature_map.shape[-2:]
        xy1 = boxes[..., :2]
        xy2 = xy1 + boxes[..., 2:]
        scale = boxes.new_tensor([width, height])
        xyxy = torch.cat([xy1 * scale, xy2 * scale], dim=-1)
        batch_index = torch.arange(batch, device=boxes.device, dtype=boxes.dtype)[:, None].expand(-1, candidates)
        rois = torch.cat([batch_index[..., None], xyxy], dim=-1).reshape(-1, 5)
        pool = PrRoIPool2D(output_side, output_side, spatial_scale=1.0)
        sampled = pool(feature_map.float(), rois.float())
        sampled = sampled.flatten(2).transpose(1, 2).reshape(batch, candidates, output_side * output_side, feature_map.shape[1])
        validity = pool(validity_map[:, None].float(), rois.float())
        validity = validity.reshape(batch, candidates, output_side * output_side)
        return (sampled * validity.unsqueeze(-1), validity)

class SearchMap48CandidateDataset(SearchMapCandidateDataset):
    """SearchMap-24 inputs plus one shared RGB crop per current frame."""

    def __init__(self, search_rgb_root: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        rgb = discover_sequence_files(search_rgb_root)
        missing = sorted(set(self.names) - set(rgb))
        if missing:
            raise FileNotFoundError(f'missing SearchMap-48 RGB inputs: {missing[:10]}')
        self.rgb = rgb
        self._rgb_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def _load_rgb(self, name: str) -> dict[str, np.ndarray]:
        if name in self._rgb_cache:
            self._rgb_cache.move_to_end(name)
            return self._rgb_cache[name]
        with np.load(self.rgb[name], allow_pickle=False) as source:
            data = {key: np.asarray(source[key]) for key in source.files}
        if str(np.asarray(data['representation_mode']).item()) != 'canonical_search_rgb192_for_shared_48':
            raise ValueError(f'{name}: wrong SearchMap-48 RGB representation')
        (searchmap, _, _) = self._load(name)
        if not np.array_equal(data['frame'], searchmap['frame']):
            raise ValueError(f'{name}: RGB/search-map frame mismatch')
        if data['search_rgb'].shape[1:] != (3, 192, 192):
            raise ValueError(f'{name}: expected [F,3,192,192] RGB crops')
        if data['search_valid_fraction_48'].shape[1:] != (48, 48):
            raise ValueError(f'{name}: expected [F,48,48] validity')
        self._rgb_cache[name] = data
        while len(self._rgb_cache) > self.cache_sequences:
            self._rgb_cache.popitem(last=False)
        return data

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(item)
        (name, local) = self.index[item]
        rgb = self._load_rgb(name)
        result['search_rgb'] = torch.from_numpy(np.array(rgb['search_rgb'][local], copy=True))
        result['search_valid_fraction_48'] = torch.from_numpy(np.array(rgb['search_valid_fraction_48'][local], copy=True))
        return result

class SearchMap48LightRefineIoU(SearchMapCandidateIoU):
    """Fuse shared 24x24 tracker semantics with a shared 48x48 shallow RGB map."""

    def __init__(self, base_model: nn.Module, heads: int=4, context_factor: float=1.75, highres_channels: int=64) -> None:
        super().__init__(base_model, heads=heads, context_factor=context_factor)
        if highres_channels <= 0 or highres_channels % 8:
            raise ValueError('highres_channels must be a positive multiple of eight')
        tracker_channels = next((int(module.in_features) for module in self.context_projection.modules() if isinstance(module, nn.Linear)))
        self.rgb_stem = nn.Sequential(nn.Conv2d(3, highres_channels // 2, 3, stride=2, padding=1), nn.GroupNorm(8, highres_channels // 2), nn.GELU(), nn.Conv2d(highres_channels // 2, highres_channels, 3, stride=2, padding=1), nn.GroupNorm(8, highres_channels), nn.GELU(), nn.Conv2d(highres_channels, highres_channels, 3, padding=1, groups=highres_channels), nn.Conv2d(highres_channels, highres_channels, 1), nn.GroupNorm(8, highres_channels), nn.GELU())
        self.semantic_reduce = nn.Sequential(nn.Conv2d(tracker_channels, highres_channels, 1), nn.GroupNorm(8, highres_channels), nn.GELU())
        self.highres_fusion = nn.Sequential(nn.Conv2d(2 * highres_channels, tracker_channels, 1), nn.GroupNorm(32, tracker_channels), nn.GELU())
        self.register_buffer('rgb_mean', torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rgb = batch['search_rgb'].float().div(255.0)
        rgb = (rgb - self.rgb_mean) / self.rgb_std
        highres = self.rgb_stem(rgb)
        if highres.shape[-2:] != (48, 48):
            raise ValueError('shared RGB stem must produce a 48x48 feature map')
        full = batch['search_tokens_full'].float()
        side = int(round(full.shape[1] ** 0.5))
        if side * side != full.shape[1]:
            raise ValueError('search_tokens_full must be a square map')
        semantic = full.transpose(1, 2).reshape(full.shape[0], full.shape[2], side, side)
        semantic = self.semantic_reduce(semantic)
        semantic = F.interpolate(semantic, size=highres.shape[-2:], mode='bilinear', align_corners=False)
        fused = self.highres_fusion(torch.cat([semantic, highres], dim=1))
        enriched = dict(batch)
        enriched['search_tokens_full'] = fused.flatten(2).transpose(1, 2)
        enriched['search_valid_fraction_full'] = batch['search_valid_fraction_48'].float()
        return super().forward(enriched)

class SearchMap48DirectGainIoU(SearchMap48LightRefineIoU):
    """Frozen absolute-IoU path plus a differentiable candidate-vs-raw gain head."""

    def __init__(self, base_model: nn.Module, heads: int=4, context_factor: float=1.75, highres_channels: int=64, gain_bins: int=50) -> None:
        super().__init__(base_model, heads=heads, context_factor=context_factor, highres_channels=highres_channels)
        if gain_bins < 2:
            raise ValueError('gain_bins must be at least two')
        d_model = int(base_model.iou_head.in_features)
        self.gain_head = nn.Sequential(nn.LayerNorm(4 * d_model), nn.Linear(4 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * d_model, gain_bins))
        centres_01 = (torch.arange(gain_bins, dtype=torch.float32) + 0.5) / gain_bins
        self.register_buffer('gain_bin_centres', 2.0 * centres_01 - 1.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = super().forward(batch)
        embedding = output['candidate_embedding']
        raw = embedding[:, :1].expand_as(embedding)
        pair = torch.cat([embedding, raw, embedding - raw, embedding * raw], dim=-1)
        gain_logits = self.gain_head(pair)
        gain_probability = torch.softmax(gain_logits, dim=-1)
        gain = (gain_probability * self.gain_bin_centres).sum(dim=-1)
        gain = torch.cat([torch.zeros_like(gain[:, :1]), gain[:, 1:]], dim=1)
        output.update(gain_logits=gain_logits, gain_probability=gain_probability, predicted_gain=gain)
        return output

class SearchMap48SetGainIoU(SearchMap48DirectGainIoU):
    """Jointly contextualize the raw/no-op box and nearby gain candidates."""

    def __init__(self, base_model: nn.Module, heads: int=4, context_factor: float=1.75, highres_channels: int=64, gain_bins: int=50, set_layers: int=1) -> None:
        super().__init__(base_model, heads=heads, context_factor=context_factor, highres_channels=highres_channels, gain_bins=gain_bins)
        if set_layers != 1:
            raise ValueError('SetGain-v1 fixes set_layers=1')
        d_model = int(base_model.iou_head.in_features)
        self.set_attention = nn.MultiheadAttention(d_model, heads, dropout=0.1, batch_first=True)
        self.set_attention_norm = nn.LayerNorm(d_model)
        self.set_feed_forward = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * d_model, d_model))
        self.set_feed_forward_norm = nn.LayerNorm(d_model)
        self.set_residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = SearchMap48LightRefineIoU.forward(self, batch)
        embedding = output['candidate_embedding']
        (update, _) = self.set_attention(embedding, embedding, embedding, need_weights=False)
        contextual = self.set_attention_norm(embedding + self.set_residual_scale * update)
        update = self.set_feed_forward(contextual)
        contextual = self.set_feed_forward_norm(contextual + self.set_residual_scale * update)
        raw = contextual[:, :1].expand_as(contextual)
        pair = torch.cat([contextual, raw, contextual - raw, contextual * raw], dim=-1)
        gain_logits = self.gain_head(pair)
        gain_probability = torch.softmax(gain_logits, dim=-1)
        gain = (gain_probability * self.gain_bin_centres).sum(dim=-1)
        gain = torch.cat([torch.zeros_like(gain[:, :1]), gain[:, 1:]], dim=1)
        output.update(gain_logits=gain_logits, gain_probability=gain_probability, predicted_gain=gain, contextual_candidate_embedding=contextual)
        return output
