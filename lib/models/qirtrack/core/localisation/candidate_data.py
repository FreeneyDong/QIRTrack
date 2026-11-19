"""Strict data contract for candidate-conditioned IoU prediction.

Candidate boxes are deterministic functions of the tracker prediction, frame
identity, and image size.  Ground truth is never consulted by candidate
generation; it is read only afterwards to build offline IoU supervision.
"""
from __future__ import annotations
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from lib.models.qirtrack.core.localisation.spatial_provenance import xywh_image_to_search_normalized
from lib.models.qirtrack.core.trigger_e2e.data import discover_sequence_files, evaluation_domain_sequence_names, train_sequence_names
SPATIAL_SCHEMA_VERSION = 2
C3_MULTISCALE_SPECS = ((0.08, 0.06), (0.25, 0.2), (0.5, 0.4))
C3_CANDIDATE_COUNT = 1 + 4 * len(C3_MULTISCALE_SPECS)
BASE_FIELDS = ('frame', 'template_tokens', 'search_tokens', 'coord_logits', 'tracker_score', 'pred_bbox', 'iou')
SPATIAL_FIELDS = ('gt_bbox', 'search_crop_xywh_image', 'search_image_shape_hw', 'search_valid_fraction', 'search_pooled_grid_side')

@dataclass(frozen=True)
class CandidateSequenceInfo:
    name: str
    path: Path
    length: int
    template_is_static: bool

def _stable_seed(sequence_name: str, frame: int, seed: int) -> int:
    payload = f'{seed}:{sequence_name}:{frame}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')

def clip_xywh(boxes: np.ndarray, image_shape_hw: np.ndarray) -> np.ndarray:
    """Clip xywh boxes to an image while retaining at least one pixel."""
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    (height, width) = np.asarray(image_shape_hw, dtype=np.float64).tolist()
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError('boxes must have shape [N,4]')
    if height < 1 or width < 1:
        raise ValueError('image dimensions must be positive')
    x1 = np.clip(boxes[:, 0], 0.0, max(width - 1.0, 0.0))
    y1 = np.clip(boxes[:, 1], 0.0, max(height - 1.0, 0.0))
    x2 = np.clip(boxes[:, 0] + boxes[:, 2], x1 + 1.0, width)
    y2 = np.clip(boxes[:, 1] + boxes[:, 3], y1 + 1.0, height)
    return np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1).astype(np.float32)

def generate_candidate_boxes(raw_bbox_xywh: np.ndarray, image_shape_hw: np.ndarray, sequence_name: str, frame: int, count: int, seed: int=20260909, translation_extent: float=0.5, log_scale_extent: float=0.4) -> np.ndarray:
    """Generate raw plus scrambled-Sobol perturbations without using GT."""
    if count < 1:
        raise ValueError('candidate count must be positive')
    raw = np.asarray(raw_bbox_xywh, dtype=np.float64)
    if raw.shape != (4,) or not np.isfinite(raw).all() or np.any(raw[2:] <= 0):
        raise ValueError('raw bbox must be a finite positive xywh box')
    boxes = np.broadcast_to(raw, (count, 4)).copy()
    if count > 1:
        engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=_stable_seed(sequence_name, int(frame), seed))
        values = engine.draw(count - 1).cpu().numpy().astype(np.float64)
        offset = (2.0 * values[:, :2] - 1.0) * translation_extent
        log_scale = (2.0 * values[:, 2:] - 1.0) * log_scale_extent
        raw_center = raw[:2] + 0.5 * raw[2:]
        size = raw[2:] * np.exp(log_scale)
        center = raw_center + offset * raw[2:]
        boxes[1:, :2] = center - 0.5 * size
        boxes[1:, 2:] = size
    boxes = clip_xywh(boxes, image_shape_hw)
    boxes[0] = raw.astype(np.float32)
    return boxes

def generate_multiscale_candidate_boxes(raw_bbox_xywh: np.ndarray, image_shape_hw: np.ndarray, sequence_name: str, frame: int, count: int=C3_CANDIDATE_COUNT, seed: int=20260910) -> np.ndarray:
    """Generate the label-blind C3 fine/medium/coarse candidate curriculum.

    Each scale contributes two joint centre/log-scale perturbations and two
    single-edge perturbations.  The layout is fixed and balanced across scales;
    ground truth is deliberately absent from this interface.
    """
    if count != C3_CANDIDATE_COUNT:
        raise ValueError(f'C3 multiscale requires {C3_CANDIDATE_COUNT} candidates, got {count}')
    raw = np.asarray(raw_bbox_xywh, dtype=np.float64)
    if raw.shape != (4,) or not np.isfinite(raw).all() or np.any(raw[2:] <= 0):
        raise ValueError('raw bbox must be a finite positive xywh box')
    candidates = [raw.copy()]
    base_seed = _stable_seed(sequence_name, int(frame), seed)
    raw_center = raw[:2] + 0.5 * raw[2:]
    for (scale_index, (translation_extent, log_scale_extent)) in enumerate(C3_MULTISCALE_SPECS):
        engine = torch.quasirandom.SobolEngine(dimension=4, scramble=True, seed=(base_seed + 104729 * (scale_index + 1)) % (2 ** 31 - 1))
        values = engine.draw(4).cpu().numpy().astype(np.float64)
        offset = (2.0 * values[:2, :2] - 1.0) * translation_extent
        log_scale = (2.0 * values[:2, 2:] - 1.0) * log_scale_extent
        size = raw[2:] * np.exp(log_scale)
        centre = raw_center + offset * raw[2:]
        candidates.extend(np.concatenate([centre - 0.5 * size, size], axis=1))
        for local_index in range(2):
            side = (base_seed + 2 * scale_index + local_index) % 4
            magnitude = translation_extent * (0.5 + 0.5 * values[2 + local_index, 0])
            direction = -1.0 if values[2 + local_index, 1] < 0.5 else 1.0
            delta = direction * magnitude
            (x1, y1) = raw[:2]
            (x2, y2) = raw[:2] + raw[2:]
            if side == 0:
                x1 += delta * raw[2]
            elif side == 1:
                x2 += delta * raw[2]
            elif side == 2:
                y1 += delta * raw[3]
            else:
                y2 += delta * raw[3]
            candidates.append(np.asarray([x1, y1, x2 - x1, y2 - y1]))
    boxes = clip_xywh(np.asarray(candidates), image_shape_hw)
    boxes[0] = raw.astype(np.float32)
    return boxes

def box_iou_xywh(boxes: np.ndarray, target: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or target.shape != (4,):
        raise ValueError('expected boxes [N,4] and target [4]')
    boxes_xy2 = boxes[:, :2] + boxes[:, 2:]
    target_xy2 = target[:2] + target[2:]
    intersection_lt = np.maximum(boxes[:, :2], target[:2])
    intersection_rb = np.minimum(boxes_xy2, target_xy2)
    intersection_wh = np.maximum(intersection_rb - intersection_lt, 0.0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
    union = boxes[:, 2] * boxes[:, 3] + target[2] * target[3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0).astype(np.float32)

def candidate_relative_geometry(candidates: np.ndarray, raw_bbox: np.ndarray) -> np.ndarray:
    """Encode centre displacement and log scale relative to the raw box."""
    candidates = np.asarray(candidates, dtype=np.float64)
    raw = np.asarray(raw_bbox, dtype=np.float64)
    raw_center = raw[:2] + 0.5 * raw[2:]
    centres = candidates[:, :2] + 0.5 * candidates[:, 2:]
    displacement = (centres - raw_center) / np.maximum(raw[2:], 1e-06)
    log_scale = np.log(np.maximum(candidates[:, 2:], 1e-06) / np.maximum(raw[2:], 1e-06))
    return np.concatenate([displacement, log_scale], axis=-1).astype(np.float32)

def inspect_candidate_sequence(path: Path) -> CandidateSequenceInfo:
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in (*BASE_FIELDS, *SPATIAL_FIELDS) if field not in data]
        if missing:
            raise ValueError(f'{path}: missing candidate fields {missing}')
        if int(np.asarray(data['schema_version']).item()) != SPATIAL_SCHEMA_VERSION:
            raise ValueError(f'{path}: candidate features require schema version 2')
        frame = data['frame']
        length = len(frame)
        if frame.ndim != 1 or length < 1 or (not np.all(frame[1:] > frame[:-1])):
            raise ValueError(f'{path}: invalid frame identifiers')
        template = data['template_tokens']
        template_is_static = template.ndim == 2
        if not template_is_static and (template.ndim != 3 or template.shape[0] != length):
            raise ValueError(f'{path}: invalid template token shape')
        expected_first = {'search_tokens': length, 'coord_logits': length, 'tracker_score': length, 'pred_bbox': length, 'gt_bbox': length, 'iou': length, 'search_crop_xywh_image': length, 'search_image_shape_hw': length, 'search_valid_fraction': length}
        for (field, expected) in expected_first.items():
            if data[field].shape[0] != expected:
                raise ValueError(f'{path}: {field} length mismatch')
        side = int(np.asarray(data['search_pooled_grid_side'])[0])
        if data['search_tokens'].shape[1] != side * side:
            raise ValueError(f'{path}: pooled grid does not match search tokens')
        if data['search_valid_fraction'].shape[1:] != (side, side):
            raise ValueError(f'{path}: invalid pooled validity shape')
    return CandidateSequenceInfo(path.stem, path, length, template_is_static)

class CandidateIoUDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy K-frame U1 input plus label-blind current-frame candidates."""

    def __init__(self, feature_root: Path, manifest_path: Path, evaluation_manifest_path: Path, *, history_length: int=16, candidate_count: int=8, seed: int=20260909, evaluation_domain: str | None=None, cache_sequences: int=2) -> None:
        if evaluation_domain is None:
            names = train_sequence_names(manifest_path, evaluation_manifest_path)
        else:
            names = evaluation_domain_sequence_names(evaluation_manifest_path, evaluation_domain)
        paths = discover_sequence_files(feature_root)
        missing = [name for name in names if name not in paths]
        if missing:
            raise FileNotFoundError(f'missing spatial feature files: {missing[:10]}')
        self.sequences = [inspect_candidate_sequence(paths[name]) for name in names]
        self.index = [(sequence_index, frame_index) for (sequence_index, sequence) in enumerate(self.sequences) for frame_index in range(sequence.length)]
        self.history_length = history_length
        self.candidate_count = candidate_count
        self.seed = seed
        self.cache_sequences = cache_sequences
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        with np.load(path, allow_pickle=False) as data:
            arrays = {field: np.asarray(data[field]) for field in (*BASE_FIELDS, *SPATIAL_FIELDS)}
        self._cache[path] = arrays
        while len(self._cache) > self.cache_sequences:
            self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        (sequence_index, current) = self.index[item]
        sequence = self.sequences[sequence_index]
        arrays = self._load(sequence.path)
        raw_indices = np.arange(current - self.history_length + 1, current + 1)
        history_mask = raw_indices >= 0
        indices = np.clip(raw_indices, 0, current)
        template = arrays['template_tokens']
        if sequence.template_is_static:
            template_window = np.broadcast_to(template[None], (self.history_length, *template.shape))
        else:
            template_window = template[indices]
        frame = int(arrays['frame'][current])
        raw_bbox = arrays['pred_bbox'][current]
        image_shape = arrays['search_image_shape_hw'][current]
        candidates = generate_candidate_boxes(raw_bbox, image_shape, sequence.name, frame, self.candidate_count, self.seed)
        crop = arrays['search_crop_xywh_image'][current]
        search_boxes = np.stack([xywh_image_to_search_normalized(box, crop) for box in candidates], axis=0)
        target = arrays['gt_bbox'][current]
        return {'template_tokens': torch.from_numpy(np.array(template_window, copy=True)), 'search_tokens': torch.from_numpy(np.array(arrays['search_tokens'][indices], copy=True)), 'coord_logits': torch.from_numpy(np.array(arrays['coord_logits'][indices], copy=True)), 'tracker_score': torch.from_numpy(np.array(arrays['tracker_score'][indices], copy=True)), 'pred_bbox': torch.from_numpy(np.array(arrays['pred_bbox'][indices], copy=True)), 'history_mask': torch.from_numpy(history_mask), 'candidate_bbox': torch.from_numpy(candidates), 'candidate_search_xywh': torch.from_numpy(search_boxes), 'candidate_geometry': torch.from_numpy(candidate_relative_geometry(candidates, raw_bbox)), 'candidate_iou': torch.from_numpy(box_iou_xywh(candidates, target)), 'search_valid_fraction': torch.from_numpy(np.array(arrays['search_valid_fraction'][current], copy=True)), 'raw_iou': torch.tensor(float(arrays['iou'][current]), dtype=torch.float32), 'frame': torch.tensor(frame, dtype=torch.int64), 'sequence_index': torch.tensor(sequence_index, dtype=torch.int64)}
