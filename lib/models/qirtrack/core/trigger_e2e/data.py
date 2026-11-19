"""Strict train-only data contract for Trigger-E2E.

Each sequence is stored in one ``.npz`` file.  A sample contains a causal
window ending at the requested frame.  Sequence starts are left padded with
their first frame, and ``history_mask`` tells the model which positions are
real observations rather than padding.

Version-1 sequence schema
-------------------------
``frame``
    ``[T]`` int64 frame identifiers.
``template_tokens``
    ``[Nz, C]`` for one immutable initial template, or ``[T, Nz, C]`` for a
    causal per-frame template policy.
``search_tokens``
    ``[T, Nx, C]`` frozen tracker search representations.
``coord_logits``
    ``[T, 4, V]`` raw coordinate logits, before softmax.
``tracker_score``
    ``[T, S]`` optional tracker-native score features.  ``S`` may be zero.
``pred_bbox``
    ``[T, 4]`` tracker prediction in xywh format.  It is an online input.
``iou``
    ``[T]`` offline supervision only.

The dataset returns IoU because the training loss needs it.  The future online
adapter will expose the same dictionary without ``iou``.
"""
from __future__ import annotations
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import numpy as np
import torch
from torch.utils.data import Dataset
from lib.models.qirtrack.core.trigger_e2e.distribution import soft_iou_targets
SCHEMA_VERSION = 1
REQUIRED_FIELDS = ('frame', 'template_tokens', 'search_tokens', 'coord_logits', 'tracker_score', 'pred_bbox', 'iou')

@dataclass(frozen=True)
class SequenceInfo:
    """Metadata needed to build a global frame index without loading tokens."""
    name: str
    path: Path
    length: int
    template_is_static: bool

def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'manifest must contain a JSON object: {path}')
    return value

def _duplicates(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)

def evaluation_sequence_names(manifest_path: Path) -> set[str]:
    """Collect sequence names from an evaluation-only manifest.

    Existing repository manifests use ``evaluation_domains``.  Supporting a
    ``splits`` mapping as well keeps the audit usable for small local fixtures.
    """
    manifest = _read_json(manifest_path)
    if 'evaluation_domains' in manifest:
        groups = manifest['evaluation_domains']
    elif 'splits' in manifest:
        groups = manifest['splits']
    else:
        raise ValueError(f'evaluation manifest has neither evaluation_domains nor splits: {manifest_path}')
    if not isinstance(groups, Mapping):
        raise ValueError(f'evaluation groups must be a mapping: {manifest_path}')
    result: list[str] = []
    for (group, names) in groups.items():
        if not isinstance(names, list) or not all((isinstance(x, str) for x in names)):
            raise ValueError(f'evaluation group {group!r} is not a string list')
        result.extend(names)
    duplicate = _duplicates(result)
    if duplicate:
        raise ValueError(f'duplicate evaluation sequences: {duplicate[:5]}')
    return set(result)

def evaluation_domain_sequence_names(manifest_path: Path, domain: str) -> list[str]:
    """Load one named evaluation domain without flattening domain identity."""
    manifest = _read_json(manifest_path)
    groups = manifest.get('evaluation_domains')
    if not isinstance(groups, Mapping):
        raise ValueError(f'evaluation_domains must be a mapping: {manifest_path}')
    if domain not in groups:
        raise KeyError(f'unknown evaluation domain {domain!r}; available={sorted(groups)}')
    names = groups[domain]
    if not isinstance(names, list) or not names or (not all((isinstance(name, str) and name for name in names))):
        raise ValueError(f'evaluation domain {domain!r} is not a non-empty string list')
    duplicate = _duplicates(names)
    if duplicate:
        raise ValueError(f'duplicate {domain} evaluation sequences: {duplicate[:5]}')
    return names

def train_sequence_names(manifest_path: Path, evaluation_manifest_path: Path | None=None) -> list[str]:
    """Load and audit a strict train-only sequence list."""
    manifest = _read_json(manifest_path)
    splits = manifest.get('splits')
    if not isinstance(splits, Mapping) or set(splits) != {'train'}:
        raise ValueError(f"Trigger-E2E training accepts exactly one split named 'train'; found {(sorted(splits) if isinstance(splits, Mapping) else splits)!r}")
    if manifest.get('strict_train_only') is not True:
        raise ValueError('training manifest must set strict_train_only=true')
    for flag in ('validation_used_for_training', 'test_used_for_training', 'public_test_used', 'final_test_included'):
        if manifest.get(flag, False):
            raise ValueError(f'unsafe training manifest flag: {flag}=true')
    names = splits['train']
    if not isinstance(names, list) or not names or (not all((isinstance(name, str) and name for name in names))):
        raise ValueError('splits.train must be a non-empty list of sequence names')
    duplicate = _duplicates(names)
    if duplicate:
        raise ValueError(f'duplicate training sequences: {duplicate[:5]}')
    if evaluation_manifest_path is not None:
        overlap = sorted(set(names) & evaluation_sequence_names(evaluation_manifest_path))
        if overlap:
            raise ValueError(f'training/evaluation sequence overlap is forbidden; first overlaps: {overlap[:10]}')
    return names

def discover_sequence_files(feature_root: Path) -> dict[str, Path]:
    """Map sequence stem to file, rejecting ambiguous duplicate files."""
    result: dict[str, Path] = {}
    for path in sorted(feature_root.glob('**/*.npz')):
        name = path.stem
        if name in result:
            raise ValueError(f'duplicate feature files for {name}: {result[name]} and {path}')
        result[name] = path
    return result

def _validate_shapes(path: Path, arrays: Mapping[str, np.ndarray]) -> SequenceInfo:
    missing = [field for field in REQUIRED_FIELDS if field not in arrays]
    if missing:
        raise ValueError(f'{path}: missing fields {missing}')
    frame = arrays['frame']
    iou = arrays['iou']
    search = arrays['search_tokens']
    template = arrays['template_tokens']
    coord = arrays['coord_logits']
    score = arrays['tracker_score']
    bbox = arrays['pred_bbox']
    if frame.ndim != 1 or len(frame) == 0:
        raise ValueError(f'{path}: frame must have shape [T] with T>0')
    length = len(frame)
    expected = {'iou': (length,), 'pred_bbox': (length, 4)}
    for (field, shape) in expected.items():
        if arrays[field].shape != shape:
            raise ValueError(f'{path}: {field} shape {arrays[field].shape} != {shape}')
    if search.ndim != 3 or search.shape[0] != length:
        raise ValueError(f'{path}: search_tokens must have shape [T, Nx, C]')
    if template.ndim == 2:
        template_is_static = True
        template_dim = template.shape[-1]
    elif template.ndim == 3 and template.shape[0] == length:
        template_is_static = False
        template_dim = template.shape[-1]
    else:
        raise ValueError(f'{path}: template_tokens must have shape [Nz,C] or [T,Nz,C]')
    if template_dim != search.shape[-1]:
        raise ValueError(f'{path}: template/search channel dimensions differ')
    if coord.ndim != 3 or coord.shape[:2] != (length, 4):
        raise ValueError(f'{path}: coord_logits must have shape [T,4,V]')
    if score.ndim != 2 or score.shape[0] != length:
        raise ValueError(f'{path}: tracker_score must have shape [T,S]')
    if not np.isfinite(iou).all() or np.any((iou < 0.0) | (iou > 1.0)):
        raise ValueError(f'{path}: iou must be finite and inside [0,1]')
    if not np.all(frame[1:] > frame[:-1]):
        raise ValueError(f'{path}: frame identifiers must be strictly increasing')
    return SequenceInfo(path.stem, path, length, template_is_static)

def inspect_sequence_file(path: Path) -> SequenceInfo:
    """Validate one sequence file without retaining its large arrays."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
        info = _validate_shapes(path, arrays)
        schema = int(np.asarray(data.get('schema_version', SCHEMA_VERSION)).item())
        if schema != SCHEMA_VERSION:
            raise ValueError(f'{path}: schema_version={schema}, expected {SCHEMA_VERSION}')
    return info

class TriggerE2ESequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy, sequence-safe K-frame dataset for token-level quality learning."""

    def __init__(self, feature_root: Path, train_manifest_path: Path, evaluation_manifest_path: Path, history_length: int=16, cache_sequences: int=2) -> None:
        if history_length < 1:
            raise ValueError('history_length must be positive')
        if cache_sequences < 1:
            raise ValueError('cache_sequences must be positive')
        self.history_length = history_length
        self.cache_sequences = cache_sequences
        names = train_sequence_names(train_manifest_path, evaluation_manifest_path)
        paths = discover_sequence_files(feature_root)
        missing = [name for name in names if name not in paths]
        if missing:
            raise FileNotFoundError(f'missing feature files: {missing[:10]}')
        self.sequences = [inspect_sequence_file(paths[name]) for name in names]
        self.index = [(sequence_index, frame_index) for (sequence_index, sequence) in enumerate(self.sequences) for frame_index in range(sequence.length)]
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        with np.load(path, allow_pickle=False) as data:
            arrays = {field: np.asarray(data[field]) for field in REQUIRED_FIELDS}
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
        return {'template_tokens': torch.from_numpy(np.array(template_window, copy=True)), 'search_tokens': torch.from_numpy(np.array(arrays['search_tokens'][indices], copy=True)), 'coord_logits': torch.from_numpy(np.array(arrays['coord_logits'][indices], copy=True)), 'tracker_score': torch.from_numpy(np.array(arrays['tracker_score'][indices], copy=True)), 'pred_bbox': torch.from_numpy(np.array(arrays['pred_bbox'][indices], copy=True)), 'history_mask': torch.from_numpy(history_mask), 'history_iou': torch.from_numpy(np.array(arrays['iou'][indices], dtype=np.float32, copy=True)), 'iou': torch.tensor(float(arrays['iou'][current]), dtype=torch.float32), 'frame': torch.tensor(int(arrays['frame'][current]), dtype=torch.int64), 'sequence_index': torch.tensor(sequence_index, dtype=torch.int64)}

class TriggerE2EEvaluationDataset(TriggerE2ESequenceDataset):
    """Evaluation-only view that cannot accept a training manifest."""

    def __init__(self, feature_root: Path, evaluation_manifest_path: Path, domain: str, history_length: int=16, cache_sequences: int=2) -> None:
        if history_length < 1:
            raise ValueError('history_length must be positive')
        if cache_sequences < 1:
            raise ValueError('cache_sequences must be positive')
        self.history_length = history_length
        self.cache_sequences = cache_sequences
        names = evaluation_domain_sequence_names(evaluation_manifest_path, domain)
        paths = discover_sequence_files(feature_root)
        missing = [name for name in names if name not in paths]
        if missing:
            raise FileNotFoundError(f'missing evaluation feature files: {missing[:10]}')
        self.sequences = [inspect_sequence_file(paths[name]) for name in names]
        self.index = [(sequence_index, frame_index) for (sequence_index, sequence) in enumerate(self.sequences) for frame_index in range(sequence.length)]
        self._cache = OrderedDict()
