"""Exact search-crop provenance for candidate-conditioned IoU prediction.

The formulas intentionally mirror ``sample_target``.  The module contains no
ground-truth access and is safe to call online.  GT remains an offline label.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np

def compute_search_crop_metadata(image_shape: tuple[int, ...], target_bbox_xywh: np.ndarray | list[float], search_area_factor: float, output_size: int, full_grid_side: int, pooled_grid_side: int) -> dict[str, np.ndarray]:
    """Reproduce the integer crop and padding geometry used by sample_target."""
    (image_height, image_width) = (int(image_shape[0]), int(image_shape[1]))
    (x, y, width, height) = np.asarray(target_bbox_xywh, dtype=np.float64).tolist()
    crop_size = int(math.ceil(math.sqrt(width * height) * search_area_factor))
    if crop_size < 1:
        raise ValueError('search crop is smaller than one pixel')
    x1 = int(round(x + 0.5 * width - 0.5 * crop_size))
    y1 = int(round(y + 0.5 * height - 0.5 * crop_size))
    (x2, y2) = (x1 + crop_size, y1 + crop_size)
    left = max(0, -x1)
    right = max(x2 - image_width + 1, 0)
    top = max(0, -y1)
    bottom = max(y2 - image_height + 1, 0)
    return {'search_previous_bbox_xywh': np.asarray([x, y, width, height], dtype=np.float32), 'search_crop_xywh_image': np.asarray([x1, y1, crop_size, crop_size], dtype=np.float32), 'search_padding_ltrb': np.asarray([left, top, right, bottom], dtype=np.int32), 'search_image_shape_hw': np.asarray([image_height, image_width], dtype=np.int32), 'search_resize_factor': np.asarray(output_size / crop_size, dtype=np.float32), 'search_output_size': np.asarray(output_size, dtype=np.int32), 'search_full_grid_side': np.asarray(full_grid_side, dtype=np.int32), 'search_pooled_grid_side': np.asarray(pooled_grid_side, dtype=np.int32)}

def pooled_valid_fraction(invalid_mask: np.ndarray, output_side: int) -> np.ndarray:
    """Return the real-image fraction of each uniformly pooled search cell."""
    mask = np.asarray(invalid_mask, dtype=np.float32)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f'expected a square 2-D mask, got {mask.shape}')
    if mask.shape[0] % output_side:
        raise ValueError(f'mask side {mask.shape[0]} is not divisible by output side {output_side}')
    cell = mask.shape[0] // output_side
    invalid_fraction = mask.reshape(output_side, cell, output_side, cell).mean(axis=(1, 3))
    return (1.0 - invalid_fraction).astype(np.float32)

def xywh_image_to_search_normalized(bbox_xywh: np.ndarray | list[float], crop_xywh_image: np.ndarray | list[float]) -> np.ndarray:
    """Map an image-space box to [x,y,w,h] in the unpadded search-crop frame."""
    box = np.asarray(bbox_xywh, dtype=np.float64)
    crop = np.asarray(crop_xywh_image, dtype=np.float64)
    if box.shape != (4,) or crop.shape != (4,):
        raise ValueError('bbox and crop must both have shape [4]')
    if crop[2] <= 0 or crop[3] <= 0:
        raise ValueError('crop width and height must be positive')
    result = box.copy()
    result[:2] = (box[:2] - crop[:2]) / crop[2:]
    result[2:] = box[2:] / crop[2:]
    return result.astype(np.float32)

def validate_resize_factor(metadata: dict[str, Any], observed: float) -> None:
    expected = float(np.asarray(metadata['search_resize_factor']))
    if not math.isclose(expected, float(observed), rel_tol=1e-07, abs_tol=1e-07):
        raise RuntimeError(f'search resize mismatch: reconstructed={expected} observed={observed}')
