"""Label-blind ARTrack coordinate-distribution proposals for ATOM refinement."""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np

def _log_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    shifted = values - maximum
    return shifted - math.log(float(np.exp(shifted).sum()))

def rank_coordinate_proposals(coordinate_logits: np.ndarray, raw_tokens: Sequence[int], *, budget: int=9, top_k: int=4, radius: int=16, beam_width: int=64) -> list[tuple[int, int, int, int]]:
    """Return high-probability valid xyxy token rows, excluding the raw row.

    Released ARTrackV2 uses four command positions whose stored coordinate logits
    do not condition on substituted coordinate prefixes. Accordingly this is a
    marginal coordinate-distribution beam, not prefix-conditioned decoding.
    Ground truth and IoU scores are never used to form or rank proposals.
    """
    logits = np.asarray(coordinate_logits, dtype=np.float64)
    raw = tuple((int(value) for value in raw_tokens))
    if logits.ndim != 2 or logits.shape[0] != 4:
        raise ValueError(f'expected [4,V] coordinate logits, got {logits.shape}')
    if len(raw) != 4:
        raise ValueError('raw_tokens must have four entries')
    vocabulary = logits.shape[1]
    if any((value < 0 or value >= vocabulary for value in raw)):
        raise ValueError('raw coordinate token outside vocabulary')
    if budget < 1 or top_k < 1 or radius < 0 or (beam_width < budget):
        raise ValueError('invalid proposal-search hyperparameters')
    log_probabilities = np.stack([_log_softmax(row) for row in logits])
    allowed: list[list[int]] = []
    for position in range(4):
        global_top = np.argsort(-log_probabilities[position], kind='stable')[:top_k]
        lower = max(0, raw[position] - radius)
        upper = min(vocabulary - 1, raw[position] + radius)
        values = set((int(value) for value in global_top))
        values.update(range(lower, upper + 1))
        values.add(raw[position])
        allowed.append(sorted(values))
    beam: list[tuple[tuple[int, ...], float]] = [(tuple(), 0.0)]
    for position in range(4):
        expanded: list[tuple[tuple[int, ...], float]] = []
        for (prefix, score) in beam:
            for value in allowed[position]:
                candidate = prefix + (value,)
                if position == 2 and candidate[2] <= candidate[0]:
                    continue
                if position == 3 and candidate[3] <= candidate[1]:
                    continue
                expanded.append((candidate, score + float(log_probabilities[position, value])))
        expanded.sort(key=lambda item: (-item[1], item[0]))
        beam = expanded[:beam_width]
        if not beam:
            break
    proposals: list[tuple[int, int, int, int]] = []
    seen = {raw}
    for (tokens, _) in beam:
        if len(tokens) != 4 or tokens in seen:
            continue
        seen.add(tokens)
        proposals.append(tuple((int(value) for value in tokens)))
        if len(proposals) == budget:
            break
    return proposals

def decode_tokens_to_image_boxes(token_rows: Sequence[Sequence[int]], previous_box: Sequence[float], image_shape: Sequence[int], *, bins: int=400, search_size: int=384, search_factor: float=5.0, margin: float=10.0) -> np.ndarray:
    """Decode ARTrack xyxy tokens through the released search-crop geometry."""
    tokens = np.asarray(token_rows, dtype=np.float64)
    if tokens.ndim != 2 or tokens.shape[1] != 4:
        raise ValueError(f'expected [N,4] tokens, got {tokens.shape}')
    previous = np.asarray(previous_box, dtype=np.float64)
    if previous.shape != (4,) or np.any(previous[2:] <= 0):
        raise ValueError('previous_box must be a positive xywh box')
    crop_size = math.ceil(math.sqrt(float(previous[2] * previous[3])) * search_factor)
    resize_factor = float(search_size) / float(crop_size)
    scale = float(search_size) / resize_factor
    xyxy = tokens / float(bins - 1) - 0.5
    centres = 0.5 * (xyxy[:, :2] + xyxy[:, 2:])
    sizes = xyxy[:, 2:] - xyxy[:, :2]
    previous_centre = previous[:2] + 0.5 * previous[2:]
    image_centres = centres * scale + previous_centre - 0.5 * scale
    boxes = np.concatenate((image_centres - 0.5 * sizes * scale, sizes * scale), axis=1)
    (height, width) = (int(image_shape[0]), int(image_shape[1]))
    x1 = np.clip(boxes[:, 0], 0.0, width - margin)
    y1 = np.clip(boxes[:, 1], 0.0, height - margin)
    x2 = np.clip(boxes[:, 0] + boxes[:, 2], margin, width)
    y2 = np.clip(boxes[:, 1] + boxes[:, 3], margin, height)
    result = np.stack((x1, y1, np.maximum(margin, x2 - x1), np.maximum(margin, y2 - y1)), axis=1)
    return result.astype(np.float32)

def build_ar_proposal_boxes(coordinate_logits: np.ndarray, raw_tokens: Sequence[int], raw_box: Sequence[float], previous_box: Sequence[float], image_shape: Sequence[int], *, budget: int=9, top_k: int=4, radius: int=16, beam_width: int=64) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Return raw + up to ``budget`` AR proposals in image xywh coordinates."""
    token_rows = rank_coordinate_proposals(coordinate_logits, raw_tokens, budget=budget, top_k=top_k, radius=radius, beam_width=beam_width)
    raw = np.asarray(raw_box, dtype=np.float32).reshape(1, 4)
    if not token_rows:
        return (raw, token_rows)
    decoded = decode_tokens_to_image_boxes(token_rows, previous_box, image_shape)
    boxes = np.concatenate((raw, decoded), axis=0)
    return (boxes, token_rows)
