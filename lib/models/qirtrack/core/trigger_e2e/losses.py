"""Ordered IoU-distribution losses for Trigger-E2E."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from lib.models.qirtrack.core.trigger_e2e.distribution import discrete_crps, distribution_mean, risk_below_thresholds, soft_iou_targets

def transition_range_weights(history_iou: torch.Tensor, history_mask: torch.Tensor, scales: tuple[int, ...]=(4, 8, 15)) -> dict[str, torch.Tensor]:
    """Continuous stable-history transition weights for training only.

    ``history_iou`` includes the current frame in its last position.  A scale
    contributes only when all of its preceding frames are real observations,
    so left padding at sequence starts can never create a transition label.
    """
    if history_iou.ndim != 2 or history_iou.shape != history_mask.shape:
        raise ValueError('history_iou/history_mask must be aligned [batch, time]')
    if history_iou.shape[1] < 2:
        raise ValueError('transition weights require at least one past frame')
    if not scales or min(scales) < 1 or max(scales) >= history_iou.shape[1]:
        raise ValueError('scales must fit strictly inside the history window')
    if any((left >= right for (left, right) in zip(scales, scales[1:]))):
        raise ValueError('scales must be strictly increasing')
    current = history_iou[:, -1]
    up_sum = torch.zeros_like(current)
    down_sum = torch.zeros_like(current)
    eligible = torch.zeros_like(current)
    for scale in scales:
        past = history_iou[:, -(scale + 1):-1]
        valid = history_mask[:, -(scale + 1):].all(dim=-1)
        past_max = past.max(dim=-1).values
        past_min = past.min(dim=-1).values
        stability = (1.0 - (past_max - past_min)).clamp(0.0, 1.0)
        up = stability * (current - past_max).clamp_min(0.0) * (current[:, None] - past).clamp_min(0.0).mean(dim=-1)
        down = stability * (past_min - current).clamp_min(0.0) * (past - current[:, None]).clamp_min(0.0).mean(dim=-1)
        valid_float = valid.to(history_iou.dtype)
        up_sum = up_sum + valid_float * up
        down_sum = down_sum + valid_float * down
        eligible = eligible + valid_float
    denominator = eligible.clamp_min(1.0)
    return {'up': up_sum / denominator, 'down': down_sum / denominator, 'eligible_scales': eligible}

def stable_history_delta_targets(history_iou: torch.Tensor, history_mask: torch.Tensor, scales: tuple[int, ...]=(4, 8, 15)) -> dict[str, torch.Tensor]:
    """Build threshold-free signed transition targets at several time scales.

    For each scale, the target is the current IoU minus the mean IoU of the
    preceding observations.  The continuous weight is one minus the preceding
    IoU range, so stable histories contribute most.  Ground-truth IoU is used
    only by the offline training loss and never becomes a model input.
    """
    if history_iou.ndim != 2 or history_iou.shape != history_mask.shape:
        raise ValueError('history_iou/history_mask must be aligned [batch, time]')
    if history_iou.shape[1] < 2:
        raise ValueError('transition targets require at least one past frame')
    if not scales or min(scales) < 1 or max(scales) >= history_iou.shape[1]:
        raise ValueError('scales must fit strictly inside the history window')
    if any((left >= right for (left, right) in zip(scales, scales[1:]))):
        raise ValueError('scales must be strictly increasing')
    current = history_iou[:, -1]
    targets = []
    weights = []
    for scale in scales:
        past = history_iou[:, -(scale + 1):-1]
        valid = history_mask[:, -(scale + 1):].all(dim=-1)
        stability = (1.0 - (past.max(dim=-1).values - past.min(dim=-1).values)).clamp(0.0, 1.0)
        targets.append(current - past.mean(dim=-1))
        weights.append(valid.to(history_iou.dtype) * stability)
    return {'delta': torch.stack(targets, dim=-1), 'weight': torch.stack(weights, dim=-1)}

def normalized_weighted_mean(values: torch.Tensor, weights: torch.Tensor, epsilon: float=1e-08) -> torch.Tensor:
    """Return a zero-valued differentiable loss when a batch has no event."""
    if values.shape != weights.shape:
        raise ValueError('values and weights must have equal shape')
    if epsilon <= 0.0:
        raise ValueError('epsilon must be positive')
    numerator = (values * weights).sum()
    denominator = weights.sum()
    weighted = numerator / denominator.clamp_min(epsilon)
    return torch.where(denominator > 0.0, weighted, values.sum() * 0.0)

@dataclass(frozen=True)
class DistributionLossWeights:
    """Pre-registered relative weights; never fit on evaluation-only data."""
    cross_entropy: float = 1.0
    crps: float = 1.0
    mean: float = 1.0

class IoUDistributionLoss(nn.Module):
    """Soft-target CE + ordered CRPS + mean Smooth-L1."""

    def __init__(self, weights: DistributionLossWeights=DistributionLossWeights(), smooth_l1_beta: float=0.1) -> None:
        super().__init__()
        if smooth_l1_beta <= 0:
            raise ValueError('smooth_l1_beta must be positive')
        if min(weights.cross_entropy, weights.crps, weights.mean) < 0:
            raise ValueError('loss weights must be non-negative')
        self.weights = weights
        self.smooth_l1_beta = smooth_l1_beta

    def forward(self, logits: torch.Tensor, target_iou: torch.Tensor) -> dict[str, torch.Tensor]:
        per_example = self.per_example(logits, target_iou)
        return {'loss': per_example['loss'].mean(), 'cross_entropy': per_example['cross_entropy'].mean(), 'crps': per_example['crps'].mean(), 'mean_smooth_l1': per_example['mean_smooth_l1'].mean(), 'predicted_iou': per_example['predicted_iou'], 'probability': per_example['probability']}

    def per_example(self, logits: torch.Tensor, target_iou: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return unreduced losses for soft expert-responsibility targets."""
        if logits.shape[:-1] != target_iou.shape:
            raise ValueError(f'logits batch shape {logits.shape[:-1]} != target {target_iou.shape}')
        target = soft_iou_targets(target_iou, bins=logits.shape[-1])
        log_probability = F.log_softmax(logits, dim=-1)
        probability = log_probability.exp()
        cross_entropy = -(target * log_probability).sum(dim=-1)
        crps = discrete_crps(probability, target_iou)
        predicted_iou = distribution_mean(probability)
        mean_loss = F.smooth_l1_loss(predicted_iou, target_iou, beta=self.smooth_l1_beta, reduction='none')
        total = self.weights.cross_entropy * cross_entropy + self.weights.crps * crps + self.weights.mean * mean_loss
        return {'loss': total, 'cross_entropy': cross_entropy, 'crps': crps, 'mean_smooth_l1': mean_loss, 'predicted_iou': predicted_iou, 'probability': probability}
