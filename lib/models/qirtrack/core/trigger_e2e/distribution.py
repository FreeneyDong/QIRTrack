"""Canonical ordered support for Trigger-E2E IoU distributions."""
from __future__ import annotations
import torch

def iou_support_points(points: int, *, dtype: torch.dtype | None=None, device: torch.device | str | None=None) -> torch.Tensor:
    """Return ``points`` endpoint-inclusive IoU support values on ``[0, 1]``."""
    if points < 2:
        raise ValueError('IoU support needs at least two points')
    return torch.linspace(0.0, 1.0, points, dtype=dtype, device=device)

def soft_iou_targets(iou: torch.Tensor, bins: int=50) -> torch.Tensor:
    """Interpolate IoU over neighbouring support points while preserving its mean."""
    points = bins
    if points < 2:
        raise ValueError('IoU support needs at least two points')
    if torch.any(~torch.isfinite(iou)) or torch.any((iou < 0) | (iou > 1)):
        raise ValueError('iou targets must be finite and inside [0,1]')
    position = iou * (points - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp_max(points - 1)
    upper_weight = position - lower.to(position.dtype)
    lower_weight = 1.0 - upper_weight
    target = torch.zeros(*iou.shape, points, dtype=iou.dtype, device=iou.device)
    target.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    target.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return target

def distribution_mean(probability: torch.Tensor) -> torch.Tensor:
    """Return expected IoU on the canonical endpoint-inclusive support."""
    support = iou_support_points(probability.shape[-1], dtype=probability.dtype, device=probability.device)
    return (probability * support).sum(dim=-1)

def discrete_crps(probability: torch.Tensor, target_iou: torch.Tensor) -> torch.Tensor:
    """Approximate continuous CRPS using the canonical support and grid spacing."""
    points = probability.shape[-1]
    support = iou_support_points(points, dtype=probability.dtype, device=probability.device)
    predicted_cdf = probability.cumsum(dim=-1)
    observed_cdf = (target_iou.unsqueeze(-1) <= support).to(probability.dtype)
    spacing = 1.0 / (points - 1)
    return ((predicted_cdf - observed_cdf) ** 2).sum(dim=-1) * spacing

def risk_below_thresholds(probability: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Derive ``P(IoU < threshold)`` without threshold-specific heads."""
    support = iou_support_points(probability.shape[-1], dtype=probability.dtype, device=probability.device)
    threshold = thresholds.to(dtype=probability.dtype, device=probability.device)
    membership = support.unsqueeze(0) < threshold.unsqueeze(1)
    return torch.einsum('...b,tb->...t', probability, membership.to(probability.dtype))
