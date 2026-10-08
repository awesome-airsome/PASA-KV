"""Model-agnostic PASA-KV objectives and action-facing Tail-AV scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class ActionFacingInfluence:
    """Visual slice from full-key attention and its attention-weighted values."""

    attention: torch.Tensor  # [B, H, Q, T]
    value_norm: torch.Tensor  # [B, H, T]
    contribution: torch.Tensor  # [B, H, Q, T] = A * ||V||_2


@dataclass(frozen=True)
class TailAVScore:
    score: float
    clean_tail_mean: float
    patched_tail_mean: float
    ratio: float
    top_count: int


def relative_displacement(clean: torch.Tensor, patched: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return ||patched-clean||_F / max(||clean||_F, eps), accumulated in FP32."""
    if clean.shape != patched.shape:
        raise ValueError(f"K/V shape mismatch: {tuple(clean.shape)} != {tuple(patched.shape)}")
    if eps <= 0:
        raise ValueError("eps must be positive")
    clean32 = clean.detach().float()
    patched32 = patched.float()
    numerator = torch.linalg.vector_norm((patched32 - clean32).reshape(-1), ord=2)
    denominator = torch.linalg.vector_norm(clean32.reshape(-1), ord=2).clamp_min(eps)
    return numerator / denominator


def joint_kv_objective(
    clean_key: torch.Tensor,
    patched_key: torch.Tensor,
    clean_value: torch.Tensor,
    patched_value: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Equal-weight K/V relative displacement for one layer."""
    return 0.5 * (
        relative_displacement(clean_key, patched_key, eps)
        + relative_displacement(clean_value, patched_value, eps)
    )


def mean_layer_objective(objectives: Sequence[torch.Tensor]) -> torch.Tensor:
    """Mean selected-layer objective used by fresh multi-route training."""
    if not objectives:
        raise ValueError("At least one layer objective is required")
    return torch.stack([value.float() for value in objectives]).mean()


def action_facing_influence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    visual_key_indices: Sequence[int] | torch.Tensor,
    *,
    key_valid_mask: torch.Tensor | None = None,
    additive_attention_bias: torch.Tensor | None = None,
    scale: float | None = None,
) -> ActionFacingInfluence:
    """Compute action-query attention over all keys, then slice visual columns.

    Inputs must be attention-ready Q/K/V after model-specific Q/K normalization
    and positional transforms. Shapes are Q=[B,H,Q,D], K=[B,H,K,D],
    V=[B,H,K,Dv]. The optional validity mask can broadcast from [K], [B,K],
    [B,Q,K], or [B,H,Q,K], where True means a valid key. An additive attention
    bias (for example, a causal/position bias) must also broadcast to [B,H,Q,K].
    The visual slice is never renormalized.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q, K, and V must have shape [B,H,sequence,D]")
    if query.shape[:2] != key.shape[:2] or key.shape[:3] != value.shape[:3]:
        raise ValueError("Q/K/V batch, head, or key dimensions do not align")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Q and K head dimensions must match")
    batch, _, key_count, _ = key.shape
    indices = torch.as_tensor(visual_key_indices, dtype=torch.long, device=key.device).flatten()
    if indices.numel() == 0:
        raise ValueError("visual_key_indices must not be empty")
    if int(indices.min()) < 0 or int(indices.max()) >= key_count:
        raise IndexError(f"Visual key index outside [0, {key_count})")
    if indices.unique().numel() != indices.numel():
        raise ValueError("visual_key_indices must be unique")

    multiplier = float(scale) if scale is not None else 1.0 / math.sqrt(query.shape[-1])
    logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) * multiplier
    target_shape = tuple(logits.shape)
    if key_valid_mask is not None:
        valid = torch.as_tensor(key_valid_mask, dtype=torch.bool, device=key.device)
        if valid.ndim == 1:
            valid = valid[None, None, None, :]
        elif valid.ndim == 2:
            valid = valid[:, None, None, :]
        elif valid.ndim == 3:
            valid = valid[:, None, :, :]
        elif valid.ndim != 4:
            raise ValueError("key_valid_mask must have 1, 2, 3, or 4 dimensions")
        try:
            valid = torch.broadcast_to(valid, target_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"key_valid_mask cannot broadcast to attention shape {target_shape}"
            ) from exc
        logits = logits.masked_fill(~valid, -torch.inf)

    if additive_attention_bias is not None:
        bias = torch.as_tensor(additive_attention_bias, dtype=logits.dtype, device=logits.device)
        try:
            logits = logits + torch.broadcast_to(bias, target_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"additive_attention_bias cannot broadcast to attention shape {target_shape}"
            ) from exc

    if bool(torch.isnan(logits).any()) or bool(torch.isposinf(logits).any()):
        raise ValueError("Attention logits must not contain NaN or positive infinity")
    if not bool(torch.isfinite(logits).any(dim=-1).all()):
        raise ValueError("Every attention row must retain at least one finite valid key")

    full_attention = torch.softmax(logits, dim=-1)
    visual_attention = full_attention.index_select(-1, indices)
    visual_value = value.float().index_select(2, indices)
    value_norm = torch.linalg.vector_norm(visual_value, ord=2, dim=-1)
    contribution = visual_attention * value_norm.unsqueeze(-2)
    return ActionFacingInfluence(visual_attention, value_norm, contribution)


def top_fraction_mean(values: torch.Tensor, fraction: float = 0.005) -> tuple[torch.Tensor, int]:
    """Mean of the largest ceil(fraction * N) values, with a minimum count of 1."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    flat = values.float().reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ValueError("values must be non-empty and finite")
    count = max(1, math.ceil(fraction * flat.numel()))
    return torch.topk(flat, count, largest=True, sorted=False).values.mean(), count


def tail_av_pair_score(
    clean: ActionFacingInfluence | torch.Tensor,
    patched: ActionFacingInfluence | torch.Tensor,
    *,
    fraction: float = 0.005,
    delta: float = 1e-12,
) -> TailAVScore:
    """Compute the floored log ratio of independently reduced clean/patch tails."""
    clean_values = clean.contribution if isinstance(clean, ActionFacingInfluence) else clean
    patch_values = patched.contribution if isinstance(patched, ActionFacingInfluence) else patched
    if clean_values.shape != patch_values.shape:
        raise ValueError("Clean and patched influence shapes must match")
    clean_mean, count = top_fraction_mean(clean_values, fraction)
    patched_mean, patch_count = top_fraction_mean(patch_values, fraction)
    if count != patch_count:
        raise AssertionError("Clean and patched influence element counts differ")
    clean_scalar = float(clean_mean.detach().cpu())
    patch_scalar = float(patched_mean.detach().cpu())
    ratio = max(patch_scalar / max(clean_scalar, delta), delta)
    return TailAVScore(
        score=math.log(ratio),
        clean_tail_mean=clean_scalar,
        patched_tail_mean=patch_scalar,
        ratio=ratio,
        top_count=count,
    )


def aggregate_task_scores(
    task_scores: Mapping[int, Sequence[float]],
    *,
    expected_task_ids: Sequence[int] | None = None,
) -> float:
    """Average observations within task, then weight each task equally."""
    if not task_scores:
        raise ValueError("No task scores supplied")
    expected = set(int(task) for task in expected_task_ids) if expected_task_ids is not None else set(task_scores)
    actual = set(int(task) for task in task_scores)
    if actual != expected:
        raise ValueError(f"Task set mismatch: expected {sorted(expected)}, got {sorted(actual)}")
    per_task = []
    for task in sorted(expected):
        values = [float(value) for value in task_scores[task]]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError(f"Task {task} has no finite scores")
        per_task.append(fmean(values))
    return fmean(per_task)


def rank_layer_scores(scores: Mapping[int, float]) -> list[tuple[int, float]]:
    """Sort descending by score; ascending layer ID breaks exact ties."""
    if not scores:
        raise ValueError("No layer scores supplied")
    rows = [(int(layer), float(score)) for layer, score in scores.items()]
    if len({layer for layer, _ in rows}) != len(rows):
        raise ValueError("Duplicate layer IDs")
    if any(layer < 0 or not math.isfinite(score) for layer, score in rows):
        raise ValueError("Layer IDs must be nonnegative and scores finite")
    return sorted(rows, key=lambda row: (-row[1], row[0]))


def select_topk(ranking: Sequence[tuple[int, float]], k: int) -> list[int]:
    if k < 1 or k > len(ranking):
        raise ValueError(f"k must lie in [1, {len(ranking)}]")
    return [int(layer) for layer, _ in ranking[:k]]
