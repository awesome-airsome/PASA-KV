"""Minimal Probe100 -> Tail-AV rank -> fresh Top-k patch pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from statistics import fmean
from typing import Callable, Mapping, Sequence

import torch

from .adapters.base import PASAAdapter, QKVPair, TrainingExample
from .core import action_facing_influence, rank_layer_scores, select_topk, tail_av_pair_score
from .training import PatchTrainConfig, PatchTrainResult, train_patch


@dataclass(frozen=True)
class ProbeSweepResult:
    probe_patches: Mapping[int, torch.Tensor]
    task_scores: Mapping[int, Mapping[int, float]]
    layer_scores: Mapping[int, float]
    ranking: tuple[tuple[int, float], ...]
    probe_traces: Mapping[int, PatchTrainResult]
    scoring_sample_ids: tuple[str, ...]


def _influence(pair: QKVPair, *, patched: bool):
    prefix = "patched" if patched else "clean"
    return action_facing_influence(
        getattr(pair, f"{prefix}_query"),
        getattr(pair, f"{prefix}_key"),
        getattr(pair, f"{prefix}_value"),
        pair.visual_key_indices,
        key_valid_mask=getattr(pair, f"{prefix}_key_valid_mask"),
        additive_attention_bias=getattr(pair, f"{prefix}_attention_bias"),
        scale=pair.attention_scale,
    )


def probe100_and_rank(
    adapter: PASAAdapter,
    train_example_at: Callable[[int, int], TrainingExample],
    scoring_examples_by_task: Mapping[int, Sequence[TrainingExample]],
    candidate_layers: Sequence[int],
    probe_config: PatchTrainConfig,
    *,
    device: str | torch.device,
    tail_fraction: float = 0.005,
    expected_task_ids: Sequence[int] = tuple(range(10)),
) -> ProbeSweepResult:
    """Train one singleton per layer and rank only with frozen Tail-AV inputs."""
    layers = [int(layer) for layer in candidate_layers]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("candidate_layers must be non-empty and unique")
    if any(layer < 0 or layer >= adapter.num_layers for layer in layers):
        raise ValueError(f"Candidate layer must lie in [0, {adapter.num_layers})")
    expected = set(int(task) for task in expected_task_ids)
    if set(int(task) for task in scoring_examples_by_task) != expected:
        raise ValueError("Scoring manifest task IDs do not match expected_task_ids")
    if any(not scoring_examples_by_task[task] for task in expected):
        raise ValueError("Every scoring task must have at least one frozen observation")
    scoring_ids = []
    for task in sorted(expected):
        for example in scoring_examples_by_task[task]:
            if example.task_id != task or not example.sample_id:
                raise ValueError("Scoring example task_id/sample_id does not match the frozen manifest")
            scoring_ids.append(str(example.sample_id))
    if len(scoring_ids) != len(set(scoring_ids)):
        raise ValueError("Scoring manifest contains duplicate sample IDs")

    traces: dict[int, PatchTrainResult] = {}
    task_scores: dict[int, dict[int, float]] = {layer: {} for layer in layers}
    for layer in layers:
        trace = train_patch(
            adapter,
            lambda exposure, target=layer: train_example_at(target, exposure),
            [layer],
            replace(probe_config),
            device=device,
        )
        traces[layer] = trace
        overlap = set(trace.sample_ids) & set(scoring_ids)
        if overlap:
            raise ValueError(
                f"Probe training and selector scoring are not disjoint for layer {layer}: "
                f"{len(overlap)} sample IDs overlap"
            )
        probe = trace.patch.to(device=device)
        for task in sorted(expected):
            pair_scores = []
            for example in scoring_examples_by_task[task]:
                with torch.no_grad():
                    captures = adapter.capture_qkv_pairs(example, probe, [layer])
                if set(captures) != {layer}:
                    raise RuntimeError(f"Adapter did not return Q/K/V for layer {layer}")
                qkv = captures[layer]
                clean_inf = _influence(qkv, patched=False)
                patch_inf = _influence(qkv, patched=True)
                pair_scores.append(
                    tail_av_pair_score(clean_inf, patch_inf, fraction=tail_fraction).score
                )
            task_scores[layer][task] = fmean(pair_scores)

    layer_scores = {layer: fmean(scores.values()) for layer, scores in task_scores.items()}
    ranking = rank_layer_scores(layer_scores)
    return ProbeSweepResult(
        probe_patches={layer: result.patch for layer, result in traces.items()},
        task_scores=task_scores,
        layer_scores=layer_scores,
        ranking=tuple(ranking),
        probe_traces=traces,
        scoring_sample_ids=tuple(sorted(scoring_ids)),
    )


def train_fresh_topk_patch(
    adapter: PASAAdapter,
    sample_at_exposure: Callable[[int], TrainingExample],
    ranking: Sequence[tuple[int, float]],
    top_k: int,
    final_config: PatchTrainConfig,
    *,
    excluded_sample_ids: Sequence[str],
    device: str | torch.device,
) -> tuple[list[int], PatchTrainResult]:
    """Train a fresh patch, rejecting final-training overlap with frozen scoring data."""
    selected = select_topk(ranking, top_k)
    result = train_patch(
        adapter,
        sample_at_exposure,
        selected,
        final_config,
        forbidden_sample_ids=excluded_sample_ids,
        device=device,
    )
    overlap = set(result.sample_ids) & {str(sample_id) for sample_id in excluded_sample_ids}
    if overlap:
        raise ValueError(
            f"Final training and excluded scoring set overlap: {len(overlap)} sample IDs"
        )
    return selected, result
