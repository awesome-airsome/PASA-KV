"""Fresh singleton-probe and selected-layer joint K/V patch optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch.optim import AdamW

from .adapters.base import PASAAdapter, TrainingExample
from .core import joint_kv_objective, mean_layer_objective
from .patch import initialize_patch


@dataclass(frozen=True)
class PatchTrainConfig:
    exposures: int
    patch_height: int
    patch_width: int
    seed: int = 42
    learning_rate: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    displacement_epsilon: float = 1e-8

    def validate(self) -> None:
        if self.exposures < 1 or self.patch_height < 1 or self.patch_width < 1:
            raise ValueError("Exposure count and patch dimensions must be positive")
        if self.learning_rate <= 0 or self.epsilon <= 0 or self.displacement_epsilon <= 0:
            raise ValueError("Learning rate and epsilons must be positive")


@dataclass(frozen=True)
class PatchTrainResult:
    patch: torch.Tensor
    loss_trace: tuple[float, ...]
    objective_trace: tuple[float, ...]
    sample_ids: tuple[str, ...]


def train_patch(
    adapter: PASAAdapter,
    sample_at_exposure: Callable[[int], TrainingExample],
    target_layers: Sequence[int],
    config: PatchTrainConfig,
    *,
    forbidden_sample_ids: Sequence[str] = (),
    device: str | torch.device,
) -> PatchTrainResult:
    """Train a fresh patch; one sample and one AdamW update per exposure."""
    config.validate()
    layers = [int(layer) for layer in target_layers]
    if not layers or len(layers) != len(set(layers)):
        raise ValueError("target_layers must be a non-empty unique sequence")
    if any(layer < 0 or layer >= adapter.num_layers for layer in layers):
        raise ValueError(f"Layer IDs must lie in [0, {adapter.num_layers})")

    patch = initialize_patch(
        config.patch_height, config.patch_width, seed=config.seed, device=device
    )
    optimizer = AdamW(
        [patch],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    losses: list[float] = []
    objectives: list[float] = []
    sample_ids: list[str] = []
    forbidden = {str(sample_id) for sample_id in forbidden_sample_ids}

    for exposure in range(config.exposures):
        example = sample_at_exposure(exposure)
        if not example.sample_id:
            raise ValueError(f"Missing stable sample_id at exposure {exposure}")
        sample_id = str(example.sample_id)
        if sample_id in forbidden:
            raise ValueError(
                f"Training sample {sample_id!r} is in the frozen scoring/exclusion set"
            )
        sample_ids.append(sample_id)
        pairs = adapter.capture_kv_pairs(example, patch, layers)
        if set(pairs) != set(layers):
            raise RuntimeError(f"Adapter returned layers {sorted(pairs)}, expected {sorted(layers)}")
        layer_objectives = [
            joint_kv_objective(
                pairs[layer].clean_key,
                pairs[layer].patched_key,
                pairs[layer].clean_value,
                pairs[layer].patched_value,
                eps=config.displacement_epsilon,
            )
            for layer in layers
        ]
        objective = mean_layer_objective(layer_objectives)
        loss = -objective
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite patch objective at exposure {exposure + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if patch.grad is None or not bool(torch.isfinite(patch.grad).all()):
            raise FloatingPointError(f"Invalid patch gradient at exposure {exposure + 1}")
        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)
        losses.append(float(loss.detach().cpu()))
        objectives.append(float(objective.detach().cpu()))

    return PatchTrainResult(
        patch=patch.detach().float().cpu(),
        loss_trace=tuple(losses),
        objective_trace=tuple(objectives),
        sample_ids=tuple(sample_ids),
    )
