"""Adapter contract between PASA-KV and external world-action-model code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import torch


@dataclass(frozen=True)
class TrainingExample:
    payload: Any
    task_id: int
    sample_id: str
    # Store/replay actual stochastic tensors here when the victim uses them.
    randomness: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class KVPair:
    clean_key: torch.Tensor
    clean_value: torch.Tensor
    patched_key: torch.Tensor
    patched_value: torch.Tensor


@dataclass(frozen=True)
class QKVPair:
    clean_query: torch.Tensor
    clean_key: torch.Tensor
    clean_value: torch.Tensor
    patched_query: torch.Tensor
    patched_key: torch.Tensor
    patched_value: torch.Tensor
    visual_key_indices: Sequence[int] | torch.Tensor
    clean_key_valid_mask: torch.Tensor | None = None
    patched_key_valid_mask: torch.Tensor | None = None
    clean_attention_bias: torch.Tensor | None = None
    patched_attention_bias: torch.Tensor | None = None
    attention_scale: float | None = None


class PASAAdapter(Protocol):
    """Methods an upstream-model integration must implement.

    capture_*_pairs must run clean and patched forwards on the same example
    and replay the same saved stochastic tensors from TrainingExample.
    K/V and Q/K/V outputs must be the post-transform, attention-ready tensors.
    """

    num_layers: int

    def capture_kv_pairs(
        self, example: TrainingExample, patch: torch.Tensor, layer_ids: Sequence[int]
    ) -> Mapping[int, KVPair]: ...

    def capture_qkv_pairs(
        self, example: TrainingExample, patch: torch.Tensor, layer_ids: Sequence[int]
    ) -> Mapping[int, QKVPair]: ...
