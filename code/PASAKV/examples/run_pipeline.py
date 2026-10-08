"""Run Probe100 and fresh Top-k training with an external adapter/data provider."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from pasakv.adapters.base import PASAAdapter, TrainingExample
from pasakv.pipeline import probe100_and_rank, train_fresh_topk_patch
from pasakv.training import PatchTrainConfig, PatchTrainResult


def run_pipeline(
    *,
    adapter: PASAAdapter,
    probe_example_at: Callable[[int, int], TrainingExample],
    scoring_examples_by_task: Mapping[int, Sequence[TrainingExample]],
    final_example_at: Callable[[int], TrainingExample],
    output_dir: str | Path,
    top_k: int = 3,
    patch_height: int = 50,
    patch_width: int = 50,
    device: str = "cuda",
    expected_task_ids: Sequence[int] = tuple(range(10)),
) -> tuple[list[int], PatchTrainResult]:
    """Caller supplies a frozen victim adapter, placement, and paired forwards.

    Run from code/PASAKV after installing the method package. Sample providers
    must be task-balanced; scoring examples must be held out from training.
    Output files are written only after both training stages finish.
    """
    output = Path(output_dir)
    if not 1 <= top_k <= adapter.num_layers:
        raise ValueError("top_k must fit the candidate layer count")
    if any((output / name).exists() for name in ("ranking.json", "patch.pt")):
        raise FileExistsError("Choose a new output directory to preserve previous artifacts")
    probe_config = PatchTrainConfig(100, patch_height, patch_width)
    final_config = PatchTrainConfig(10_000, patch_height, patch_width)
    probe = probe100_and_rank(
        adapter, probe_example_at, scoring_examples_by_task,
        range(adapter.num_layers), probe_config, device=device,
        expected_task_ids=expected_task_ids,
    )
    ranking, scoring_ids = probe.ranking, probe.scoring_sample_ids
    layer_scores, task_scores = probe.layer_scores, probe.task_scores
    # Release singleton patches/traces before constructing the fresh final patch.
    del probe
    selected, result = train_fresh_topk_patch(
        adapter, final_example_at, ranking, top_k, final_config,
        excluded_sample_ids=scoring_ids, device=device,
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result.patch.detach().cpu(), output / "patch.pt")
    manifest = {
        "ranking": ranking, "selected_layers": selected,
        "layer_scores": layer_scores, "task_scores": task_scores,
        "scoring_sample_ids": scoring_ids,
        "final_training_sample_ids": result.sample_ids,
        "probe_config": asdict(probe_config), "final_config": asdict(final_config),
        "expected_task_ids": list(expected_task_ids),
    }
    (output / "ranking.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return selected, result
