"""A small adapter checks the generic orchestration without claiming model fidelity."""

import torch

from pasakv.adapters.base import KVPair, QKVPair, TrainingExample
from pasakv.pipeline import probe100_and_rank, train_fresh_topk_patch
from pasakv.training import PatchTrainConfig


class ToyAdapter:
    num_layers = 2

    def capture_kv_pairs(self, example, patch, layer_ids):
        clean = torch.ones(1, 1, 2, 2)
        return {
            layer: KVPair(clean, clean, clean + patch.mean() / (layer + 1), clean + patch.mean() / (layer + 1))
            for layer in layer_ids
        }

    def capture_qkv_pairs(self, example, patch, layer_ids):
        query = torch.zeros(1, 1, 1, 2)
        key = torch.zeros(1, 1, 2, 2)
        value = torch.ones(1, 1, 2, 2)
        return {
            layer: QKVPair(
                query, key, value, query, key,
                value + patch.mean() / (layer + 1), [0, 1],
            )
            for layer in layer_ids
        }


def test_probe_rank_and_fresh_training_use_disjoint_samples():
    adapter = ToyAdapter()
    scoring = {
        task: [TrainingExample(None, task, f"score-{task}")]
        for task in (0, 1)
    }
    probe = probe100_and_rank(
        adapter,
        lambda layer, exposure: TrainingExample(None, exposure % 2, f"probe-{layer}-{exposure}"),
        scoring,
        [0, 1],
        PatchTrainConfig(exposures=2, patch_height=2, patch_width=2),
        device="cpu",
        expected_task_ids=(0, 1),
    )
    assert [layer for layer, _ in probe.ranking] == [0, 1]
    assert probe.scoring_sample_ids == ("score-0", "score-1")

    selected, trained = train_fresh_topk_patch(
        adapter,
        lambda exposure: TrainingExample(None, exposure % 2, f"final-{exposure}"),
        probe.ranking,
        1,
        PatchTrainConfig(exposures=2, patch_height=2, patch_width=2, seed=99),
        excluded_sample_ids=probe.scoring_sample_ids,
        device="cpu",
    )
    assert selected == [0]
    assert trained.sample_ids == ("final-0", "final-1")
    assert not torch.equal(trained.patch, probe.probe_patches[0])
