import pytest
import torch

from pasakv.evaluation import validate_eval_grid
from pasakv.patch import initialize_patch, overlay_video
from pasakv.sampling import task_balanced_schedule
from pasakv.adapters.base import KVPair, TrainingExample
from pasakv.training import PatchTrainConfig, train_patch


def test_patch_initialization_is_reproducible_and_bounded():
    a = initialize_patch(4, 5, seed=42)
    b = initialize_patch(4, 5, seed=42)
    assert torch.equal(a, b)
    assert float(a.detach().min()) >= 0.0 and float(a.detach().max()) <= 1.0


def test_overlay_video_changes_only_selected_frame_and_rectangle():
    video = torch.zeros(1, 3, 2, 8, 8)
    patch = torch.ones(3, 2, 3)
    output = overlay_video(video, patch, x=2, y=1, frame_index=0, value_range="0_1")
    assert torch.all(output[:, :, 1] == 0)
    assert torch.all(output[:, :, 0, 1:3, 2:5] == 1)
    assert torch.all(output[:, :, 0, 0, 0] == 0)


def test_task_balanced_schedule_is_deterministic_and_exact():
    pools = {task: [task * 100 + i for i in range(7)] for task in range(10)}
    first = task_balanced_schedule(pools, 100, seed=17)
    second = task_balanced_schedule(pools, 100, seed=17)
    assert first == second
    assert len(first) == 100
    counts = {task: sum(row.task_id == task for row in first) for task in range(10)}
    assert counts == {task: 10 for task in range(10)}


def test_eval_grid_rejects_duplicate_and_reports_complete_sr():
    rows = [
        {"task_id": task, "trial_id": trial, "success": bool((task + trial) % 2)}
        for task in range(2)
        for trial in range(2)
    ]
    summary = validate_eval_grid(rows, task_ids=[0, 1], trial_ids=[0, 1])
    assert summary["episodes"] == 4
    assert summary["policy_sr"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_eval_grid(rows + [rows[0]], task_ids=[0, 1], trial_ids=[0, 1])


def test_task_balanced_requires_divisible_exposures():
    with pytest.raises(ValueError, match="divisible"):
        task_balanced_schedule({0: [0], 1: [1]}, 3, seed=0)


class ToyAdapter:
    num_layers = 2

    def capture_kv_pairs(self, example, patch, layer_ids):
        clean = torch.ones(1, 1, 3, 2)
        delta = patch.mean()
        return {
            layer: KVPair(
                clean_key=clean,
                clean_value=clean,
                patched_key=clean + delta,
                patched_value=clean + 2 * delta,
            )
            for layer in layer_ids
        }


def test_training_loop_makes_one_update_per_exposure_and_returns_fresh_patch():
    adapter = ToyAdapter()
    examples = lambda step: TrainingExample(
        payload=None, task_id=0, sample_id=f"train-{step}"
    )
    result = train_patch(
        adapter,
        examples,
        [0, 1],
        PatchTrainConfig(exposures=3, patch_height=2, patch_width=2, seed=9),
        device="cpu",
    )
    assert len(result.loss_trace) == 3
    assert len(result.objective_trace) == 3
    assert result.sample_ids == ("train-0", "train-1", "train-2")
    assert result.patch.requires_grad is False
    assert float(result.patch.min()) >= 0.0
    assert float(result.patch.max()) <= 1.0


def test_training_rejects_scoring_sample_id_before_update():
    adapter = ToyAdapter()
    examples = lambda step: TrainingExample(
        payload=None, task_id=0, sample_id=f"train-{step}"
    )
    with pytest.raises(ValueError, match="frozen scoring/exclusion set"):
        train_patch(
            adapter,
            examples,
            [0],
            PatchTrainConfig(exposures=2, patch_height=2, patch_width=2, seed=9),
            forbidden_sample_ids=["train-1"],
            device="cpu",
        )
