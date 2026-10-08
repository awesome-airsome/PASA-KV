import math

import pytest
import torch

from pasakv.core import (
    action_facing_influence,
    aggregate_task_scores,
    joint_kv_objective,
    rank_layer_scores,
    tail_av_pair_score,
    top_fraction_mean,
)


def test_full_key_softmax_then_visual_slice_without_renormalization():
    q = torch.zeros(1, 1, 1, 2)
    k = torch.zeros(1, 1, 4, 2)
    v = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]]]])
    result = action_facing_influence(q, k, v, [0, 1])
    assert torch.allclose(result.attention, torch.full((1, 1, 1, 2), 0.25))
    assert torch.allclose(result.value_norm, torch.tensor([[[1.0, 2.0]]]))
    assert torch.allclose(result.contribution, torch.tensor([[[[0.25, 0.5]]]]))
    assert result.attention.sum().item() == pytest.approx(0.5)


def test_masked_softmax_normalizes_over_all_valid_keys():
    q = torch.zeros(1, 1, 1, 1)
    k = torch.zeros(1, 1, 4, 1)
    v = torch.ones(1, 1, 4, 2)
    result = action_facing_influence(q, k, v, [0, 1], key_valid_mask=[True, True, True, False])
    assert result.attention.sum().item() == pytest.approx(2 / 3)


def test_query_dependent_mask_and_additive_bias_apply_before_visual_slice():
    q = torch.zeros(1, 1, 2, 1)
    k = torch.zeros(1, 1, 4, 1)
    v = torch.ones(1, 1, 4, 1)
    valid = torch.tensor([[[True, True, True, False], [True, False, True, False]]])
    bias = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]])
    result = action_facing_influence(
        q, k, v, [0, 2], key_valid_mask=valid, additive_attention_bias=bias
    )
    assert torch.allclose(result.attention[0, 0, 0], torch.tensor([1 / 3, 1 / 3]))
    expected_second = torch.tensor([1 / (1 + math.e), math.e / (1 + math.e)])
    assert torch.allclose(result.attention[0, 0, 1], expected_second)
    assert result.attention[0, 0, 1].sum().item() == pytest.approx(1.0)


def test_attention_rejects_all_masked_query_rows():
    q = torch.zeros(1, 1, 1, 1)
    k = torch.zeros(1, 1, 2, 1)
    v = torch.ones(1, 1, 2, 1)
    with pytest.raises(ValueError, match="at least one finite"):
        action_facing_influence(q, k, v, [0], key_valid_mask=[False, False])


def test_top_fraction_uses_ceiling_and_minimum_one():
    mean, count = top_fraction_mean(torch.arange(1.0, 11.0), 0.25)
    assert count == 3
    assert mean.item() == pytest.approx(9.0)
    mean, count = top_fraction_mean(torch.tensor([4.0, 2.0]), 0.005)
    assert count == 1
    assert mean.item() == pytest.approx(4.0)


def test_tailav_floored_log_ratio():
    clean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    patched = 2.0 * clean
    result = tail_av_pair_score(clean, patched, fraction=0.5)
    assert result.score == pytest.approx(math.log(2.0))
    assert result.top_count == 2


def test_joint_kv_is_equal_weighted_relative_displacement():
    clean = torch.ones(2, 3)
    k_patch = clean * 2
    v_patch = clean * 3
    score = joint_kv_objective(clean, k_patch, clean, v_patch)
    assert score.item() == pytest.approx(1.5)


def test_task_scores_weight_tasks_equally_and_rank_ties_by_layer():
    assert aggregate_task_scores({0: [1.0, 3.0], 1: [10.0]}, expected_task_ids=[0, 1]) == pytest.approx(6.0)
    assert rank_layer_scores({3: 1.0, 2: 1.0, 0: 2.0}) == [(0, 2.0), (2, 1.0), (3, 1.0)]
