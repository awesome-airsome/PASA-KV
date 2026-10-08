"""Deterministic task-balanced sample-exposure schedules."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SampleExposure:
    exposure_index: int
    task_id: int
    sample_index: int


def task_balanced_schedule(
    task_to_sample_indices: Mapping[int, Sequence[int]],
    exposures: int,
    *,
    seed: int,
) -> list[SampleExposure]:
    """Create an exactly balanced schedule, cycling shuffled task pools as needed."""
    tasks = sorted(int(task) for task in task_to_sample_indices)
    if not tasks or any(not task_to_sample_indices[task] for task in tasks):
        raise ValueError("Every task must have at least one sample")
    if exposures <= 0 or exposures % len(tasks) != 0:
        raise ValueError("exposures must be positive and divisible by the number of tasks")
    original_pools = {task: [int(index) for index in task_to_sample_indices[task]] for task in tasks}
    if any(len(pool) != len(set(pool)) for pool in original_pools.values()):
        raise ValueError("Sample indices must be unique within each task")
    rng = random.Random(int(seed))
    per_task = exposures // len(tasks)
    pools: dict[int, list[int]] = {}
    schedule: list[SampleExposure] = []
    for round_index in range(per_task):
        task_order = list(tasks)
        rng.shuffle(task_order)
        for task in task_order:
            if round_index % len(original_pools[task]) == 0:
                pools[task] = list(original_pools[task])
                rng.shuffle(pools[task])
            sample_index = pools[task][round_index % len(original_pools[task])]
            schedule.append(SampleExposure(len(schedule), task, sample_index))
    if len(schedule) != exposures:
        raise AssertionError("Internal schedule length mismatch")
    return schedule
