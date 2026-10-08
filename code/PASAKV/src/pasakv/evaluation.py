"""Trial-grid integrity checks and Policy SR aggregation."""

from __future__ import annotations

from typing import Iterable, Mapping


def validate_eval_grid(
    records: Iterable[Mapping[str, object]],
    *,
    task_ids: Iterable[int],
    trial_ids: Iterable[int],
) -> dict[str, float | int]:
    """Require every expected task/trial exactly once before computing SR."""
    expected_tasks = {int(value) for value in task_ids}
    expected_trials = {int(value) for value in trial_ids}
    if not expected_tasks or not expected_trials:
        raise ValueError("Expected task and trial sets must be non-empty")
    seen: set[tuple[int, int]] = set()
    successes = 0
    for row in records:
        task = int(row["task_id"])
        trial = int(row["trial_id"])
        success = row["success"]
        if task not in expected_tasks or trial not in expected_trials:
            raise ValueError(f"Unexpected task/trial key {(task, trial)}")
        key = (task, trial)
        if key in seen:
            raise ValueError(f"Duplicate task/trial key {key}")
        if not isinstance(success, bool):
            raise TypeError(f"success must be bool for {key}")
        seen.add(key)
        successes += int(success)
    expected = {(task, trial) for task in expected_tasks for trial in expected_trials}
    missing = expected - seen
    if missing:
        raise ValueError(f"Evaluation grid incomplete: {len(missing)} keys missing")
    episodes = len(expected)
    return {"successes": successes, "episodes": episodes, "policy_sr": successes / episodes}
