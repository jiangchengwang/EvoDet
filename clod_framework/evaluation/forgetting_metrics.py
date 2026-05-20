from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ForgettingSummary:
    average_forgetting: float
    per_task_forgetting: dict[int, float]


def compute_average_forgetting(history: Mapping[int, Mapping[int, float]]) -> ForgettingSummary:
    """Compute simple forgetting from a matrix of task performance.

    Args:
        history: mapping `eval_stage -> {task_id -> score}`. For each old task,
            forgetting = best previous score - final score.
    """
    if not history:
        return ForgettingSummary(average_forgetting=0.0, per_task_forgetting={})
    final_stage = max(history)
    final_scores = history[final_stage]
    per_task: dict[int, float] = {}
    for task_id in final_scores:
        previous = [scores.get(task_id) for stage, scores in history.items() if stage <= final_stage]
        previous = [x for x in previous if x is not None]
        if not previous:
            continue
        per_task[task_id] = max(previous) - final_scores[task_id]
    avg = sum(per_task.values()) / len(per_task) if per_task else 0.0
    return ForgettingSummary(average_forgetting=avg, per_task_forgetting=per_task)
