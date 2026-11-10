"""Screening eval: recall at a fixed false alert budget, the way a screening team runs it.

For each system, thresholds are chosen on the dev split so that at most 1, 5 or 10 of every
1,000 clean names raise an alert. The same thresholds are then applied, unchanged, to the
test split, and test recall and test false alerts are reported.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from sanctions_mcp.evalset import Query

FP_BUDGETS = (1, 5, 10)  # false alerts per 1,000 clean screenings
SLICE_BUDGET = 5


@dataclass
class SystemRun:
    system: str
    split: str
    target_score: dict[str, float] = field(default_factory=dict)  # positives
    max_score: dict[str, float] = field(default_factory=dict)  # clean names
    candidate_hit: dict[str, bool] = field(default_factory=dict)
    latency_ms: list[float] = field(default_factory=list)


def latency_indices(n: int, sample: int) -> list[int]:
    """Evenly spaced query positions. Positives and clean names sit in two blocks, so a
    stride takes both in proportion."""
    return list(range(0, n, max(1, n // sample))) if n and sample else []


def run_system(
    system,
    queries: Sequence[Query],
    split: str,
    *,
    latency_sample: int = 1000,
    on_progress: Callable[[int, int], None] | None = None,
) -> SystemRun:
    """Score every query. Latency is timed one query at a time, on the same evenly spaced
    sample of queries for every system, after a few untimed warm up calls."""
    out = SystemRun(system=system.name, split=split)
    texts = [q.text for q in queries]
    for text in texts[:3]:
        system.scores(text)
    timed = set(latency_indices(len(texts), latency_sample))
    batch = hasattr(system, "scores_many") and not hasattr(system, "candidate_entities")
    results = system.scores_many(texts) if batch else []
    for i, text in enumerate(texts):
        if i in timed:
            start = time.perf_counter()
            scores = system.scores(text)
            out.latency_ms.append((time.perf_counter() - start) * 1000)
        elif not batch:
            scores = system.scores(text)
        if not batch:
            results.append(scores)
        if on_progress and not batch and (i + 1) % 2000 == 0:
            on_progress(i + 1, len(texts))
    for q, scores in zip(queries, results, strict=True):
        if q.target is None:
            out.max_score[q.qid] = max(scores.values(), default=0.0)
        else:
            out.target_score[q.qid] = scores.get(q.target, 0.0)
    if hasattr(system, "candidate_entities"):
        for q in queries:
            if q.target is not None:
                out.candidate_hit[q.qid] = q.target in system.candidate_entities(q.text)
    return out


def threshold_for(clean_scores: Sequence[float], budget_per_1000: float) -> float:
    """Lowest threshold at which at most `budget` of every 1,000 clean names alert
    (an alert is score >= threshold)."""
    ranked = sorted(clean_scores, reverse=True)
    allowed = math.floor(budget_per_1000 / 1000 * len(ranked))
    if allowed >= len(ranked):
        return 0.0
    return float(np.nextafter(ranked[allowed], np.inf))


def _rate(values: Sequence[float], t: float) -> float:
    return float(np.mean([v >= t for v in values])) if values else float("nan")


def summarize(dev: SystemRun, test: SystemRun, test_queries: Sequence[Query]) -> dict:
    dev_clean = list(dev.max_score.values())
    points = []
    for budget in FP_BUDGETS:
        t = threshold_for(dev_clean, budget)
        points.append(
            {
                "budget_per_1000": budget,
                "threshold": t,
                "test_recall": _rate(list(test.target_score.values()), t),
                "test_false_alerts_per_1000": 1000 * _rate(list(test.max_score.values()), t),
                "dev_recall": _rate(list(dev.target_score.values()), t),
            }
        )
    t_slice = next(p["threshold"] for p in points if p["budget_per_1000"] == SLICE_BUDGET)
    by_kind: dict[str, list[float]] = defaultdict(list)
    by_group: dict[str, list[float]] = defaultdict(list)
    clean_by_group: dict[str, list[float]] = defaultdict(list)
    for q in test_queries:
        if q.target is None:
            clean_by_group[q.group].append(test.max_score[q.qid])
        else:
            by_kind[q.kind].append(test.target_score[q.qid])
            by_group[q.group].append(test.target_score[q.qid])
    lat = np.asarray(test.latency_ms or [float("nan")])
    return {
        "operating_points": points,
        "recall_by_kind": {
            k: {"n": len(v), "recall": _rate(v, t_slice)} for k, v in sorted(by_kind.items())
        },
        "recall_by_group": {
            g: {"n": len(v), "recall": _rate(v, t_slice)} for g, v in sorted(by_group.items())
        },
        "false_alerts_per_1000_by_group": {
            g: {"n": len(v), "rate": 1000 * _rate(v, t_slice)}
            for g, v in sorted(clean_by_group.items())
        },
        "candidate_recall": (
            float(np.mean(list(test.candidate_hit.values()))) if test.candidate_hit else None
        ),
        "latency_ms": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "n": len(test.latency_ms),  # queries timed, the same sample for every system
        },
        "counts": {
            "test_positive": len(test.target_score),
            "test_clean": len(test.max_score),
            "dev_positive": len(dev.target_score),
            "dev_clean": len(dev.max_score),
        },
    }
