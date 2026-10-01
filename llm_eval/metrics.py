"""Six task-specific metrics (M1-M6) and the weighted quality score Q.

Standard text-similarity metrics (BLEU, ROUGE, BERTScore) measure how similar
two texts look. For expert classification tasks the real questions are
different: did the model cover every required attribute, did it use the right
terms, is it stable across runs, can it interpret, does it know when it is
unsure, and does it invent facts? Each metric below answers one of them.

All metrics return a value on a 0-100 scale.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

# Weights of each metric in the final score Q (they sum to 1.0).
WEIGHTS = {
    "m1": 0.25,  # completeness
    "m2": 0.25,  # terminology accuracy
    "m3": 0.15,  # run-to-run consistency
    "m4": 0.15,  # depth of interpretation
    "m5": 0.10,  # confidence calibration
    "m6": 0.10,  # absence of hallucinations
}

# Lower bound (inclusive) of each quality level for Q.
LEVELS = [
    (80.0, "high"),
    (60.0, "satisfactory"),
    (40.0, "unsatisfactory"),
    (0.0, "unusable"),
]

# Points deducted from 100 for each penalty in M5 and M6.
PENALTY_STEP = 30.0

# A run-to-run Jaccard of at least this value counts as reproducible.
REPRODUCIBILITY_THRESHOLD = 0.85


def _check_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}, got {value}")


def completeness(features_described: int, total_features: int = 12) -> float:
    """M1: share of required attributes the model addressed.

    "Not assessable from this image" counts as addressed: the model did not
    skip the attribute, it correctly said it could not judge it.
    """
    _check_range("features_described", features_described, 0, total_features)
    return features_described / total_features * 100


def terminology(points: float, total_features: int = 12) -> float:
    """M2: terminology accuracy.

    Each attribute earns 2 points for a fully correct term, 1 for a partial
    match and 0 for a fundamental error, so the maximum is 2 x total_features.
    """
    max_points = 2 * total_features
    _check_range("term_points", points, 0, max_points)
    return points / max_points * 100


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard index |A ∩ B| / |A ∪ B| between two sets of identified features."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def consistency(runs: Sequence[Iterable[str]]) -> float:
    """M3: mean pairwise Jaccard index across repeated runs, as 0-100.

    `runs` holds the features the model identified in each repeated run of the
    same input. At least two runs are needed.
    """
    if len(runs) < 2:
        raise ValueError("consistency needs at least two runs")
    pairs = list(combinations(runs, 2))
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs) * 100


def interpretation(level: int) -> float:
    """M4: depth of interpretation on a 0-3 scale.

    0 - none, or a completely wrong attribution
    1 - partial interpretation, or a correct attribution without reasons
    2 - correct attribution supported by one or two features
    3 - full interpretation: several features, chronology and stated confidence
    """
    _check_range("interpretation_level", level, 0, 3)
    return level / 3 * 100


def _penalised(penalties: float) -> float:
    _check_range("penalties", penalties, 0, float("inf"))
    return max(0.0, 100.0 - PENALTY_STEP * penalties)


def calibration(penalties: float) -> float:
    """M5: confidence calibration.

    One penalty for every case of HIGH confidence on a wrong answer or LOW
    confidence on a right one. Half penalties are allowed for borderline cases.
    """
    return _penalised(penalties)


def no_hallucination(penalties: float) -> float:
    """M6: absence of hallucinations.

    One penalty for every invented fact: a decoration element that is not on
    the object, a fake reference, an unfounded parallel.
    """
    return _penalised(penalties)


def quality_score(m1: float, m2: float, m3: float, m4: float, m5: float, m6: float) -> float:
    """Weighted quality score Q on a 0-100 scale."""
    scores = {"m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6}
    for name, value in scores.items():
        _check_range(name, value, 0, 100)
    return sum(WEIGHTS[name] * value for name, value in scores.items())


def quality_level(q: float) -> str:
    """Quality level for a Q score: high, satisfactory, unsatisfactory or unusable."""
    for lower_bound, label in LEVELS:
        if q >= lower_bound:
            return label
    return LEVELS[-1][1]
