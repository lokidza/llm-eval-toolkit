import csv
from pathlib import Path

import pytest

from llm_eval import metrics
from llm_eval.evaluate import load_consistency, run, score_response

DATA = Path(__file__).resolve().parent.parent / "data"


def test_completeness_counts_not_assessable_as_described():
    assert metrics.completeness(12) == 100
    assert metrics.completeness(9) == 75


def test_terminology_scale():
    assert metrics.terminology(24) == 100
    assert metrics.terminology(12) == 50


def test_jaccard():
    assert metrics.jaccard({"a", "b"}, {"a", "b"}) == 1
    assert metrics.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert metrics.jaccard(set(), set()) == 1


def test_consistency_is_mean_pairwise_jaccard():
    runs = [{"a", "b"}, {"a", "b"}, {"a", "c"}]
    expected = (1 + 1 / 3 + 1 / 3) / 3 * 100
    assert metrics.consistency(runs) == pytest.approx(expected)


def test_consistency_needs_two_runs():
    with pytest.raises(ValueError):
        metrics.consistency([{"a"}])


def test_penalties_floor_at_zero():
    assert metrics.calibration(0) == 100
    assert metrics.calibration(1) == 70
    assert metrics.calibration(0.5) == 85
    assert metrics.no_hallucination(5) == 0


def test_quality_score_weights_sum_to_one():
    assert sum(metrics.WEIGHTS.values()) == pytest.approx(1.0)
    assert metrics.quality_score(100, 100, 100, 100, 100, 100) == pytest.approx(100)


@pytest.mark.parametrize("q, level", [(80, "high"), (79.9, "satisfactory"), (40, "unsatisfactory"), (39.9, "unusable")])
def test_quality_levels(q, level):
    assert metrics.quality_level(q) == level


def test_out_of_range_input_is_rejected():
    with pytest.raises(ValueError):
        metrics.interpretation(4)
    with pytest.raises(ValueError):
        metrics.terminology(25)


def test_scores_reproduce_the_original_study():
    """Every Q recomputed from the raw annotations matches the value in the original protocol."""
    consistency = load_consistency(DATA / "series_consistency.csv")
    with open(DATA / "catacomb_ceramics_annotations.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scored = score_response(row, consistency[(row["model"], row["mode"])])
            assert scored.q == pytest.approx(float(row["reference_q"]), abs=0.15), row["item_id"]


def test_run_writes_reports(tmp_path):
    summaries = run(DATA / "catacomb_ceramics_annotations.csv", DATA / "series_consistency.csv", tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"scores.csv", "summary.csv", "report.md"}
    assert [s for s in summaries if s.reproducible][0].model == "Grok"
