"""Score annotated model responses and summarise them per model and prompting mode."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from . import metrics

ANNOTATION_COLUMNS = [
    "model",
    "mode",
    "item_id",
    "features_described",
    "term_points",
    "interpretation_level",
    "calibration_penalties",
    "hallucination_penalties",
]


@dataclass
class ScoredResponse:
    model: str
    mode: str
    item_id: str
    m1: float
    m2: float
    m3: float
    m4: float
    m5: float
    m6: float
    q: float
    level: str


@dataclass
class SeriesSummary:
    model: str
    mode: str
    n: int
    m1: float
    m2: float
    m3: float
    m4: float
    m5: float
    m6: float
    q_mean: float
    q_min: float
    q_max: float
    high: int
    satisfactory: int
    unsatisfactory: int
    unusable: int
    reproducible: bool


def load_consistency(path: str | Path) -> dict[tuple[str, str], float]:
    """Read run-to-run Jaccard values per (model, mode) from a CSV file."""
    with open(path, newline="", encoding="utf-8") as f:
        return {(row["model"], row["mode"]): float(row["jaccard"]) for row in csv.DictReader(f)}


def load_annotations(path: str | Path) -> list[dict[str, str]]:
    """Read the expert annotation sheet and check that all columns are present."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ANNOTATION_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {', '.join(missing)}")
        return list(reader)


def score_response(row: dict[str, str], jaccard: float, total_features: int = 12) -> ScoredResponse:
    """Turn one annotated response into M1-M6, Q and a quality level."""
    m1 = metrics.completeness(int(row["features_described"]), total_features)
    m2 = metrics.terminology(float(row["term_points"]), total_features)
    m3 = jaccard * 100
    m4 = metrics.interpretation(int(row["interpretation_level"]))
    m5 = metrics.calibration(float(row["calibration_penalties"]))
    m6 = metrics.no_hallucination(float(row["hallucination_penalties"]))
    q = metrics.quality_score(m1, m2, m3, m4, m5, m6)
    return ScoredResponse(
        row["model"], row["mode"], row["item_id"], m1, m2, m3, m4, m5, m6, q, metrics.quality_level(q)
    )


def score_all(rows: Iterable[dict[str, str]], consistency: dict[tuple[str, str], float]) -> list[ScoredResponse]:
    scored = []
    for row in rows:
        key = (row["model"], row["mode"])
        if key not in consistency:
            raise ValueError(f"no consistency value for {key[0]} / {key[1]}")
        scored.append(score_response(row, consistency[key]))
    return scored


def summarise(scored: list[ScoredResponse]) -> list[SeriesSummary]:
    """Average each metric per (model, mode) and count responses per quality level."""
    groups: dict[tuple[str, str], list[ScoredResponse]] = {}
    for s in scored:
        groups.setdefault((s.model, s.mode), []).append(s)

    summaries = []
    for (model, mode), items in groups.items():
        qs = [s.q for s in items]
        levels = [s.level for s in items]
        m3 = mean(s.m3 for s in items)
        summaries.append(
            SeriesSummary(
                model=model,
                mode=mode,
                n=len(items),
                m1=mean(s.m1 for s in items),
                m2=mean(s.m2 for s in items),
                m3=m3,
                m4=mean(s.m4 for s in items),
                m5=mean(s.m5 for s in items),
                m6=mean(s.m6 for s in items),
                q_mean=mean(qs),
                q_min=min(qs),
                q_max=max(qs),
                high=levels.count("high"),
                satisfactory=levels.count("satisfactory"),
                unsatisfactory=levels.count("unsatisfactory"),
                unusable=levels.count("unusable"),
                reproducible=m3 / 100 >= metrics.REPRODUCIBILITY_THRESHOLD,
            )
        )
    return sorted(summaries, key=lambda s: s.q_mean, reverse=True)


def _round(record: dict, digits: int = 1) -> dict:
    return {k: round(v, digits) if isinstance(v, float) else v for k, v in record.items()}


def write_csv(records: list, path: Path) -> None:
    rows = [_round(asdict(r)) for r in records]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(summaries: list[SeriesSummary], scored: list[ScoredResponse]) -> str:
    """A readable report: ranking, per-metric means and the weakest responses."""
    lines = [
        "# LLM evaluation report",
        "",
        f"{len(scored)} scored responses across {len(summaries)} model / mode series.",
        "",
        "Q = 0.25·M1 + 0.25·M2 + 0.15·M3 + 0.15·M4 + 0.10·M5 + 0.10·M6. "
        "Levels: high ≥ 80, satisfactory 60–79, unsatisfactory 40–59, unusable < 40.",
        "",
        "## Ranking by mean Q",
        "",
        "| Model | Mode | n | Mean Q | Min Q | Max Q | High | Satisf. | Unsatisf. | Unusable | Reproducible (J ≥ 0.85) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.model} | {s.mode} | {s.n} | {s.q_mean:.1f} | {s.q_min:.1f} | {s.q_max:.1f} | "
            f"{s.high} | {s.satisfactory} | {s.unsatisfactory} | {s.unusable} | {'yes' if s.reproducible else 'no'} |"
        )

    lines += [
        "",
        "## Mean score per metric",
        "",
        "| Model | Mode | M1 Completeness | M2 Terminology | M3 Consistency | M4 Interpretation | M5 Calibration | M6 No hallucination |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.model} | {s.mode} | {s.m1:.1f} | {s.m2:.1f} | {s.m3:.1f} | {s.m4:.1f} | {s.m5:.1f} | {s.m6:.1f} |"
        )

    weakest = sorted(scored, key=lambda s: s.q)[:5]
    lines += [
        "",
        "## Five weakest responses",
        "",
        "| Model | Mode | Item | Q | Level | Lowest metric |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in weakest:
        values = {"M1": s.m1, "M2": s.m2, "M3": s.m3, "M4": s.m4, "M5": s.m5, "M6": s.m6}
        lowest = min(values, key=values.get)
        lines.append(
            f"| {s.model} | {s.mode} | {s.item_id} | {s.q:.1f} | {s.level} | {lowest} = {values[lowest]:.1f} |"
        )
    return "\n".join(lines) + "\n"


def run(annotations: str | Path, consistency: str | Path, out_dir: str | Path) -> list[SeriesSummary]:
    """Score every response, then write per-response and per-series CSVs and a Markdown report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scored = score_all(load_annotations(annotations), load_consistency(consistency))
    summaries = summarise(scored)
    write_csv(scored, out / "scores.csv")
    write_csv(summaries, out / "summary.csv")
    (out / "report.md").write_text(markdown_report(summaries, scored), encoding="utf-8")
    return summaries


def consistency_from_runs(path: str | Path) -> dict[str, float]:
    """Compute M3 per item from a JSON file of repeated runs: {item_id: [[features], ...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item: metrics.consistency(runs) for item, runs in data.items()}
