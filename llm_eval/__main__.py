"""Command-line interface.

    python -m llm_eval evaluate --annotations data/catacomb_ceramics_annotations.csv \
        --consistency data/series_consistency.csv --out reports
    python -m llm_eval consistency examples/repeat_runs.json
"""

from __future__ import annotations

import argparse

from . import metrics
from .evaluate import consistency_from_runs, run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm_eval", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="score annotated responses and write reports")
    ev.add_argument("--annotations", required=True, help="CSV with one annotated response per row")
    ev.add_argument("--consistency", required=True, help="CSV with run-to-run Jaccard per model and mode")
    ev.add_argument("--out", default="reports", help="folder for scores.csv, summary.csv and report.md")

    co = sub.add_parser("consistency", help="compute M3 from repeated runs stored as JSON")
    co.add_argument("runs", help='JSON: {"item_id": [["feature", ...], ...]}')

    args = parser.parse_args(argv)

    if args.command == "evaluate":
        summaries = run(args.annotations, args.consistency, args.out)
        for s in summaries:
            print(f"{s.model:<8} {s.mode:<10} n={s.n:<3} mean Q={s.q_mean:5.1f}  ({metrics.quality_level(s.q_mean)})")
        print(f"Reports written to {args.out}/")
    else:
        for item, m3 in consistency_from_runs(args.runs).items():
            verdict = "reproducible" if m3 / 100 >= metrics.REPRODUCIBILITY_THRESHOLD else "below 0.85"
            print(f"{item}: M3 = {m3:.1f} ({verdict})")


if __name__ == "__main__":
    main()
