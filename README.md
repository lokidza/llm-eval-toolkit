# LLM Eval Toolkit

Expert-grounded evaluation of large language model answers with six task-specific metrics, M1–M6, combined into one quality score Q.

Text-similarity metrics like BLEU, ROUGE or BERTScore tell you how close a model's wording is to a reference. They do not tell you whether the model identified the right thing, used the right term, stayed stable between runs, knew when it was unsure, or invented facts. This toolkit measures exactly those things, using annotations made by a domain expert.

It was built for, and tested on, a real study: Grok and Gemini describing Bronze Age ceramics (Catacomb culture, Ukraine) from photographs, in zero-shot and few-shot modes. The method itself is domain-agnostic: any task where an expert checks a model's answer against a fixed list of required attributes fits.

## The metrics

| Metric | What it measures | How it is scored | Weight |
| --- | --- | --- | --- |
| **M1** Completeness | Did the model address every required attribute? | attributes addressed / total × 100. "Not assessable" counts as addressed | 25% |
| **M2** Terminology | Did it use the correct expert terms? | 2 / 1 / 0 points per attribute (full / partial / wrong), normalised to 100 | 25% |
| **M3** Consistency | Does it give the same answer when asked again? | mean pairwise Jaccard index across repeated runs; J ≥ 0.85 = reproducible | 15% |
| **M4** Interpretation | Can it go beyond description to a reasoned conclusion? | 0–3 scale, normalised to 100 | 15% |
| **M5** Calibration | Does its stated confidence match its accuracy? | 100 − 30 per HIGH-confidence error or LOW-confidence correct answer | 10% |
| **M6** No hallucination | Does it invent facts, references or parallels? | 100 − 30 per invented fact | 10% |

```
Q = 0.25·M1 + 0.25·M2 + 0.15·M3 + 0.15·M4 + 0.10·M5 + 0.10·M6
```

| Q | Level |
| --- | --- |
| ≥ 80 | high |
| 60–79 | satisfactory |
| 40–59 | unsatisfactory |
| < 40 | unusable |

## Quick start

Python 3.10+, no third-party dependencies.

```bash
git clone https://github.com/lokidza/llm-eval-toolkit.git
cd llm-eval-toolkit

python -m llm_eval evaluate \
    --annotations data/catacomb_ceramics_annotations.csv \
    --consistency data/series_consistency.csv \
    --out reports
```

```
Grok     few_shot   n=18  mean Q= 77.4  (satisfactory)
Grok     zero_shot  n=19  mean Q= 75.1  (satisfactory)
Gemini   zero_shot  n=18  mean Q= 71.7  (satisfactory)
Gemini   few_shot   n=18  mean Q= 71.0  (satisfactory)
Reports written to reports/
```

This writes three files to `reports/`:

- `scores.csv`: M1–M6, Q and level for every response
- `summary.csv`: per model and mode, the mean of each metric, the Q range and the count per quality level
- `report.md`: a readable report with the ranking and the weakest responses ([example](reports/report.md))

To compute M3 directly from repeated runs:

```bash
python -m llm_eval consistency examples/repeat_runs.json
```

## Input format

**Annotations** (`--annotations`), one row per model response:

| Column | Meaning |
| --- | --- |
| `model`, `mode`, `item_id` | which model, which prompting mode, which test item |
| `features_described` | attributes the model addressed (0–12) |
| `term_points` | terminology points (0–24) |
| `interpretation_level` | depth of interpretation (0–3) |
| `calibration_penalties` | confidence errors (half penalties allowed) |
| `hallucination_penalties` | invented facts (half penalties allowed) |

**Consistency** (`--consistency`): the Jaccard index for each model and mode, measured on a subsample of items run three times each.

**Repeated runs** (`consistency` command): `{"item_id": [[features from run 1], [run 2], ...]}`.

## Results from the original study

73 scored responses, 25 vessels, Grok 4.20 Beta and Gemini 2.5 Pro.

- **Grok few-shot was the only setup that passed the reproducibility threshold** (J = 0.874). It also had the widest Q range (55.5–96.0): the typological reference helped most answers and badly misled a few.
- **Few-shot prompting raised interpretation** for Grok (M4: 22.8 → 53.7), but introduced a suggestion effect. The reference text mentions cord impressions often, so the model started "seeing" cord decoration on vessels that have none.
- **Zero-shot Grok was the best calibrated** (M5 = 87.4). It mostly refused to attribute rather than guess.
- **Gemini was much less stable** between runs (J = 0.40–0.44), which caps its Q regardless of answer quality.

The full methodology is described in the author's coursework (Taras Shevchenko National University of Kyiv, 2026). Every Q in this repository is recomputed from the raw annotations, and the tests check that it matches the original study protocol.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Project structure

```
llm_eval/
  metrics.py      M1–M6, Q and quality levels
  evaluate.py     loading, scoring, summaries, reports
  __main__.py     command-line interface
data/             annotations and consistency values from the original study
examples/         sample repeated-runs file for M3
reports/          generated example output
tests/            unit tests and a check against the original results
```

---

**Українською.** Інструмент для оцінювання відповідей великих мовних моделей за авторською системою з шести метрик (М1–М6) та зваженим балом якості Q. Розроблено в межах курсової роботи про застосування ШІ в археологічній типології кераміки (КНУ імені Тараса Шевченка, 2026) і перевірено на реальних даних тестування Grok і Gemini.

## Author

Vasilisa Loki Brik

## License

MIT
