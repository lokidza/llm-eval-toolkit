# LLM evaluation report

73 scored responses across 4 model / mode series.

Q = 0.25·M1 + 0.25·M2 + 0.15·M3 + 0.15·M4 + 0.10·M5 + 0.10·M6. Levels: high ≥ 80, satisfactory 60–79, unsatisfactory 40–59, unusable < 40.

## Ranking by mean Q

| Model | Mode | n | Mean Q | Min Q | Max Q | High | Satisf. | Unsatisf. | Unusable | Reproducible (J ≥ 0.85) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grok | few_shot | 18 | 77.4 | 55.5 | 96.0 | 7 | 8 | 3 | 0 | yes |
| Grok | zero_shot | 19 | 75.1 | 64.1 | 82.4 | 5 | 14 | 0 | 0 | no |
| Gemini | zero_shot | 18 | 71.7 | 39.3 | 84.9 | 4 | 11 | 2 | 1 | no |
| Gemini | few_shot | 18 | 71.0 | 41.0 | 81.3 | 3 | 13 | 2 | 0 | no |

## Mean score per metric

| Model | Mode | M1 Completeness | M2 Terminology | M3 Consistency | M4 Interpretation | M5 Calibration | M6 No hallucination |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Grok | few_shot | 100.0 | 64.1 | 87.4 | 53.7 | 71.7 | 80.0 |
| Grok | zero_shot | 100.0 | 69.7 | 77.0 | 22.8 | 87.4 | 89.7 |
| Gemini | zero_shot | 100.0 | 70.1 | 40.1 | 48.1 | 76.1 | 83.3 |
| Gemini | few_shot | 100.0 | 63.7 | 44.2 | 53.7 | 71.1 | 83.3 |

## Five weakest responses

| Model | Mode | Item | Q | Level | Lowest metric |
| --- | --- | --- | --- | --- | --- |
| Gemini | zero_shot | vessel_01 | 39.3 | unusable | M4 = 0.0 |
| Gemini | few_shot | vessel_01 | 41.0 | unsatisfactory | M4 = 0.0 |
| Gemini | zero_shot | vessel_22 | 50.5 | unsatisfactory | M4 = 0.0 |
| Grok | few_shot | vessel_20 | 55.5 | unsatisfactory | M4 = 0.0 |
| Gemini | few_shot | vessel_02 | 57.2 | unsatisfactory | M4 = 0.0 |
