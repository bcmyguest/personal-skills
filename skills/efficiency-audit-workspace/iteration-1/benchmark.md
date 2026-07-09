# Skill Benchmark: efficiency-audit (iteration 1)

Assertions passed (behavioral + plant-detection). Baseline = same prompt, no skill.
Config runs are single-sample (non-deterministic; see caveats).

| Eval | Config | Pass | Time | Tokens |
|------|--------|------|------|--------|
| eval-0 (default target) | with_skill | 6/6 | 392s | 64k |
| eval-0 (default target) | baseline | 4/6 | 155s | 51k |
| eval-1 (single worst)   | with_skill | 4/4 | 503s | 70k |
| eval-1 (single worst)   | baseline | 4/4 | 79s | 44k |

**Aggregate:** with_skill 10/10 (100%) · baseline 8/10 (80%).
Baseline lost cap (8 findings) AND category discipline (flagged SQLi + shell injection inside an efficiency audit). Skill held to 3, stayed in-category.
