# Skill Benchmark: security-audit (iteration 1)

Assertions passed (behavioral + plant-detection). Baseline = same prompt, no skill.
Config runs are single-sample (non-deterministic; see caveats).

| Eval | Config | Pass | Time | Tokens |
|------|--------|------|------|--------|
| eval-0 (default target) | with_skill | 6/6 | n/a* | n/a* |
| eval-0 (default target) | baseline | 4/6 | 671s | 57k |
| eval-1 (single worst)   | with_skill | 4/4 | 376s | 64k |
| eval-1 (single worst)   | baseline | 4/4 | 76s | 44k |

*timing lost to a process restart; report completed on disk.

**Aggregate:** with_skill 10/10 (100%) · baseline 8/10 (80%).
Baseline lost both cap assertions on the open-ended eval (returned 8 findings, padded with a Low-Medium nit). Skill held to 3, deferred extras explicitly.
