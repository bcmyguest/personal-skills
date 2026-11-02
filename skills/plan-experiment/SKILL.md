---
name: plan-experiment
description: Design a comparison whose result can change a decision, before spending anything on it. Use for "run the benchmark", "prove it", "make the evidence stronger or disprove it", or comparing agent setups, models, or configurations. Not for executing an approved plan (run-experiment) or checking one number (sanity-check-numbers).
---

# Plan experiment

Pre-register the comparison: what would confirm the claim, what would disprove it, and what it may cost.

## Gather first

- The claim and the decision it feeds.
- Candidate arms, tasks, harness, and the budget.

## Plan contents

1. **Hypotheses** with a disproval rule each.
2. **Arms** with exact environment and config, including a baseline arm.
3. **Tasks and seeds:** tuning seeds separate from held-out scoring seeds.
4. **Metrics:** quality plus cost, tokens (including cache read and write), and wall time. Say how token use is penalised.
5. **Statistics:** sample size, interval method (for example, bootstrap CIs), and what counts as a difference.
6. **Exclusion rules,** fixed before results are seen.
7. **Phase 0 check:** one trial per arm proving the arms actually differ (the feature under test fired).
8. **Budget:** spend cap per arm and in total, and abort triggers.
9. **Null-result policy:** a null result is published, not rerun until it changes.
10. **Reuse:** when a re-run follows a change, re-measure only what the change affects.

## Boundaries

Plan only; no paid runs. The plan goes to the user for approval of the spend.

## Next

- `spend-approval` (transition): the pre-registered plan.
- `run-experiment` (prerequisite): the approved plan.
