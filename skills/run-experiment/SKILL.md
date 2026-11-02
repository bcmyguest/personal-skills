---
name: run-experiment
description: Execute an approved experiment plan, keep it honest while it runs, and stop the line on design faults. Use when a plan is approved and the user says to start the run, or to resume or monitor one. Not for designing the comparison (plan-experiment) or writing it up (write-evidence-report).
---

# Run experiment

Run exactly the approved plan, watch it, and leave it resumable.

## Gather first

- The approved plan, its budget caps and abort triggers.
- The harness, its resume mechanism, and where result rows are written.

## Method

1. **Smoke one trial** per arm and check the Phase 0 condition before queuing the rest.
2. **Wait** when the user asked to approve a paid run; do not start it on your own.
3. **Monitor** spend, error rate, and stalls against the plan's triggers. A trial that runs far past its expected time is stuck until shown otherwise.
4. **Stop a broken queue.** When a design fault appears (identical arms, leaked answers, crashing harness), stop the remaining trials, fix, and record the restart. Do not let a known-bad queue burn budget.
5. **Commit along the way:** results, logs, and config, so a crash loses little.
6. **Leave a resume command** that continues without redoing finished trials.

## Deliverable

Result rows for every trial, exclusions applied by the plan's rules, a log of reruns and stops with reasons, spend, and the resume command.

## Boundaries

Spend only within the approved plan. Changing arms, tasks, or budget goes back to plan-experiment.

## Next

- `sanity-check-numbers` (transition): the result rows.
- `refine` (suggested): a pipeline arm that stalls or loops.
