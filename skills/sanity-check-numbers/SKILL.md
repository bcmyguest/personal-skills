---
name: sanity-check-numbers
description: Decide whether a metric, score, rate, or cost is real before showing, trusting, or publishing it. Use when a number appears or changes on a dashboard, benchmark table, report, or chart, or looks too good, too low, or suspiciously round. Not a statistics method (plan-experiment) and not for fixing a crash (diagnose-failure).
---

# Sanity-check numbers

Give a verdict on a number: plausible, wrong, or unproven, with its definition traced.

## Gather first

- The number, where it is shown, and the code or query that produces it.
- An independent expectation: order of magnitude from first principles, a second source, or a previous run.

## Method

1. **Compare to the expectation first.** A throughput, cost, or count that is off by an order of magnitude is wrong until explained.
2. **Trace the definition:** numerator, denominator, window, units, and which records are summed. For token counts, name which classes count (input, output, cache read, cache write).
3. **Look for fabrication and placeholders:** constants, seeded demo data, values the code never computes.
4. **Look for degenerate results:** a saturated score (24/24), two arms that should differ but match, leakage of the answer into the input, a grader that can be gamed.
5. **Check provenance:** rates, prices, and limits come from the system or its config, not from values hardcoded in the display code.
6. **Label it:** the window and unit appear next to the number ("output tokens/s, last 5 min").

## Deliverable

Verdict with the traced definition, the expectation it was compared against, and the evidence. For "wrong": the step where the definition breaks. For "unproven": what would settle it.

## Boundaries

Report. Edit only to correct the definition or its label.

## Next

- `diagnose-failure` (transition): the wrong number and its traced definition, when the cause is a bug.
- `write-evidence-report` (transition): verified numbers, ready to publish.
