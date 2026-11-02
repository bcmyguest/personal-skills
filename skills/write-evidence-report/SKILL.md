---
name: write-evidence-report
description: Publish results or capability claims that say exactly what the evidence supports. Use for "write it up", "update the docs, site, or README with the results", "publish if they make sense", or a compatibility or support matrix. Not for running the experiment (run-experiment) or checking a single number (sanity-check-numbers).
---

# Write evidence report

Write up results so every claim traces to a run, and every limit is visible.

## Gather first

- Result rows with intervals, the plan they came from, and the exact config per arm.
- For capability claims: the evidence level per claim (verified live, tests only, partial, untested).

## Method

1. **One source:** chart, text, and config come from the same run. Regenerate charts from the data file; do not edit numbers by hand.
2. **Name things:** the benchmark, the model, and every abbreviation, defined in plain words on first use.
3. **Report what matters** to the decision (quality, cost, time, cache use), not the metric that looks best.
4. **Show uncertainty:** n, intervals, exclusions, and caveats next to the claim they qualify.
5. **Null and negative results** are reported as such. No fabricated, padded, or placeholder data; sparse evidence is called sparse.
6. **Evidence levels** per capability claim, so "supported" means verified.
7. **Docs first:** after a run, update the write-up before starting new work.

## Deliverable

A doc, site, or README section with method, n, intervals, caveats, a chart that matches the text, and evidence levels per claim.

## Boundaries

Edit docs and data-driven charts. Publishing goes through land-change after review.

## Next

- `independent-review` (transition): the draft.
