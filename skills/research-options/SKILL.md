---
name: research-options
description: Compare technologies, approaches, or data sources using primary evidence, including the option of building it yourself, and produce findings or a short ADR. Use for "what other options are there", "review this ADR", "is there recent literature on this", "does this push the frontier or retread it", or "what data can we get". Not for choosing between options the user must decide (resolve-design-decisions) or for measuring them (plan-experiment).
---

# Research options

Widen the candidate list, filter by hard constraints, and back each claim with a primary source or a spike.

## Gather first

- Decision drivers: cost, maintenance, hosting limits (offline, no cloud, no paid tools), licence, team skills.
- The current shortlist and where it came from.

## Method

1. **Challenge the shortlist.** Search for alternatives the shortlist missed, including "build it" and "do nothing".
2. **Filter on hard constraints first,** before comparing features.
3. **Use primary sources:** papers, source code, official docs, changelogs. Link the exact section.
4. **Spike the load-bearing claim** instead of trusting the docs: run the thing and record what happened.
5. **For data sources:** historical versus current, update cadence, access terms, coverage of the target region or population, and which phase of the work it serves.
6. **Pin deferred items** explicitly with what would reopen them.

## Deliverable

Findings or a short ADR: options, drivers, evidence per option with links, spike results, a recommendation or the decisions left for the user, and pinned items.

## Boundaries

Report. Spikes run in scratch space and are deleted or labelled as throwaway.

## Next

- `independent-review` (transition): the findings or ADR, for an adversarial read.
- `resolve-design-decisions` (transition): facts per option, when a choice is left for the user.
- `watchlist` (suggested): pinned items that wait on an upstream issue, PR, or release.
