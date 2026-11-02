# Routing examples

Requests that should activate each method skill, and near-misses that should go elsewhere.
Use them to check a skill's `description` by hand: read the request, decide which skill a
host would pick from the descriptions alone, and compare. `scripts/test_chains.py` checks
that every method skill has a section here and that every near-miss target exists.

## diagnose-failure

Routes here:
- "The feed sync stopped updating yesterday. Find out why and fix it."
- "The stack crashes the host every few hours; logs are on the other machine."

Near-misses:
- "Is 11 tokens/s plausible for this model?" → `sanity-check-numbers`
- "The agents in the run keep looping on the same ticket." → `refine`

## sanity-check-numbers

Routes here:
- "The dashboard says the run cost $0.40 for 178M tokens. Is that right?"
- "Both arms scored 24/24. Is the benchmark saturated?"

Near-misses:
- "How many seeds do we need for a significant difference?" → `plan-experiment`
- "The cost panel shows nothing since the last deploy." → `diagnose-failure`

## audit-agent-diff

Routes here:
- "Before I commit, check what the agent changed."
- "Did the agent remove anything it shouldn't have?"

Near-misses:
- "Find bugs in this PR." → `bugbot`
- "Commit this and open a PR." → `land-change`

## land-change

Routes here:
- "Commit and push, then open a PR with a screenshot."
- "Merge it once the checks are green."

Near-misses:
- "Take the auth changes from the old branch." → `port-changes`
- "Review this diff first." → `audit-agent-diff`

## port-changes

Routes here:
- "Find what's in this tarball that isn't on the branch."
- "Bring her PR's queue changes into ours, but only those two files."

Near-misses:
- "Push this branch." → `land-change`
- "Why does the merged branch fail to start?" → `diagnose-failure`

## match-ui-to-reference

Routes here:
- "Make the settings page match this Figma frame."
- "The header doesn't match the screenshot. Fix the spacing and colours."

Near-misses:
- "Should the dashboard show cost or tokens first?" → `resolve-design-decisions`
- "The page shows a blank screen after login." → `diagnose-failure`

## resolve-design-decisions

Routes here:
- "Before building the sync service, settle the storage and API choices with me."
- "This plan has open forks. Ask me what you need."

Near-misses:
- "What vector stores could we use offline?" → `research-options`
- "Write the spec for the decided design." → `to-spec`

## research-options

Routes here:
- "Review this ADR. Did we miss options?"
- "What public data on flood response exists for this region?"

Near-misses:
- "Pick between the two for me now." → `resolve-design-decisions`
- "Benchmark both on our tasks." → `plan-experiment`

## independent-review

Routes here:
- "Have a different model review the results write-up adversarially."
- "Add a review agent at the end that checks the report."

Near-misses:
- "Review this code diff for bugs." → `bugbot`
- "Check the plan's tickets before we run." → `wteam`

## brief-agent

Routes here:
- "Give me a prompt for the next agent to finish this."
- "Context is almost full. Write the handoff."

Near-misses:
- "Run these tickets with a worker and reviewer." → `wteam`
- "Summarise this conversation for me." → `none` (a plain summary, no skill)

## plan-experiment

Routes here:
- "Prove the trim actually saves tokens without hurting scores."
- "Design a comparison of the three agent setups."

Near-misses:
- "The plan is approved. Start the run." → `run-experiment`
- "Is this throughput number real?" → `sanity-check-numbers`

## run-experiment

Routes here:
- "Plan approved. Smoke one trial, then run the rest under the $10 cap."
- "Resume the benchmark where it stopped."

Near-misses:
- "Which arms should we compare?" → `plan-experiment`
- "Write up the results on the site." → `write-evidence-report`

## write-evidence-report

Routes here:
- "Publish the benchmark results to the docs if they make sense."
- "Update the provider support matrix with what we verified."

Near-misses:
- "Are these scores plausible?" → `sanity-check-numbers`
- "Commit the docs and open a PR." → `land-change`
