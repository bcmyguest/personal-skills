---
name: resolve-design-decisions
description: Settle costly-to-reverse design choices by asking the user in dependency-ordered rounds, then proceed without re-asking. Use for architecture, data model, API shape, dependency, persistence, or concurrency choices, or a plan with open forks. Not for facts you can look up (research-options) or questions with an obvious answer.
---

# Resolve design decisions

Map the open choices as a tree, ask every unblocked one at once, record the answers, and stop asking.

## Gather first

- The goal and hard constraints (hosting, cost, offline, licensing).
- Facts you can find yourself: read the code, docs, and prior decision records before asking anything.

## Method

1. **Separate facts from decisions.** Look facts up. Ask only for decisions.
2. **Build the tree.** A decision whose options depend on another open one waits for a later round.
3. **Ask a round:** every unblocked decision in one message, numbered, each naming the constraint at stake and two or three lettered options with their costs. Short answers like "1a 2c" must be enough.
4. **Picture before words** when the choice concerns flow, state, or data shape: a sequence diagram, table, or small example.
5. **Give a preference** once the user has answered, or when asked.
6. **Stop.** Once decided, proceed. Do not re-confirm settled points or end a turn on a restated plan.

## Deliverable

A decision record: question, options with costs, answer, and the round it was settled in. Then the plan that follows from it.

## Boundaries

Plan only. Implementation waits for the decisions it depends on, and nothing else.

## Next

- `independent-review` (suggested): the decision record, for a fresh-context read.
- `to-spec` (suggested): the decision record, when the work needs a written spec.

Downstream of that, `to-spec` hands a spec to `to-tickets`, whose tickets run in `wteam` or, for one-off work, `implement`.
