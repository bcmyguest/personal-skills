---
name: independent-review
description: Get a verdict on an artifact (ADR, plan, report, doc, analysis) from a reviewer that did not produce it, and loop fixes until it passes. Use when asked for an adversarial review, a second model to validate, or a review agent at the end. Not for code diffs (audit-agent-diff, Bugbot) or specs before implementation (wteam plan-review).
---

# Independent review

The author never reviews its own artifact. A separate reviewer reads it cold and returns a verdict.

## Rules

1. **Separate reviewer.** A fresh session, subagent, or different model with no access to the author's reasoning. Read-only.
2. **Explicit scope and bar.** State what to review and what to ignore ("errors only; accept minor style issues").
3. **Several lenses for decisions:** claims versus evidence, the strongest counter-reading, and scope.
4. **Honest record.** Say who reviewed what and when. A review that did not run is reported as not run, never as passed.

## Running it

- With `wteam`: copy the [`wteam/`](wteam) folder beside this file into the wteam config dir once (`cp -r wteam/. "${WTEAM_CONFIG_DIR:-$HOME/.config/wteam}/"`), then `wteam phase artifact-review --input <artifact>` runs three fresh-context reviewers (claims, critic, scope) and records the verdict.
- Elsewhere: start a separate agent with the artifact, the claim it makes, the acceptance criteria, and the verdict format below. Do not pass the author's conversation.

## Verdict format

```
VERDICT: APPROVE | REVISE
BLOCKERS: <item, evidence, required change>
WARNINGS: <item, evidence>
NITS: <item>
```

On REVISE, the author fixes the blockers and the reviewer re-reads only what changed. Record each round.

## Boundaries

Report. The reviewer does not edit the artifact.

## Next

- `resolve-design-decisions` (suggested): the verdict, when it leaves a choice open.
- `land-change` (suggested): the approved artifact.
