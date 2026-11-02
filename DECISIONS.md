# Decisions: method skills and skill chains

Why the method skills (`diagnose-failure` … `write-evidence-report`) and the chain
registry look the way they do. Settled with the owner in question rounds, November 2026.

## Evidence

The skills were recovered from the owner's own requests to coding agents (Claude Code,
Codex, Cursor), their instruction files, plan documents, and existing skills, sampled
across about ten projects. Delegated prompts written by agents, host-injected text, and
benchmark sandboxes were excluded. Session references and project details stay in a
private evidence directory outside this repo. Each rule in a skill was seen in that
evidence or stated in the owner's instructions; relationships in the registry are marked
`observed` or `inferred` accordingly.

Not covered: the agents' side of the transcripts in depth, PR review threads, and work on
other machines or in ticket systems. Treat the method skills as a first cut on that basis.

## Choices

| Question | Decision |
| --- | --- |
| Separate fast and strict debugging skills | Fold both into `diagnose-failure`; `fast-hypothesis` and `reproduce-bug` are retired |
| Session handoff | `brief-agent` replaces baton's `handoff`/`pickup` for this library |
| Experiments | Split into `plan-experiment` (no spend) and `run-experiment` (spend within the plan) |
| Diff audit versus landing | Two skills: `audit-agent-diff` reports, `land-change` performs external actions |
| When to ask the user | Only before steps that are costly to reverse, paid, destructive, or scope-changing |
| Style and personal preferences | Live in bcmyguest/claude-memory, with `writing-with-clarity`; not in method skills |
| Slice orchestration | No skill: bcmyguest/wteam does it and appears as an external node |
| Independent review | Thin host-neutral skill; its wteam phase (`artifact-review`) ships as phase and role files in `skills/independent-review/wteam/`, not as wteam code |
| Agent-pipeline tuning | Folded into wteam's `refine` skill, generalised beyond wteam runs |
| Registry format | DOT in wteam's subset, so wteam parses it; four workflows generate runnable teams |
| Team node prompts | wteam appends a node's `prompt=` to the agent's role prompt, keeping the verdict protocol |

## Tests

Existing tests are protected. A test an agent added in the current change may be removed
when it tests the wrong thing (`audit-agent-diff`, rule 4).

## Not done

No paid or comparative model evaluation was run on these skills. The routing examples in
[chains/routing.md](chains/routing.md) are for review by hand; nothing measured how often a
host picks the right skill.
