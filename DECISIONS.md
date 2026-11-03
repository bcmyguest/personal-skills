# Decisions: method skills and skill chains

Why the method skills (`diagnose-failure` … `write-evidence-report`) and the skill
chains look the way they do. Settled with the owner in question rounds, November 2026.

## Evidence

The skills were recovered from the owner's own requests to coding agents (Claude Code,
Codex, Cursor), their instruction files, plan documents, and existing skills, sampled
across about ten projects. Delegated prompts written by agents, host-injected text, and
benchmark sandboxes were excluded. Session references and project details stay in a
private evidence directory outside this repo. Each rule in a skill was seen in that
evidence or stated in the owner's instructions.

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
| Independent review | Thin host-neutral skill; its wteam phase (`artifact-review`) ships as phase and role files in `wteam/phases/` and `wteam/roles/`, not as wteam code |
| Agent-pipeline tuning | Folded into wteam's `refine` skill, generalised beyond wteam runs |
| Chain source | The chains are wteam teams: hand-written `wteam/teams/<name>/team.dot`, imported by `scripts/install-wteam.sh` (copy, overwriting). The `chains/skills.dot` registry is gone, with its observed/inferred tags; hand-offs outside teams come from each skill's `## Next` |
| Which chains are teams | fix-defect, port, ui, report-review, experiment, decide-and-build. `brief-agent` (handoff) has one step, so no team |
| Multi-step teams | One worker node per step (wteam steps); the ticket commits once after the last step. A worker step may set `continue=true` to keep the previous step's session while it fits; reviewers always run as separate agents |
| Decisions inside a team | The `decide` step asks its round by `wteam block`; answers return as a `wteam requeue --note`, which resumes the same step |
| Team node prompts | wteam appends a node's `prompt=` to the agent's role prompt, keeping the verdict protocol |

## Tests

Existing tests are protected. A test an agent added in the current change may be removed
when it tests the wrong thing (`audit-agent-diff`, rule 4).

## Not done

No paid or comparative model evaluation was run on these skills. The routing examples in
[chains/routing.md](chains/routing.md) are for review by hand; nothing measured how often a
host picks the right skill.
