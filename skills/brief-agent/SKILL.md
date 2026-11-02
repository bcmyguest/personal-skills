---
name: brief-agent
description: Write a self-contained brief so another agent, model, or session can continue the work or take a bounded task. Use when asked for "a prompt or command for the next agent", when context is running out, when an agent is going in circles, when switching hosts, or when delegating a task. Not for runs managed by wteam, which hands off on its own.
---

# Brief agent

Write the brief the next agent needs to act without the current conversation.

## Gather first

- The user's original asks, from the conversation or transcript, verbatim.
- Current state: branch, commits, running processes, files touched.
- What is verified versus only claimed.

## Brief

1. **Goal** in one sentence.
2. **Original asks,** quoted verbatim.
3. **Done versus claimed:** what was verified, with the evidence, and what was only reported.
4. **Verified facts** with `file:line` anchors for code.
5. **Ruled out:** hypotheses and approaches already disproven, and why. Drop facts that turned out false.
6. **Next action:** the first concrete step.
7. **Constraints:** files not to touch, approvals still needed, budgets.
8. **Stop condition:** when the task is done ("commit and report the hash"; "do not stop after stating a plan").
9. **Report back:** what the next agent must return.

## Boundaries

Write the brief. Launching the next agent is a separate action the user or host takes.

## Next

- `next-agent` (suggested): the brief.
