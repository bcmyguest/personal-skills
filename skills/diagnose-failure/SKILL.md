---
name: diagnose-failure
description: Find the actual cause of a failure in a running system and fix that cause, not a symptom. Use when something is broken, stuck, crashing, no longer updating, or a run failed, and the job is to explain why and fix it. Not for judging whether a number is plausible (sanity-check-numbers) or tuning how an agent pipeline behaves (refine).
---

# Diagnose failure

Explain a failure with evidence, fix the cause, and show the original symptom is gone.

## Gather first

1. The symptom in the user's words and the exact failing case (command, URL, run id, timestamp).
2. The user's own diagnosis, if they gave one. Pursue it first; only propose alternatives after it is ruled out with evidence.
3. The system of record: database rows, event logs, container state, CI logs, `git log`. Not the UI and not an agent's self-report.
4. What changed since it last worked: diff against the last good commit, branch, release, or archive.

## Method

- State one hypothesis, the observation that would confirm it, and the observation that would rule it out. Run the one or two commands that decide it.
- Check that a step *succeeded*, not that it was invoked: a tool call that ran and errored looks like a tool call.
- Stop a probe loop that is not converging. Report what you have before starting another chain of commands.
- Separate fix from mitigation. Reject changes that hide the cause: polling instead of a missing event, retries around a crash, extra steps stuffed into a startup script.
- Intermittent or remote-only failures: add instrumentation and correlate logs across machines instead of insisting on a local reproduction. Add protective limits (memory, timeouts) when the failure can take a host down.

## Strict path

Use when a reliable local reproduction exists and the fix must be proven:

1. Reproduce the exact failure and record the command and output.
2. Change one thing per attempt.
3. Re-run the original failing case after the fix, not a neighbouring one.

## Deliverable

- Cause, with the evidence that shows it (log line, query result, diff).
- The fix, or the mitigation clearly labelled as one.
- The live check that shows the symptom gone, or a plain statement that it was not re-checked and why.

## Boundaries

Edit only what the fix needs and what the request covers. Ask before destructive steps on shared systems (dropping data, resetting a stack others use).

## Next

- `audit-agent-diff` (transition): the fix and its live check, before it lands.
- `refine` (alternative): when the failure is an agent pipeline behaving badly rather than a crash.
