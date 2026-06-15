---
name: pickup
description: Resume work from the most recent troubleshooting/debugging handoff — read it, state the next goal in one sentence, and confirm understanding with the user before doing anything. Use whenever the user invokes /pickup, says "pick up the handoff", "resume where we left off", "continue the last session", "load the latest handoff", or starts a session that clearly continues prior work captured in a handoff. The counterpart to the `handoff` skill.
---

# Handoff pickup

Read a handoff written by the `handoff` skill and resume the investigation **without
re-deriving anything** — but also without blindly trusting it. A handoff is a snapshot
from an earlier moment; some "confirmed facts" may have gone stale. Your job is to load
it, restate the goal, re-verify the cheap load-bearing facts, and **confirm with the user
before acting**. Do not start changing things on the strength of the doc alone.

## Process

1. **Find the handoff.** Handoffs live in `~/.claude/handoffs/<topic-slug>-<YYYY-MM-DD>.md`.
   - If the user named a topic, match it against the slugs.
   - Otherwise pick the most recently modified: `ls -t ~/.claude/handoffs/*.md 2>/dev/null | head`.
   - If several are recent and the topic is ambiguous, list the candidates (slug + date)
     and ask which one — do not guess.
   - If the directory is empty or missing, say so plainly and stop; there is nothing to
     pick up.
2. **Read it in full** with the Read tool — the whole doc, not the first screen. Pay
   special attention to *Next steps*, *Constraints*, *Ruled out*, and *Blockers*.
3. **Re-verify the cheap facts.** For each *Confirmed fact* that is fast to re-check
   (a version, a `curl`, an `ls`, a service status) and that the next action depends on,
   re-run the check now. The doc records what *was* true; you act on what *is* true. Note
   any fact that has drifted.
4. **Restate, in one sentence**, the single next goal — the immediate end state you are
   about to work toward, drawn from *Goal* + the first *Next step*. Not a summary of the
   whole doc; the one thing you're about to do.
5. **Confirm before proceeding.** Use `AskUserQuestion` to check understanding *before*
   touching anything. Ask about:
   - whether the restated next goal is still the right one (priorities may have shifted
     since the handoff was written),
   - any genuine ambiguity in *Next steps* or *Constraints* that changes what you'd do,
   - any fact that re-verification showed has drifted (surface it, don't bury it).
   Keep it to the questions whose answers actually change your next move — if the handoff
   is unambiguous and the facts still hold, ask the single confirming question and move on.
6. **Only then proceed** — and honor every rule in the handoff's *Constraints* section
   (e.g. sudo / system-state changes are user-run; services that must not be restarted).

## Output before you ask

Lead your reply with, in this order:

- **Picked up:** `<path>` (`<topic>`, written `<YYYY-MM-DD>`)
- **Next goal:** one sentence.
- **Still true / drifted:** one line if any re-verified fact changed; omit if all hold.
- **Watch out:** the constraints and open blockers that bound the work, condensed.

Then ask your clarifying question(s). Do not edit files, run state-changing commands, or
start the next step until the user has answered.
