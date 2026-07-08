---
name: reproduce-bug
description: Debug any bug using strict reproduce-verify discipline
---

# Reproduce & Verify Bug Diagnosis

Your task: **diagnose and fix a bug** using strict reproduce-verify discipline.

## Non-Negotiable Steps

1. **Reproduce the exact failure**: Run the failing command or scenario and capture the full output
   - Do not skip or paraphrase the error
   - Show your exact command and the complete output
   - Include stderr and any stack traces

2. **Form ONE hypothesis at a time**: Pick one likely root cause and state your confidence
   - High confidence: known issue, obvious pattern match, clear from stack trace
   - Medium confidence: plausible but requires testing
   - Low confidence: wild guess, needs exploration
   - Do NOT propose multiple hypotheses in parallel

3. **Test each hypothesis with a concrete command** before acting
   - Show what command tests it
   - Show the output of that command
   - Interpret the result: does it support or refute the hypothesis?

4. **Never claim a root cause you haven't verified** by reproducing the fixed state
   - After your fix, run the original failing command or scenario again
   - Paste the complete output showing success
   - Confirm: "The originally failing scenario now works"

## What Counts as "Reproducing the Fixed State"

- The original command now exits with code 0 (or expected code)
- Error messages are gone
- The expected behavior occurs
- You show this proof explicitly

## Debugging Discipline

This approach prevents the common trap of:
- Running exploratory commands without a hypothesis
- Claiming a fix without re-running the original failure case
- Guessing at multiple causes instead of testing one at a time

One verified hypothesis beats ten plausible guesses.
