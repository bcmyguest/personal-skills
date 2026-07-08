---
name: fast-hypothesis
description: Rapid bug diagnosis—form one hypothesis and test it before exploring further
---

# Fast Hypothesis Diagnosis

You are in tight-diagnosis mode: form your **single best hypothesis** and test it rigorously before branching out into exploratory Bash chains.

## Constraints

- **No long exploratory chains**: Run at most 2-3 diagnostic commands in your first response
- **Hypothesis-first**: State your hypothesis clearly and explain why you think it's the root cause
- **Confidence level**: Include a quick confidence estimate (high/medium/low)
- **One fix**: Propose the exact fix that matches your hypothesis
- **Validate**: After applying the fix, show proof that the original symptom is gone

## Process

1. Listen to the user's bug description
2. Ask clarifying questions only if truly essential (usually 0-1)
3. Form your best single hypothesis (if multiple theories exist, pick the most likely)
4. Run 1-2 commands to validate it
5. If confirmed, give the fix. If not, state that clearly and pivot to the next hypothesis
6. Do NOT explore without a hypothesis

## Why This Works

This pairs with the Debugging Workflow rule: "When investigating a bug, act quickly: reproduce once, form a hypothesis, and make a fix. Do not run long chains of exploratory Bash commands before responding to the user."

Fast, tight diagnosis beats thorough exploration for most real bugs.
