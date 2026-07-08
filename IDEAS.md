# Ideas

Not yet skills or config — parked here until proven useful.

## PostToolUse lint hook

Auto-run ruff + pyrefly after every edit in Python repos:

```json
// .claude/settings.json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{"type": "command", "command": "uv run ruff check --fix . && uv run pyrefly check"}]
    }]
  }
}
```

## Skill candidate: fast-hypothesis debugging

Prompt that works well as-is (maybe make it a skill):

> Diagnose this bug. Give me your single best hypothesis and the exact fix in one
> response before running more than 2-3 commands.

Pairs with the CLAUDE.md Debugging Workflow rule about not over-exploring.

## Skill candidate: broken-dev-environment diagnosis (reproduce-verify discipline)

Generalized from a `pnpm i` debugging session. Applies to any broken build/install/dev
environment:

> My dev environment / build command is broken. Diagnose it with strict
> reproduce-verify discipline:
>
> 1. Reproduce the exact failure and paste the error.
> 2. Form ONE hypothesis at a time and state your confidence.
> 3. Test each hypothesis with a concrete command before acting.
> 4. Never claim a root cause you haven't verified by reproducing the fixed state.
>
> Check the common culprits for the toolchain in question (stray config files in $HOME,
> version-manager/tool mismatch, empty bin dirs, stale caches or dependency dirs).
> Only declare it fixed after the originally failing command actually succeeds, and
> show the proof.
