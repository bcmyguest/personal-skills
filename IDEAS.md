# Ideas

Parked items awaiting refinement. Split into two tracks:

## Hook Configurations (Pending Design)

These are system-level hooks that should live in plugin settings, not as discrete skills.

### PostToolUse lint hook (HOLD - needs design)

Auto-run ruff + pyrefly after every edit in Python repos:

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{"type": "command", "command": "uv run ruff check --fix . && uv run pyrefly check"}]
    }]
  }
}
```

**Status**: ON HOLD. Needs solution for:
- How to detect if project is uv-based (vs pip, poetry, etc.)
- How to avoid running in non-Python projects
- Should it be always-on or opt-in?

**Possible approaches**:
- Check for `pyproject.toml` + `uv.lock` before running
- Make it opt-in via skill flag or user setting
- Only enable in projects with SessionStart hook that detects uv

**Defer until**: We clarify project detection strategy

---

## Skills (Implemented)

These have been created as executable skills.

### diagnose-failure ✅

**Location**: [`skills/diagnose-failure/SKILL.md`](skills/diagnose-failure/SKILL.md)

Replaces `fast-hypothesis` and `reproduce-bug`: one hypothesis tested first, with a strict reproduce-verify path when a reliable local repro exists.
