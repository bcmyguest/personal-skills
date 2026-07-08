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

### fast-hypothesis ✅

**Location**: `claude-memory/skills/fast-hypothesis/SKILL.md`

**Description**: Rapid bug diagnosis—form one hypothesis and test it before exploring further

Constrains diagnosis to hypothesis-first mode: form your best single hypothesis, test it with 1-2 commands, propose a fix, validate.

Pairs with MEMORY.md Debugging Workflow rule.

### reproduce-bug ✅

**Location**: `claude-memory/skills/reproduce-bug/SKILL.md`

**Description**: Debug any bug using strict reproduce-verify discipline

Generic reproduction-verify framework for any bug (not toolchain-specific):
1. Reproduce the exact failure (show complete output)
2. Form ONE hypothesis at a time
3. Test each hypothesis before acting
4. Never claim a fix without re-running the original failure case and showing success

Applies to Python, Node, Rust, system bugs—anything reproducible.
