## Security audit — 3 confirmed vulnerabilities

Target: `/home/bguest/personal-skills/skills/_fixtures/vuln-app` (an 8-file Django-style fixture app — no `manage.py`/`settings.py`/ORM models; per its own `README.md` it is "deliberately flawed," not runnable, and used only to test audit tooling). All findings below were reproduced by reading the live source and re-verified adversarially, checking for any mitigating control before confirming.

### 1. Unauthenticated SQL Injection in `search_products`  ·  Critical
**Where:** `app/views.py:16-22` (route wired at `app/urls.py:7`, executed via `app/db.py:22-28`)
**Vulnerability:** The `products/search` route has no `@login_required` or any auth wrapper — contrast with every other view in the file — and no global auth middleware exists in this fixture to catch it, so it is reachable by a fully anonymous, unauthenticated attacker. The handler takes `term = request.GET.get("q", "")` and concatenates it directly into a SQL string: `sql = "SELECT id, name, price FROM products WHERE name LIKE '%" + term + "%'"`, then calls `db.query(sql)` with **no params tuple**. `db.query()`'s implementation (`app/db.py:22-28`) calls `cur.execute(sql, params or ())`; with `params=None` this executes the already-concatenated string verbatim — psycopg2 cannot retroactively sanitize text baked into the query. Nothing validates, escapes, or allowlists `term` anywhere in the chain. The codebase demonstrably knows the correct pattern (`get_invoice` at `app/views.py:28-31` uses `%s` placeholders + a params tuple) — it simply wasn't applied here.
**Proof:** Minimal PoC (non-destructive, boolean-based):
```
GET /products/search?q=nonexistent' OR '1'='1
```
Substituting into line 20 yields `... WHERE name LIKE '%nonexistent' OR '1'='1%'`, which the SQL parser evaluates as `(name LIKE '%nonexistent')  OR  ('1'='1')` — always true — returning **every row in `products`** regardless of the intended filter, and confirming the quote character breaks out of the string literal. A UNION-based follow-up payload (e.g. `q=x' UNION SELECT username,password,1 FROM users-- `) would extend this to reading arbitrary tables.
**Fix:** Parameterize the query: `db.query("SELECT id, name, price FROM products WHERE name LIKE %s", (f"%{term}%",))`.

### 2. Authenticated OS Command Injection in `export_report`  ·  Critical
**Where:** `app/views.py:38-46`
**Vulnerability:** The route is gated by `@login_required`, but that decorator (`app/auth.py:12-19`) only checks whether *some* user is logged in — no role, permission, or ownership check — so **any authenticated user**, regardless of privilege level, can reach it. The handler reads `name = request.GET.get("name", "report")` and `fmt = request.GET.get("format", "csv")`, builds a shell command via `%`-formatting — `cmd = "python -m app.tools.gen_report --name %s --format %s" % (name, fmt)` — and executes it with `subprocess.Popen(cmd, shell=True)`. Because `cmd` is a single pre-built string passed with `shell=True`, the entire string is handed to `/bin/sh -c` for full shell parsing; there is no allowlist on `fmt`, no `shlex.quote()`, and no sanitization of either parameter. Contrast with the safe pattern (a list-form `Popen([...])` with no `shell=True`), which is not used here.
**Proof:** Minimal PoC (non-destructive, constructed only):
```
GET /reports/export?name=report&format=csv;touch%20/tmp/pwned;
Cookie: <any valid authenticated session>
```
URL-decoded, `fmt = "csv;touch /tmp/pwned;"`, producing `cmd = "python -m app.tools.gen_report --name report --format csv;touch /tmp/pwned;"`. `/bin/sh -c` splits this on `;` into three sequential commands, the second of which — `touch /tmp/pwned` — executes as an entirely separate, attacker-injected shell command before `gen_report` is ever invoked. `touch /tmp/pwned` is filesystem-harmless and trivially reversible, suitable purely as a proof marker; the same injection point accepts any shell metacharacter (`&&`, `|`, backticks, `$()`).
**Fix:** Drop `shell=True` and use the list form: `subprocess.Popen(["python", "-m", "app.tools.gen_report", "--name", name, "--format", fmt])`, plus allowlist `fmt` to a known set (e.g. `{"csv", "pdf"}`).

### 3. Broken Object-Level Authorization (IDOR) in `get_invoice`  ·  High
**Where:** `app/views.py:25-35`
**Vulnerability:** Reachable by any authenticated user (same weak `login_required` gate as above — session presence only, no ownership/role check). The handler fetches `SELECT id, user_id, amount, pdf_path FROM invoices WHERE id = %s` (parameterized — not itself injectable) using the `invoice_id` path parameter, which Django's `<int:invoice_id>` converter only constrains to be numeric, providing no authorization value. The returned row's `user_id` (`row[1]`) is fetched but **never referenced again** — not compared to the requesting user's identity anywhere in the function — before the response is built from `row[0]`, `row[2]`, `row[3]` and returned as JSON with a 200. Invoice IDs are plain sequential integers (no UUID or per-user secondary key), making enumeration trivial once the missing check is exploited.
**Proof:** User A (`user_id=1`) owns invoice `101` (private `amount` and `pdf_path`). User B, a distinct authenticated account with no relationship to invoice 101, sends:
```
GET /invoices/101
Cookie: <User B's valid session cookie>
```
The query matches invoice 101 regardless of who is asking, since `user_id` is never checked against User B's identity. The server returns `200 OK` with `{"id": 101, "amount": <A's amount>, "pdf_path": <A's pdf_path>}`. Iterating `/invoices/1`, `/invoices/2`, … `/invoices/N` discloses every invoice in the system to any logged-in user.
**Fix:** After the fetch, compare ownership before responding: `if row[1] != current_user(request)["id"]: return JsonResponse({"error": "not found"}, status=404)` (or scope the `WHERE` clause by `user_id` directly).

---

**Other findings surfaced but not carried through verification** (dropped per the skill's stop-at-target rule, not disproven — noted for completeness): a path-traversal / arbitrary-file-read candidate in `download_avatar` (`app/views.py:49-55`, unsanitized `file` query param joined into a filesystem path); an unsafe `yaml.load()` call with no `Loader=` in `app/reports.py:7-10` (paired with the outdated `PyYAML==5.1` pin) — currently unreachable from any entry point in this fixture, so it didn't meet the "concrete, reachable" bar, but is a live landmine if a caller is ever added; a hardcoded, non-overridable `SECRET_KEY` in `app/config.py:15-16` used to sign both session cookies and password-reset tokens, and a hardcoded fallback `DB_PASSWORD` at `app/config.py:13`; and six outdated pinned dependencies in `requirements.txt` (`Django==3.2.4`, `PyYAML==5.1`, `requests==2.19.1`, `Jinja2==2.10`, `gunicorn==20.0.4`, `psycopg2-binary==2.9.1`) that warrant a CVE lookup pass if this audit is extended.

This is a focused top-3 audit (target reached and verification stopped per the skill's design), not an exhaustive line-by-line review — the items above are worth a follow-up pass if more than the top 3 are wanted.
