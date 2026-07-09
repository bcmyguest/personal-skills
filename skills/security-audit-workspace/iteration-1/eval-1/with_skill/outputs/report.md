# Security audit — 1 confirmed vulnerability

**Target:** `/home/bguest/personal-skills/skills/_fixtures/vuln-app/app`
**Verdict:** The one thing that gets you paged is an **unauthenticated SQL injection** on the public product-search endpoint. Anyone on the internet can read your database with a single GET request — no login, no token, no tricks. Fix this before you ship.

---

### 1. Unauthenticated SQL injection in `search_products`  ·  Critical

**Where:** `app/views.py:20` (route registered at `app/urls.py:7`; sink at `app/db.py:25`)

**Vulnerability:** The public `search_products` handler takes the raw `q` query-string parameter and concatenates it straight into a SQL string, which is then executed. The endpoint has **no `@login_required` decorator** (unlike every other handler in the file — its own docstring says "Anyone can call this, no auth required"), so the attacker is **completely unauthenticated**. `db.query()` does support parameterized queries (`cur.execute(sql, params or ())`), but this call passes no params, so the payload is already baked into the SQL string before psycopg2 ever sees it — zero escaping happens. The result is arbitrary read access to the database: an attacker can UNION/boolean/error-extract any table the DB user can reach (credentials, invoices, PII), reachable by anyone who can hit the gateway.

```python
# app/views.py:16-22
def search_products(request):
    """Public product search. Anyone can call this, no auth required."""
    term = request.GET.get("q", "")
    # Build the LIKE filter from the raw query string.
    sql = "SELECT id, name, price FROM products WHERE name LIKE '%" + term + "%'"
    rows = db.query(sql)                       # -> db.py:25  cur.execute(sql, params or ())
    return JsonResponse({"results": [...]})
```

**Proof (minimal, non-destructive PoC):**

Boolean probe — if this returns the full products table instead of filtered/empty results, injection is proven:
```
GET /products/search?q=' OR '1'='1
```

Data-exfiltration escalation (read-only UNION; find the column count with `ORDER BY 1--`, `ORDER BY 2--`, … first):
```
GET /products/search?q=x%25' UNION SELECT username,password,1 FROM users--
```
Decoded `q`: `x%' UNION SELECT username,password,1 FROM users-- ` — closes the `LIKE '%…%'` literal and appends a UNION that dumps another table into the JSON response. No writes, no `DROP`/`UPDATE` needed to prove it.

**Fix:** Parameterize — let the driver bind the value instead of string-building it:
```python
rows = db.query("SELECT id, name, price FROM products WHERE name LIKE %s", (f"%{term}%",))
```
(And reconsider whether this endpoint should truly be public / unrate-limited, since every other handler here is `@login_required`.)

---

### Also found, but not the scariest (deliberately not crowned)

The audit adversarially checked two other serious-looking candidates and downgraded both because neither is **reachable** in this codebase — reachability is what separates a page from a lint nit:

- **Hardcoded `SECRET_KEY` (`app/config.py:16`)** — a static, committed `django-insecure-…` key whose comment says it signs session cookies and password-reset tokens. Alarming on its face (a known key is forge-anyone's-session), **but no code in this app actually consumes it** to sign or verify anything — `auth.py`'s `current_user` is a bare `getattr(request, "user", None)`, with identity set by upstream middleware that isn't in this repo. So the auth-bypass is **unproven from this code**; treat it as a hardcoded-secret / weak-config issue: move the key (and the `DB_PASSWORD` fallback at `config.py:13`) to a secret manager with no committed fallback, fail startup if unset, and rotate if this value was ever used outside a fixture.
- **`yaml.load(raw_yaml)` without `Loader=` (`app/reports.py:10`, with `PyYAML==5.1` pinned in `requirements.txt`)** — a genuine RCE-class bug (PyYAML 5.1's default loader deserializes arbitrary Python objects; CVE-2020-1747 / CVE-2020-14343 class). **But `load_report_spec` has no caller anywhere in the app and no route reaches it** — it's dead code today. Latent, not a live exploit. Fix it (`yaml.safe_load`) and upgrade PyYAML before anything wires it to input.

Lower-severity items confirmed present for your backlog: command injection via `subprocess.Popen(..., shell=True)` on `name`/`format` GET params (`views.py:44-45`, auth-gated), path traversal / arbitrary file read in `download_avatar` (`views.py:52-54`, auth-gated), an IDOR in `get_invoice` (`views.py:26`, no ownership check), and an EOL dependency set in `requirements.txt` (Django 3.2.4, PyYAML 5.1, requests 2.19.1, Jinja2 2.10, gunicorn 20.0.4) carrying multiple known CVEs.

---

*This is a focused top-1 audit (you asked for the single scariest hole), not exhaustive coverage. The items in the second section are real and worth fixing, but the unauthenticated SQL injection is the one that pages you.*
