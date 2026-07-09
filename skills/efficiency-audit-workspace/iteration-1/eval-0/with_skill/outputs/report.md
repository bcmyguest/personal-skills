## Efficiency audit — 3 confirmed issue(s)

**Scope:** `/home/bguest/personal-skills/skills/_fixtures/vuln-app/app` (Django-style app: `auth.py`, `config.py`, `db.py`, `reports.py`, `services.py`, `urls.py`, `views.py`)

### 1. N+1 query per customer in `get_order_totals` (and `enrich_orders`)  ·  High
**Where:** `app/services.py:5-19` (also `app/services.py:37-47`, same pattern)
**Issue:** `get_order_totals(customer_ids)` loops over the customer-id list and issues one `SELECT COALESCE(SUM(amount), 0) FROM orders WHERE customer_id = %s` round-trip per customer via `db.query_one`. The function's own docstring states it's "called by the account dashboard for every customer in the org, so `customer_ids` routinely has hundreds of entries" — this is a documented production hot path, not a rare batch job. `enrich_orders(orders)` (`services.py:37-47`) has the identical shape: one `SELECT name FROM customers WHERE id = %s` per order row, run on every order-list render.
**Cost:** 1 query issued per element of `customer_ids` → **N round-trips for N customers**, with N in the hundreds per the docstring. Each round-trip pays full network + connection overhead (see `db.py:22-28`, no batching helper). At 300 customers that's 300 sequential DB round-trips to render one dashboard page instead of 1. `enrich_orders` compounds this on every order list of size M with another M round-trips.
**Fix:** Replace the loop with a single aggregate query — `SELECT customer_id, COALESCE(SUM(amount),0) FROM orders WHERE customer_id = ANY(%s) GROUP BY customer_id` — and similarly batch `enrich_orders` with `SELECT id, name FROM customers WHERE id IN (...)` joined in Python, or use a real ORM `select_related`/`prefetch_related`-equivalent join.

### 2. `export_all_events` loads the entire unbounded events table into memory  ·  High
**Where:** `app/reports.py:13-23`
**Issue:** `export_all_events()` runs `SELECT id, type, payload, created_at FROM events ORDER BY created_at` with no `LIMIT`, `WHERE`, or keyset pagination, then materializes every row into a Python list of dicts. The docstring states the events table "grows without bound over the lifetime of the deployment" and that this function backs both a nightly export job *and* an admin "download everything" button — i.e., it's reachable on demand, not just at a controlled cron time.
**Cost:** Memory and query cost scale **linearly and unboundedly with total historical row count** (not bounded by any window). The result is buffered twice at peak — once as DB row tuples (`db.py:26` `cur.fetchall()`), once as the rebuilt list of dicts (`reports.py:21-22`) — roughly 2x the raw payload size resident in memory at once. A deployment with millions of events (plausible for an "events" table over months/years) turns one admin click into a multi-GB, unbounded-duration in-process materialization with no backpressure.
**Fix:** Stream results in bounded batches (server-side cursor / `fetchmany(page_size)`) and write/yield each page directly to the response or export file instead of building one Python list; add a keyset-paginated `WHERE created_at > :cursor LIMIT :page_size` loop.

### 3. Synchronous, uncached FX API call per invoice line item  ·  Medium-High
**Where:** `app/reports.py:26-41` (`get_exchange_rate` called from `render_invoice_lines`)
**Issue:** `render_invoice_lines(line_items)` calls `get_exchange_rate(item["currency"])` inside its loop — once per line item, with no memoization across items and no timeout on the `requests.get` call. `config.py:18` defines `CACHE_TTL_SECONDS = 300`, implying caching infrastructure was intended, but it is never referenced anywhere in the codebase — this is dead config, not a real cache.
**Cost:** **N synchronous external HTTP round-trips per invoice**, where N = line-item count, even when many lines share the same currency (e.g., a 50-line USD/EUR invoice makes up to 50 outbound calls instead of at most 2). Each call is network-latency-bound and unbounded (no timeout), so invoice rendering time scales linearly with line count and is exposed to third-party latency/outages once per line rather than once per distinct currency.
**Fix:** Cache exchange rates per currency for the duration of one render call (a local `dict` keyed by currency is enough to cut N calls to len(distinct currencies)), and layer in the already-declared `CACHE_TTL_SECONDS` as a real TTL cache (e.g. `django.core.cache`) so repeated invoice renders within the TTL window make zero external calls.

---

**Also observed but not in the top 3** (worth a follow-up pass): `services.py:22-34` `tag_vip_customers` does an O(N·M) list-membership scan (`c["id"] in vip_ids`) computed twice per customer instead of once against a `set`; `views.py` `search_products` has no `LIMIT`/pagination despite `config.PAGE_SIZE = 100` existing unused; `db.py` shares one global connection process-wide with no pool and no try/finally around the cursor. These scale with data/traffic too but were judged lower-impact than the three above for this pass, capped at 3 per the audit's target.

*This is a focused top-3 audit (target reached), not exhaustive coverage — other paths in the app were not deeply chased once the target was hit.*
