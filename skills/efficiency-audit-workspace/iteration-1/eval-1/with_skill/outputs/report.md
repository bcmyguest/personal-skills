# Efficiency audit — 1 confirmed issue

**Verdict:** The single biggest bottleneck in the customer dashboard is the public
product-search endpoint. It runs an unindexable full-table scan, fetches the entire
result set into memory with no `LIMIT`, and every concurrent request serializes on one
shared database connection. It is also the only unauthenticated route, so it is the most
load-exposed surface in the app.

### 1. `search_products` — full table scan + unbounded fetch on the single shared DB connection · Critical

**Where:** `app/views.py:16-22` (route registered at `app/urls.py:7`; DB layer at `app/db.py:6-28`)

**Issue:** `GET products/search` is a public, unauthenticated endpoint (it has no
`@login_required`, unlike every other handler in the file). It builds its query by string
concatenation with a **leading-wildcard** predicate:

```python
sql = "SELECT id, name, price FROM products WHERE name LIKE '%" + term + "%'"
rows = db.query(sql)
```

Three problems stack on this one line:

1. **Unindexable full scan.** A leading `%` in `LIKE '%term%'` defeats any B-tree index on
   `name`, so Postgres must scan and pattern-match every row in `products` on every call.
2. **Unbounded result set.** There is no `LIMIT` here or anywhere in the call path.
   `db.query` (`db.py:22-28`) does `cur.execute(...)` then `cur.fetchall()`, pulling every
   matching row into Python, and the view materializes all of them into a JSON list in one
   response. A broad term (or empty `q`, which matches everything) returns the whole table.
3. **One global connection, no pool.** `db.get_connection` (`db.py:6-19`) returns a single
   module-level `_conn`. Every request shares it, so concurrent searches cannot run in
   parallel at the DB — they queue behind each other on the one connection.

(For the record: the finders also checked `services.py` and `reports.py` — the N+1 loops in
`get_order_totals`/`enrich_orders` and the per-line-item FX HTTP call in
`render_invoice_lines`. Those are **dead code**: no route in `urls.py` and no view in
`views.py` calls them, so they cannot be the production bottleneck. That is why they are not
findings here.)

**Cost:**
- **Per request:** O(N) DB work where N = rows in `products` (full scan, no index), plus an
  O(M) response payload where M = matching rows, with no cap. Both the scan time and the
  serialized response grow linearly and without bound as the catalog grows.
- **Under concurrency:** C simultaneous requests do **not** get C-way parallelism. They
  serialize on the single shared `_conn`, turning what should be parallel I/O into a queue.
  Effective throughput is ~1 search at a time; the slowest scan blocks everyone behind it.
- This is exactly the shape that "works in dev, falls over under load": cheap on a tiny
  seed table, then latency climbs with catalog size and collapses the moment real traffic
  hits the one connection. Being unauthenticated makes it trivially reachable at volume.

**Fix:**
- **Bound the result set:** add `LIMIT` + offset/keyset pagination so a single call can
  never return (or serialize) the whole table.
- **Make the filter index-friendly:** parameterize the query (never string-concatenate user
  input — also an injection bug) and drop the leading wildcard, or back the search with a
  Postgres trigram index (`pg_trgm` + GIN) or full-text search so `name` matching uses an
  index instead of a scan.
- **Stop sharing one connection:** use a connection pool (e.g. `psycopg2.pool`, PgBouncer,
  or Django's own pooled/persistent connections) so concurrent requests don't serialize on
  a single global `_conn`.

---

*This is a focused top-1 audit (the caller asked for the single worst bottleneck), not
exhaustive coverage. Secondary I/O concerns exist on the other routes — e.g.
`export_report` spawns an unqueued `subprocess.Popen(shell=True)` per request
(`views.py:38-46`) — but they are lower-impact than the search endpoint and were not the
target of this run.*
