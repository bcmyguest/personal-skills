# Efficiency Audit: vuln-app Django Backend

Target: `app/`. Read-only review. Ranked worst-first by cost impact (frequency × magnitude).

## 1. `render_invoice_lines` — synchronous FX API call per line item (worst offender)
`reports.py:26-41`. One blocking HTTPS call per line item, no caching, no dedup within an invoice, no timeout. 100-300ms × 20 line items = 2-6s added serial latency per invoice. CACHE_TTL_SECONDS exists in config.py, unused. Fix: dedup currencies with set(), cache by currency, add timeout=.

## 2. `get_order_totals` — N+1 query, one round trip per customer
`services.py:5-19`. O(n) DB round trips, n = hundreds. 300 customers ≈ 3s of latency. Fix: `WHERE customer_id = ANY(%s) GROUP BY customer_id`.

## 3. `enrich_orders` — N+1 query per order, redundant refetching
`services.py:37-47`. Same O(n) shape but per order, zero memoization. 500 orders / 50 customers = 500 queries where 1 batched query suffices.

## 4. `tag_vip_customers` — O(n·m) list membership scan, computed twice
`services.py:22-34`. vip_ids is a list, so `in` is O(m); both grow with org size. Also computed twice per customer. Fix: `vip_set = set(vip_ids)`; delete dead line.

## 5. `export_report` — unbounded subprocess spawn per request
`views.py:38-46`. Fresh Python interpreter per hit, no concurrency cap, handle never reaped.

## 6. `db.py::get_connection` — single global connection, no pooling
`db.py:6-19`. One shared connection app-wide, caps DB throughput at 1 query in flight; multiplies #2/#3. Thread-safety risk too.

## 7. `export_all_events` — unbounded full-table materialization
`reports.py:13-23`. No LIMIT/pagination/cursor; fetchall() pulls whole ever-growing table into memory, ~2x peak. Holds the shared connection for the scan.

## 8. `search_products` — unanchored LIKE, no pagination, public
`views.py:16-22`. Leading-wildcard LIKE can't use an index → full scan every call; no LIMIT despite config.PAGE_SIZE. (Also SQLi — separate security defect.)

## Summary
Highest-value fixes: #6 (pooling) and collapsing N+1s in #1/#2/#3 into batched/cached lookups.

**Flag:** config.py has a hardcoded DB password fallback and Django SECRET_KEY (fixture values). Several items (search_products SQL concat, export_report shell injection) are primarily security defects; recommend a separate security pass.
