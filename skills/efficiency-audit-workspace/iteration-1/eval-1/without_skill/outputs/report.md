# Customer Dashboard Slowness — Biggest Bottleneck

## Bottleneck: N+1 query loop in `get_order_totals`

**File:** `app/services.py`, lines 5-19

```python
def get_order_totals(customer_ids):
    totals = {}
    for cid in customer_ids:
        row = db.query_one(
            "SELECT COALESCE(SUM(amount), 0) FROM orders WHERE customer_id = %s",
            (cid,),
        )
        totals[cid] = row[0]
    return totals
```

### Why this is the biggest bottleneck

- One database round-trip per customer. Docstring says it's called by the account dashboard "for every customer in the org," with customer_ids "routinely hundreds of entries."
- app/db.py uses a single global psycopg2 connection with a plain blocking execute/fetchall per call — no batching, no connection pool.
- Under concurrent load: (concurrent requests) × (hundreds of customers) round-trips contend for the same DB connection.
- **Fix:** collapse to one aggregate query: `SELECT customer_id, COALESCE(SUM(amount),0) FROM orders WHERE customer_id = ANY(%s) GROUP BY customer_id`. O(n) round-trips → O(1).

### Other inefficiencies (smaller impact)

- enrich_orders (services.py:37-47) — same N+1 shape, one query per order.
- tag_vip_customers (services.py:22-34) — O(n×m) list-membership scan (vip_ids is a list, not a set).
- get_exchange_rate (reports.py:26-32) — blocking external HTTP call per invoice line item, uncached (invoice-render path, not the dashboard).

## Recommendation

Fix get_order_totals first — replace the per-customer loop with a single batched/aggregated query.
