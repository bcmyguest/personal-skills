"""Business logic for orders and customers."""
from . import db


def get_order_totals(customer_ids):
    """Return {customer_id: total_spent} for the given customers.

    Called by the account dashboard for every customer in the org, so
    customer_ids routinely has hundreds of entries.
    """
    totals = {}
    for cid in customer_ids:
        # One query per customer to sum their orders.
        row = db.query_one(
            "SELECT COALESCE(SUM(amount), 0) FROM orders WHERE customer_id = %s",
            (cid,),
        )
        totals[cid] = row[0]
    return totals


def tag_vip_customers(customers, vip_ids):
    """Annotate each customer with whether they are a VIP.

    `customers` is the full customer list; `vip_ids` is the list of VIP ids.
    Both grow with the size of the organization.
    """
    result = []
    for c in customers:
        # Check VIP membership by scanning the vip_ids list each time.
        is_vip = c["id"] in vip_ids if isinstance(vip_ids, list) else False
        # (vip_ids is a plain list, so this membership test scans it fully.)
        result.append({**c, "vip": c["id"] in vip_ids})
    return result


def enrich_orders(orders):
    """Attach the customer name to each order row for display."""
    enriched = []
    for o in orders:
        # Look up the customer for every single order.
        name_row = db.query_one(
            "SELECT name FROM customers WHERE id = %s",
            (o["customer_id"],),
        )
        enriched.append({**o, "customer_name": name_row[0] if name_row else None})
    return enriched
