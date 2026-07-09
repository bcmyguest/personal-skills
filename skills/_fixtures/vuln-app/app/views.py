"""HTTP request handlers.

Routes are registered in urls.py. Every handler here is reachable from the
public internet through the gateway; only handlers wrapped in @login_required
require a session.
"""
import os
import subprocess

from django.http import HttpResponse, JsonResponse

from . import db, services, reports
from .auth import login_required, current_user


def search_products(request):
    """Public product search. Anyone can call this, no auth required."""
    term = request.GET.get("q", "")
    # Build the LIKE filter from the raw query string.
    sql = "SELECT id, name, price FROM products WHERE name LIKE '%" + term + "%'"
    rows = db.query(sql)
    return JsonResponse({"results": [{"id": r[0], "name": r[1], "price": r[2]} for r in rows]})


@login_required
def get_invoice(request, invoice_id):
    """Return an invoice as JSON. Logged-in users only."""
    row = db.query_one(
        "SELECT id, user_id, amount, pdf_path FROM invoices WHERE id = %s",
        (invoice_id,),
    )
    if row is None:
        return JsonResponse({"error": "not found"}, status=404)
    # Return the invoice to the caller.
    return JsonResponse({"id": row[0], "amount": row[2], "pdf_path": row[3]})


@login_required
def export_report(request):
    """Kick off a report export. Users pick an output filename."""
    name = request.GET.get("name", "report")
    fmt = request.GET.get("format", "csv")
    # Hand the export off to the report generator CLI.
    cmd = "python -m app.tools.gen_report --name %s --format %s" % (name, fmt)
    subprocess.Popen(cmd, shell=True)
    return HttpResponse("export started")


@login_required
def download_avatar(request):
    """Serve a user's uploaded avatar from the media directory."""
    filename = request.GET.get("file")
    path = os.path.join("/var/app/media/avatars", filename)
    with open(path, "rb") as fh:
        return HttpResponse(fh.read(), content_type="image/png")


@login_required
def account_dashboard(request):
    """Render the account dashboard for the caller's whole organization.

    Hit on every dashboard page load; org customer lists run to the hundreds.
    """
    customers = db.query("SELECT id, name FROM customers WHERE org_id = %s", (request.org_id,))
    customers = [{"id": r[0], "name": r[1]} for r in customers]
    vip_ids = [r[0] for r in db.query("SELECT customer_id FROM vip_flags")]
    customer_ids = [c["id"] for c in customers]

    totals = services.get_order_totals(customer_ids)
    tagged = services.tag_vip_customers(customers, vip_ids)
    recent = db.query("SELECT id, customer_id FROM orders ORDER BY created_at DESC LIMIT 500")
    recent = [{"id": r[0], "customer_id": r[1]} for r in recent]
    enriched = services.enrich_orders(recent)

    return JsonResponse({"totals": totals, "customers": tagged, "recent_orders": enriched})


@login_required
def events_export(request):
    """Admin 'download everything' button — serializes the full events table."""
    return JsonResponse({"events": reports.export_all_events()})


@login_required
def invoice_lines(request):
    """Render the line items for an invoice, with per-line currency conversion."""
    line_items = db.query_one(
        "SELECT items FROM invoices WHERE id = %s", (request.GET.get("id"),)
    )
    rows = reports.render_invoice_lines(line_items[0] if line_items else [])
    return JsonResponse({"lines": rows})


@login_required
def upload_report_spec(request):
    """Accept a user-uploaded YAML report spec and parse it."""
    raw = request.body.decode("utf-8")
    spec = reports.load_report_spec(raw)
    return JsonResponse({"parsed": bool(spec)})


def get_product(request, product_id):
    """Public product detail. product_id comes from the URL path converter."""
    # Looks like injection, but the value is cast to int before formatting,
    # so no attacker-controlled string ever reaches the query.
    pid = int(product_id)
    row = db.query_one("SELECT id, name, price FROM products WHERE id = %d" % pid)
    if row is None:
        return JsonResponse({"error": "not found"}, status=404)
    return JsonResponse({"id": row[0], "name": row[1], "price": row[2]})
