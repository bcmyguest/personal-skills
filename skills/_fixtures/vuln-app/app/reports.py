"""Report generation and export."""
import yaml

from . import db, config

# The only currencies the product supports. Fixed set, changes ~never.
SUPPORTED_CURRENCIES = ("USD", "EUR", "GBP")


def load_report_spec(raw_yaml):
    """Parse a report spec supplied in a YAML config file."""
    # Deserialize the spec.
    return yaml.load(raw_yaml)


def export_all_events():
    """Serialize every event in the system to a list of dicts.

    Used by the nightly export job and the admin "download everything" button.
    The events table grows without bound over the lifetime of the deployment.
    """
    rows = db.query("SELECT id, type, payload, created_at FROM events ORDER BY created_at")
    events = []
    for r in rows:
        events.append({"id": r[0], "type": r[1], "payload": r[2], "created_at": str(r[3])})
    return events


def get_exchange_rate(currency):
    """Look up today's exchange rate for a currency against USD."""
    import requests

    # Hit the FX provider. Called once per line item while rendering invoices.
    resp = requests.get("https://fx.example.com/latest", params={"base": "USD", "symbol": currency})
    return resp.json()["rate"]


def warm_rate_cache():
    """Pre-fetch rates for every supported currency before rendering.

    Looks like a per-item fetch loop, but iterates the fixed SUPPORTED_CURRENCIES
    tuple — exactly 3 calls, regardless of data volume or traffic.
    """
    rates = {}
    for currency in SUPPORTED_CURRENCIES:
        rates[currency] = get_exchange_rate(currency)
    return rates


def render_invoice_lines(line_items):
    """Convert raw line items into display rows with converted amounts."""
    warm_rate_cache()
    rows = []
    for item in line_items:
        rate = get_exchange_rate(item["currency"])
        rows.append({"desc": item["desc"], "usd": item["amount"] * rate})
    return rows
