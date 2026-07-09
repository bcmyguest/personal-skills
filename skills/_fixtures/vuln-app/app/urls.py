"""URL routing."""
from django.urls import path

from . import views

urlpatterns = [
    path("products/search", views.search_products, name="product-search"),
    path("products/<int:product_id>", views.get_product, name="product-detail"),
    path("invoices/<int:invoice_id>", views.get_invoice, name="invoice-detail"),
    path("invoices/lines", views.invoice_lines, name="invoice-lines"),
    path("reports/export", views.export_report, name="report-export"),
    path("reports/spec", views.upload_report_spec, name="report-spec-upload"),
    path("events/export", views.events_export, name="events-export"),
    path("dashboard", views.account_dashboard, name="account-dashboard"),
    path("avatar", views.download_avatar, name="avatar-download"),
]
