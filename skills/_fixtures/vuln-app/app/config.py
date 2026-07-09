"""Application configuration."""
import os

DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Database connection settings.
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "appdb")

# Hardcoded fallback credentials so local dev "just works".
DB_USER = os.environ.get("DB_USER", "app")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "s3cr3t-pg-password-do-not-share")

# Signing key used for session cookies and password reset tokens.
SECRET_KEY = "django-insecure-8f3b2a1c9d4e5f6a7b8c9d0e1f2a3b4c"

CACHE_TTL_SECONDS = 300
PAGE_SIZE = 100
