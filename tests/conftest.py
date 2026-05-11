"""
conftest.py — shared pytest configuration.
Sets dummy environment variables so imports don't fail
when GEMINI_API_KEY or Postgres vars aren't set.
"""
import os
import sys
import pytest

# Add app/ to path so tests can import main, ingest, retrieval, db
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# Set dummy env vars before any module imports trigger them
os.environ.setdefault("GEMINI_API_KEY", "dummy-key-for-tests")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "ragdb")
os.environ.setdefault("POSTGRES_USER", "postgres")
os.environ.setdefault("POSTGRES_PASSWORD", "postgres")
