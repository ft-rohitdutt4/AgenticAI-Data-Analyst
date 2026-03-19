"""
ingestion/ingestor.py
Upload a CSV → clean → auto-create MySQL table → load data.
"""

import re
from io import BytesIO
from typing import Optional

import pandas as pd
from sqlalchemy import inspect

from database.connection import get_engine

# Helpers

def _sanitize_name(name: str) -> str:
    """Convert an arbitrary string into a valid SQL identifier."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "unnamed"


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: sanitize column names, drop empties, parse dates."""
    df.columns = [_sanitize_name(c) for c in df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    for col in df.columns:
        if "date" in col or "time" in col:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            except Exception:
                pass
    return df

# Public API

def ingest_csv(
    file_bytes: bytes,
    table_name: Optional[str] = None,
    *,
    if_exists: str = "replace",
) -> dict:
    """
    Parse a CSV from raw bytes, clean it, and load it into MySQL.

    Parameters
    ----------
    file_bytes : bytes
        Raw CSV content.
    table_name : str | None
        Desired table name. Will be sanitized. Defaults to 'uploaded_data'.
    if_exists : str
        'replace' (default) | 'append' | 'fail'

    Returns
    -------
    dict with keys: table_name, rows, columns
    """
    try:
        df = pd.read_csv(BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    df = _clean_dataframe(df)

    safe_name = _sanitize_name(table_name or "uploaded_data")

    engine = get_engine()

    # MySQL TEXT columns can't serve as index — use index=False (already done)
    # dtype mapping: pandas → MySQL-friendly via SQLAlchemy's auto inference
    df.to_sql(
        safe_name,
        con=engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=500,   # smaller chunks — MySQL is stricter on packet size
    )

    return {
        "table_name": safe_name,
        "rows": len(df),
        "columns": list(df.columns),
    }


def list_tables() -> list[str]:
    """Return names of all tables in the current MySQL database."""
    engine = get_engine()
    inspector = inspect(engine)
    # MySQL: no schema='public' — pass the database name or None for current DB
    return inspector.get_table_names()