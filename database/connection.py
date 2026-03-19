import os
import re
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

load_dotenv()

_engine = None


def _build_url() -> str:
    """
    Build the MySQL connection URL.
    Reads DATABASE_URL from .env if set, otherwise builds from individual vars.

    Format:  mysql+pymysql://user:password@host:port/dbname?charset=utf8mb4
    """
    url = os.getenv("DATABASE_URL", "")
    if url:
        # Allow postgres-style URLs to be swapped easily:
        # replace postgresql:// or postgres:// prefix if someone forgot to update .env
        url = re.sub(r"^postgresql(\+\w+)?://", "mysql+pymysql://", url)
        url = re.sub(r"^postgres(\+\w+)?://", "mysql+pymysql://", url)
        # Ensure PyMySQL driver is specified
        if url.startswith("mysql://"):
            url = url.replace("mysql://", "mysql+pymysql://", 1)
        if "charset=" not in url:
            sep = "&" if "?" in url else "?"
            url += f"{sep}charset=utf8mb4"
        return url

    # Build from individual env vars (convenient for Docker/cloud envs)
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "password")
    database = os.getenv("MYSQL_DATABASE", "analyst_db")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"


def get_engine():
    global _engine
    if _engine is None:
        url = _build_url()
        _engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,        # reconnects on stale connections
            pool_recycle=1800,         # recycle connections every 30 min (MySQL drops idle ones)
        )
    return _engine

# Safety guard: only SELECT queries allowed
_UNSAFE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC|EXECUTE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def is_safe_sql(sql: str) -> bool:
    """Return True only if the query is a pure SELECT statement."""
    stripped = sql.strip().lstrip(";").strip()
    if _UNSAFE.search(stripped):
        return False
    if not stripped.upper().startswith("SELECT"):
        return False
    return True


def execute_query(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    """
    Execute a safe SELECT query and return results as a DataFrame.
    Raises ValueError if the query is not safe.
    """
    if not is_safe_sql(sql):
        raise ValueError(
            "Only SELECT queries are permitted. "
            "Detected a potentially unsafe SQL statement."
        )
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        rows = result.fetchall()
        columns = list(result.keys())
    return pd.DataFrame(rows, columns=columns)


def execute_write(sql: str, params: Optional[dict] = None) -> None:
    """
    Execute a write statement (CREATE TABLE / INSERT).
    Used ONLY by the ingestion pipeline — never exposed to the agent.
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def get_schema() -> dict[str, list[str]]:
    """
    Return {table_name: [col1, col2, ...]} for all user tables.

    MySQL uses the database name (schema) instead of PostgreSQL's 'public'.
    We derive it from the engine URL so no extra config is needed.
    """
    db_name = get_engine().url.database
    sql = """
        SELECT TABLE_NAME, COLUMN_NAME
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = :db
        ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"db": db_name}).fetchall()
    schema: dict[str, list[str]] = {}
    for table, col in rows:
        schema.setdefault(table, []).append(col)
    return schema


def get_schema_text() -> str:
    """Return a plain-text description of all tables + columns."""
    schema = get_schema()
    if not schema:
        return "No tables found in the database."
    lines = []
    for table, cols in schema.items():
        lines.append(f"Table: {table}")
        lines.append("  Columns: " + ", ".join(cols))
    return "\n".join(lines)