import json

from database.connection import execute_query, get_schema_text
from llm.client import chat
from llm.prompts import ANALYST_SYSTEM, SQL_GEN_PROMPT


def run_sql_tool(task: str, context: str = "") -> dict:
    """
    Given a task description and optional prior context, generate and run a SQL query.

    Returns
    -------
    {
        "sql": str,
        "data": list[dict],   # rows as list of dicts
        "columns": list[str],
        "row_count": int,
        "error": str | None,
    }
    """
    schema = get_schema_text()
    system_msg = ANALYST_SYSTEM.format(schema=schema)

    user_msg = SQL_GEN_PROMPT.format(task=task, context=context or "None")

    raw_sql = chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    ).strip()

    # Strip accidental markdown fences
    if raw_sql.startswith("```"):
        lines = raw_sql.split("\n")
        raw_sql = "\n".join(
            l for l in lines if not l.startswith("```")
        ).strip()

    try:
        df = execute_query(raw_sql)
        return {
            "sql": raw_sql,
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "row_count": len(df),
            "error": None,
        }
    except Exception as exc:
        return {
            "sql": raw_sql,
            "data": [],
            "columns": [],
            "row_count": 0,
            "error": str(exc),
        }