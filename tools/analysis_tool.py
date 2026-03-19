import json

import pandas as pd

from llm.client import chat
from llm.prompts import ANALYST_SYSTEM, ANALYSIS_PROMPT
from database.connection import get_schema_text


def run_analysis_tool(task: str, data: list[dict]) -> dict:
    """
    Perform pandas analysis on raw data rows and return an LLM-generated insight.

    Returns
    -------
    {
        "summary_stats": dict,
        "insight": str,
        "error": str | None,
    }
    """
    if not data:
        return {"summary_stats": {}, "insight": "No data to analyse.", "error": None}

    try:
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include="number")

        summary: dict = {}

        if not numeric_cols.empty:
            desc = numeric_cols.describe().to_dict()
            summary["describe"] = desc

            # Period-over-period change (if there's exactly one numeric + one date/text col)
            date_cols = [c for c in df.columns if "date" in c.lower() or "month" in c.lower() or "year" in c.lower()]
            num_cols_list = list(numeric_cols.columns)

            if date_cols and num_cols_list:
                date_col = date_cols[0]
                num_col = num_cols_list[0]
                try:
                    sorted_df = df.sort_values(date_col)
                    sorted_df["_pct_change"] = sorted_df[num_col].pct_change() * 100
                    summary["pct_change"] = sorted_df[[date_col, num_col, "_pct_change"]].tail(6).to_dict(orient="records")
                except Exception:
                    pass

        # Truncate data for prompt (max 50 rows)
        data_sample = data[:50]
        data_json = json.dumps(data_sample, default=str, indent=2)

        schema = get_schema_text()
        system_msg = ANALYST_SYSTEM.format(schema=schema)
        user_msg = ANALYSIS_PROMPT.format(task=task, data=data_json)

        insight = chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        )

        return {"summary_stats": summary, "insight": insight, "error": None}

    except Exception as exc:
        return {"summary_stats": {}, "insight": "", "error": str(exc)}