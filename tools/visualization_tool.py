import json
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _detect_chart_type(df: pd.DataFrame, task: str) -> str:
    """Heuristic to pick the best chart type."""
    task_lower = task.lower()
    if any(w in task_lower for w in ["trend", "over time", "monthly", "daily", "weekly", "yearly"]):
        return "line"
    if any(w in task_lower for w in ["compare", "breakdown", "distribution", "category"]):
        return "bar"
    if any(w in task_lower for w in ["share", "proportion", "percent", "pie"]):
        return "pie"
    if any(w in task_lower for w in ["scatter", "correlation", "relationship"]):
        return "scatter"

    # Default heuristic: date/text x + numeric y → line/bar
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        return "bar"
    return "table"


def run_visualization_tool(
    task: str,
    data: list[dict],
    chart_type: Optional[str] = None,
) -> dict:
    """
    Build a Plotly figure from data rows.

    Returns
    -------
    {
        "chart_json": str,   # plotly figure as JSON string
        "chart_type": str,
        "error": str | None,
    }
    """
    if not data:
        return {"chart_json": None, "chart_type": None, "error": "No data to visualise."}

    try:
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        text_cols = df.select_dtypes(exclude="number").columns.tolist()

        if not chart_type:
            chart_type = _detect_chart_type(df, task)

        fig = None

        if chart_type == "line" and text_cols and numeric_cols:
            fig = px.line(
                df,
                x=text_cols[0],
                y=numeric_cols[0],
                title=task,
                markers=True,
                template="plotly_white",
            )

        elif chart_type == "bar" and text_cols and numeric_cols:
            fig = px.bar(
                df,
                x=text_cols[0],
                y=numeric_cols[0],
                title=task,
                template="plotly_white",
                color=numeric_cols[0],
                color_continuous_scale="Blues",
            )

        elif chart_type == "pie" and text_cols and numeric_cols:
            fig = px.pie(
                df,
                names=text_cols[0],
                values=numeric_cols[0],
                title=task,
                template="plotly_white",
            )

        elif chart_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=task,
                template="plotly_white",
            )

        else:
            # Fallback: table
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(values=list(df.columns), fill_color="#1f77b4", font=dict(color="white")),
                        cells=dict(values=[df[c].tolist() for c in df.columns]),
                    )
                ]
            )
            fig.update_layout(title=task)

        fig.update_layout(
            margin=dict(l=40, r=40, t=60, b=40),
            font=dict(size=13),
        )

        return {
            "chart_json": fig.to_json(),
            "chart_type": chart_type,
            "error": None,
        }

    except Exception as exc:
        return {"chart_json": None, "chart_type": None, "error": str(exc)}