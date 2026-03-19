import json

import numpy as np
import pandas as pd

from llm.client import chat
from llm.prompts import ANALYST_SYSTEM
from database.connection import get_schema_text


def _zscore_anomalies(series: pd.Series, threshold: float = 2.5) -> pd.Series:
    """Return boolean mask of anomalies by Z-score."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    z = (series - mean).abs() / std
    return z > threshold


def _iqr_anomalies(series: pd.Series) -> pd.Series:
    """Return boolean mask of anomalies by IQR fences."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def run_anomaly_tool(task: str, data: list[dict]) -> dict:
    """
    Detect anomalies in numeric columns and return an LLM-generated explanation.

    Returns
    -------
    {
        "anomalies": list[dict],   # rows that are anomalous
        "explanation": str,
        "error": str | None,
    }
    """
    if not data:
        return {"anomalies": [], "explanation": "No data provided.", "error": None}

    try:
        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            return {
                "anomalies": [],
                "explanation": "No numeric columns found for anomaly detection.",
                "error": None,
            }

        anomaly_mask = pd.Series([False] * len(df), index=df.index)

        for col in numeric_cols:
            z_mask = _zscore_anomalies(df[col])
            iqr_mask = _iqr_anomalies(df[col])
            anomaly_mask = anomaly_mask | (z_mask & iqr_mask)

        anomaly_df = df[anomaly_mask]
        anomalies = anomaly_df.to_dict(orient="records")

        # Build LLM explanation
        schema = get_schema_text()
        system_msg = ANALYST_SYSTEM.format(schema=schema)

        if anomalies:
            data_str = json.dumps(anomalies[:20], default=str, indent=2)
            user_msg = (
                f"TASK: {task}\n\n"
                f"The following rows were flagged as statistical anomalies "
                f"(both Z-score > 2.5 and outside IQR fences):\n{data_str}\n\n"
                "Explain what these anomalies mean in plain business language. "
                "What could have caused them? What action should be taken?"
            )
        else:
            user_msg = (
                f"TASK: {task}\n\n"
                "No statistical anomalies were detected in the dataset. "
                "Confirm this finding and advise what that means."
            )

        explanation = chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        )

        return {"anomalies": anomalies, "explanation": explanation, "error": None}

    except Exception as exc:
        return {"anomalies": [], "explanation": "", "error": str(exc)}