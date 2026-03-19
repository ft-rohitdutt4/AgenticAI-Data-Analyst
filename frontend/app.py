import json
import os
import time

import plotly.io as pio
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Agentic AI Data Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #aab;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .step-card {
        background: #1e1e2e;
        border-left: 4px solid #667eea;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #e0e0f0 !important;
    }
    .step-card * {
        color: #e0e0f0 !important;
    }
    .tool-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .answer-box {
        background: #1a1a2e;
        border: 1px solid #667eea88;
        border-radius: 10px;
        padding: 1.4rem 1.8rem;
        margin-top: 1rem;
        font-size: 1rem;
        line-height: 1.85;
        color: #e8e8f8 !important;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .answer-box * {
        color: #e8e8f8 !important;
    }
    .answer-box h3, .answer-box strong, .answer-box b {
        color: #a78bfa !important;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Helpers

TOOL_COLORS = {
    "sql_query": "#3b82f6",
    "pandas_analysis": "#10b981",
    "anomaly_detection": "#f59e0b",
    "visualization": "#8b5cf6",
    "final_answer": "#ef4444",
}


def fix_markdown(text: str) -> str:
    """
    Escape characters that Streamlit's markdown renderer misinterprets:
    - $ triggers LaTeX math mode  → escape to \\$
    - ~~ triggers strikethrough   → fine in newer Streamlit, leave it
    We only escape $ that are followed by a number or space+number
    (currency amounts), not $$ which is intentional LaTeX.
    """
    import re
    if not text:
        return text
    # Escape lone $ (currency) but not $$ (LaTeX block)
    # Replace $NUMBER patterns with \$NUMBER
    text = re.sub(r'\$(?!\$)(\d)', r'\\$\1', text)
    # Also escape $ at end of words like "$21,400" → already covered above
    # Escape remaining standalone $ not followed by another $
    text = re.sub(r'\$(?!\$)(?!\d)', r'\\$', text)
    return text

def tool_badge(tool: str) -> str:
    color = TOOL_COLORS.get(tool, "#888")
    return f'<span class="tool-badge" style="background:{color}22;color:{color}">{tool}</span>'


def check_api():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_tables():
    try:
        r = requests.get(f"{API_BASE}/tables", timeout=5)
        return r.json().get("tables", [])
    except Exception:
        return []


def get_schema():
    try:
        r = requests.get(f"{API_BASE}/schema", timeout=5)
        return r.json().get("schema", {})
    except Exception:
        return {}

# Sidebar
with st.sidebar:
    st.markdown("## 🤖 Agentic Data Analyst")

    # API status
    api_ok = check_api()
    if api_ok:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline — start the FastAPI server first")

    st.divider()

    #Upload CSV
    st.markdown("### 📂 Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    custom_table = st.text_input("Table name (optional)", placeholder="e.g. sales_2024")

    if st.button("Upload", use_container_width=True, disabled=not api_ok):
        if uploaded_file:
            with st.spinner("Uploading and ingesting..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                        data={"table_name": custom_table or ""},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"✅ Loaded **{data['rows_loaded']:,}** rows into `{data['table_name']}`"
                        )
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', 'Unknown error')}")
                except Exception as exc:
                    st.error(f"Request error: {exc}")
        else:
            st.warning("Please select a CSV file first.")

    st.divider()

    #Available Tables
    st.markdown("### 🗄️ Available Tables")
    tables = get_tables()
    if tables:
        for t in tables:
            st.markdown(f"• `{t}`")
    else:
        st.caption("No tables yet — upload a CSV above.")

    # Schema
    if tables:
        with st.expander("View Schema"):
            schema = get_schema()
            for table, cols in schema.items():
                st.markdown(f"**{table}**")
                st.caption(", ".join(cols))

    st.divider()
    st.markdown("### ⚙️ Settings")
    show_steps = st.toggle("Show reasoning steps", value=True)
    show_sql = st.toggle("Show generated SQL", value=False)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main area
st.markdown('<div class="main-header">🤖 Agentic AI Data Analyst</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask natural language questions — the agent plans, queries, analyses and explains.</div>',
    unsafe_allow_html=True,
)

#Example questions
example_questions = [
    "Why did revenue drop in March?",
    "Which product category has the highest sales?",
    "Show me monthly revenue trends",
    "Are there any anomalies in the sales data?",
    "Compare this quarter vs last quarter",
]

st.markdown("**💡 Try an example:**")
cols = st.columns(len(example_questions))
for i, q in enumerate(example_questions):
    if cols[i].button(q, key=f"eg_{i}", use_container_width=True):
        st.session_state["prefilled_question"] = q

# Chat history 
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            # Show steps if enabled
            if show_steps and msg.get("steps"):
                with st.expander(f"🔍 Reasoning steps ({len(msg['steps'])} steps)", expanded=False):
                    for step in msg["steps"]:
                        tool = step.get("tool", "")
                        step_name = step.get("step", "")
                        st.markdown(
                            f'<div class="step-card">{step_name} {tool_badge(tool)}</div>',
                            unsafe_allow_html=True,
                        )
                        if show_sql and tool == "sql_query":
                            sql = step.get("result", {}).get("sql", "")
                            if sql:
                                st.code(sql, language="sql")
                        if tool in ("pandas_analysis", "anomaly_detection"):
                            insight = step.get("result", {}).get("insight", "")
                            if insight:
                                st.info(insight)

            # Charts
            for chart_json in msg.get("charts", []):
                try:
                    fig = pio.from_json(chart_json)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

            # Final answer — render as native markdown inside a styled container
            st.markdown("---")
            st.markdown(fix_markdown(msg["content"]))

            # Metrics row
            if msg.get("iterations"):
                st.caption(f"🔄 {msg['iterations']} agent iterations | session: `{msg.get('session_id', '')[:8]}...`")

# Chat input 
prefilled = st.session_state.pop("prefilled_question", "")
user_input = st.chat_input("Ask a question about your data...", key="chat_input")
question = user_input or prefilled

if question and api_ok:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤖 Agent thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"question": question, "session_id": st.session_state.session_id},
                    timeout=120,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    final_answer = data.get("final_answer", "No answer generated.")
                    steps = data.get("steps", [])
                    charts = data.get("charts", [])
                    iterations = data.get("iterations", 0)
                    session_id = data.get("session_id", "")

                    # Render steps
                    if show_steps and steps:
                        with st.expander(f"🔍 Reasoning steps ({len(steps)} steps)", expanded=True):
                            for step in steps:
                                tool = step.get("tool", "")
                                step_name = step.get("step", "")
                                st.markdown(
                                    f'<div class="step-card">{step_name} {tool_badge(tool)}</div>',
                                    unsafe_allow_html=True,
                                )
                                if show_sql and tool == "sql_query":
                                    sql = step.get("result", {}).get("sql", "")
                                    if sql:
                                        st.code(sql, language="sql")
                                if tool in ("pandas_analysis", "anomaly_detection"):
                                    insight = step.get("result", {}).get("insight", "")
                                    if insight:
                                        st.info(insight)

                    # Render charts
                    for chart_json in charts:
                        try:
                            fig = pio.from_json(chart_json)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass

                    # Final answer — render as native markdown (handles $, ##, **, ~~ correctly)
                    st.markdown("---")
                    st.markdown(fix_markdown(final_answer))
                    st.caption(f"🔄 {iterations} iterations | session: `{session_id[:8]}...`")

                    # Store in history
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": final_answer,
                            "steps": steps,
                            "charts": charts,
                            "iterations": iterations,
                            "session_id": session_id,
                        }
                    )
                else:
                    err = resp.json().get("detail", "Unknown error")
                    st.error(f"❌ Error: {err}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The analysis may be taking too long.")
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")

elif question and not api_ok:
    st.error("❌ Cannot connect to the API server. Please start it first.")