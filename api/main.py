import json
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ingestion.ingestor import ingest_csv, list_tables
from database.connection import get_schema_text, get_schema
from agent.loop import run_agent, run_agent_streaming

app = FastAPI(
    title="Agentic AI Data Analyst",
    description="Upload CSVs, ask natural language questions, get agentic analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema_endpoint():
    return {"schema": get_schema()}


@app.get("/tables")
def tables_endpoint():
    return {"tables": list_tables()}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    table_name: Optional[str] = Form(None),
):
    """Upload a CSV file. Auto-creates a MySQL table."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    default_name = file.filename.replace(".csv", "").replace(" ", "_").lower()
    name = table_name or default_name

    try:
        result = ingest_csv(content, table_name=name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return {
        "message": "File uploaded and stored successfully.",
        "table_name": result["table_name"],
        "rows_loaded": result["rows"],
        "columns": result["columns"],
    }


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


@app.post("/query")
def query(req: QueryRequest):
    """Run the full agentic analysis loop synchronously."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    schema_text = get_schema_text()
    if schema_text == "No tables found in the database.":
        raise HTTPException(
            status_code=400,
            detail="No tables found. Please upload a CSV file first.",
        )

    try:
        # run_agent only accepts: question, session_id
        # Schema is fetched internally by the agent — do NOT pass it here
        result = run_agent(
            question=req.question,
            session_id=req.session_id or str(uuid.uuid4()),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    return result


@app.post("/analyze")
def analyze(req: QueryRequest):
    """Run the agentic loop with Server-Sent Events streaming."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    schema_text = get_schema_text()
    if schema_text == "No tables found in the database.":
        raise HTTPException(
            status_code=400,
            detail="No tables found. Please upload a CSV file first.",
        )

    def event_stream():
        # run_agent_streaming only accepts: question, session_id
        # Schema is fetched internally by the agent — do NOT pass it here
        for event in run_agent_streaming(
            question=req.question,
            session_id=req.session_id or str(uuid.uuid4()),
        ):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")