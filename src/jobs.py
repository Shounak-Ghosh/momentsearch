"""Prefect Cloud trigger layer — the API schedules runs, workers execute them.

Two flows share the workspace ("ms-" prefix keeps both distinct from the
digital-twin-akash flow living alongside them): "ms-ingest-video" (deployment
"ingest") and "ms-ingest-document" (deployment "ingest-doc"), both served by
worker.py's serve() call. The API never imports either pipeline or its heavy
deps (torch, ffmpeg, pypdf) — it just asks Prefect Cloud to schedule a run;
any live worker picks it up. Retries/backoff live on each flow's tasks
(src/ingest/pipeline.py, src/ingest/document_pipeline.py); failed runs are
visible + retryable in the Prefect Cloud UI.
"""
from __future__ import annotations

from prefect.deployments import run_deployment

INGEST_DEPLOYMENT = "ms-ingest-video/ingest"
DOCUMENT_DEPLOYMENT = "ms-ingest-document/ingest-doc"


def enqueue_video(video_id: str, user_id: str) -> str:
    """Schedule the ingest flow for one video. Returns the Prefect flow-run id."""
    flow_run = run_deployment(
        name=INGEST_DEPLOYMENT,
        parameters={"video_id": video_id, "user_id": user_id},
        timeout=0,  # fire-and-forget: don't block the API waiting for the run
        flow_run_name=f"ingest-{video_id}",
    )
    return str(flow_run.id)


def enqueue_document(doc_id: str, user_id: str) -> str:
    """Schedule the ingest flow for one document (paper/deck). Returns the
    Prefect flow-run id."""
    flow_run = run_deployment(
        name=DOCUMENT_DEPLOYMENT,
        parameters={"doc_id": doc_id, "user_id": user_id},
        timeout=0,
        flow_run_name=f"ingest-{doc_id}",
    )
    return str(flow_run.id)
