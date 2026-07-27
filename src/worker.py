"""Ingest worker entrypoint — serves both Prefect flows.

    python -m src.worker

prefect.serve() registers the "ms-ingest-video/ingest" AND
"ms-ingest-document/ingest-doc" deployments in Prefect Cloud (idempotent) and
long-polls for scheduled runs on both — outbound HTTPS only, no ports. Scale
horizontally by running more replicas of this process; each executes up to
WORKER_CONCURRENCY runs (of EITHER flow) at once.

(A single flow's `.serve()` blocks the process, so it can only ever serve one
deployment; `prefect.serve()` is the multi-deployment form — same polling
loop, both flows share the worker's concurrency slots.)

Sample seeding is NOT done here — it's a one-shot startup gate (seed.py /
src/seeding.py) that the whole stack waits on, so the app never serves a
half-indexed corpus. This worker only handles user uploads + YouTube adds +
document ingests.

Embedding goes to the warm CLIP service when CLIP_SERVICE_URL is set
(docker-compose default); unset, each run loads the model in-process.
"""
import os
import time

from . import config
from .db import init_schema
from .ingest.document_pipeline import ingest_document
from .ingest.pipeline import ingest_video


def main():
    init_schema()  # make sure migrations ran before consuming runs
    from .rag import vector_store
    vector_store.ensure_collection()  # up front, not mid-first-ingest
    if config.ENABLE_TRANSCRIPT or config.ENABLE_DOCUMENTS:
        vector_store.ensure_text_collection()
    # Fair scheduler (WFQ): admits pending videos/documents round-robin across
    # users so one bulk uploader can't starve everyone else (src/dispatcher.py).
    from . import dispatcher
    dispatcher.start_in_background()
    limit = int(os.getenv("WORKER_CONCURRENCY", "2"))
    # serve() talks to Prefect Cloud on startup; a transient outage (e.g. a 503)
    # used to crash the worker permanently and stop the machine. Self-heal:
    # retry forever so a blip pauses ingest instead of killing the worker.
    while True:
        try:
            print(f"[worker] serving 'ms-ingest-video/ingest' + "
                  f"'ms-ingest-document/ingest-doc' (concurrency {limit})")
            from prefect import serve

            serve(
                ingest_video.to_deployment(name="ingest"),          # name UNCHANGED
                ingest_document.to_deployment(name="ingest-doc"),
                limit=limit,
            )
            break  # clean shutdown
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[worker] serve crashed: {type(exc).__name__}: {exc} — retrying in 15s")
            time.sleep(15)


if __name__ == "__main__":
    main()
