"""Per-document ingest pipeline — a Prefect flow of three stage-tasks.

pending -> fetching -> parsing -> chunking -> embedding -> indexed | skipped | failed

Stages:
  1. fetch   acquire the source PDF into worker scratch (URL download or
             bucket pull for uploads), hash it, skip duplicates. Archives the
             raw bytes to object storage so a retry never re-hits the origin.
  2. parse   pypdf page-by-page text extraction. Cached to object storage
             (docs/{user}/{doc}/parsed.json) so a worker crash mid-ingest
             doesn't force a 60-page PDF to be re-parsed on resume.
  3. embed   chunk_pages() (page-aware — a chunk never spans two pages) ->
             batched text embeddings -> idempotent upsert into the SAME
             Qdrant text collection videos' transcripts already live in.

Same orchestration shape as src/ingest/pipeline.py (video): Prefect Cloud
runs it, src/jobs.py triggers it, src/worker.py serves it. Each task carries
its own retry policy; Postgres (ms_documents) remains the business-status
source of truth. kind currently only branches to "paper" — deck ingestion
(Part 2) joins this same flow.
"""
from __future__ import annotations

import json
from pathlib import Path

from prefect import flow, task

# Absolute imports, deliberately — NOT `from .. import db, storage`. Prefect
# Cloud reloads a scheduled run from this file's entrypoint path
# ("src/ingest/document_pipeline.py:ingest_document"), executing it as a
# standalone script with no parent package, where a `from ..` import fails
# with "attempted relative import beyond top-level package". Absolute imports
# rooted at `src` resolve correctly either way (normal package import from
# worker.py, or Prefect's fresh entrypoint reload) — and once `src` is
# imported through the real package machinery, everything transitively
# imported below (fetch.py, paper.py, db.py, ...) gets its OWN relative
# imports resolved normally, so only this entrypoint file needs the fix.
from src import db, storage
from src.config import DOC_EMBED_BATCH, TEXT_EMBED_VERSION
from src.rag import vector_store
from src.rag.embeddings import embed_docs
from src.ingest import fetch as fetch_mod
from src.ingest import paper as paper_mod


@task(name="doc-fetch", retries=2, retry_delay_seconds=[30, 120])
def t_fetch_doc(doc_id: str, user_id: str) -> str:
    """Source PDF -> worker scratch file; duplicate check via source_hash.

    Returns "" when the content is a duplicate of an already-indexed document
    for this user (row marked 'skipped' — a plain outcome, not a retryable
    error), exactly like the video flow's t_fetch.
    """
    db.set_doc_status(doc_id, "fetching")
    row = db.get_document(doc_id)
    if row is None:
        raise ValueError(f"no manifest row for {doc_id}")

    archive_key = storage.doc_key(user_id, doc_id)
    if storage.exists(archive_key):
        # A prior attempt already archived the raw bytes — reuse them instead
        # of re-hitting the origin URL on every retry/resume.
        path = storage.download_to(archive_key, fetch_mod.scratch_dir() / f"{doc_id}.pdf")
    elif row["source"] == "upload":
        path = fetch_mod.fetch_upload(row["storage_key"], doc_id)
    else:
        path = fetch_mod.fetch_pdf(row["uri"], doc_id)

    source_hash = fetch_mod.sha256_file(path)
    db.set_doc_status(doc_id, "fetching", source_hash=source_hash)

    dup = db.find_duplicate_document(user_id, source_hash, exclude_id=doc_id)
    if dup:
        path.unlink(missing_ok=True)
        db.set_doc_status(doc_id, "skipped", error=f"duplicate of {dup['id']}")
        return ""

    if not storage.exists(archive_key):
        storage.upload_file(path, archive_key, "application/pdf")
    return str(path)


@task(name="doc-parse", retries=2, retry_delay_seconds=60)
def t_parse(doc_id: str, user_id: str, path: str) -> dict:
    """PDF -> page-numbered text. Cached so a resumed run (worker killed
    mid-ingest, flow re-entered) skips re-parsing a stage that already
    finished — the resilience gate cares about exactly this."""
    db.set_doc_status(doc_id, "parsing")
    row = db.get_document(doc_id) or {}
    cache_key = storage.doc_parsed_key(user_id, doc_id)

    cached = None
    if storage.exists(cache_key):
        try:
            payload = json.loads(storage.get_bytes(cache_key))
            if payload.get("source_hash") == row.get("source_hash"):
                cached = payload["pages"]
        except Exception as exc:
            print(f"[paper] {doc_id}: cached parse unreadable ({exc}) — re-parsing")

    if cached is not None:
        print(f"[paper] {doc_id}: reusing cached parse ({len(cached)} pages)")
        pages = cached
    else:
        pages = paper_mod.parse_pdf(Path(path))
        storage.put_bytes(
            cache_key,
            json.dumps({"source_hash": row.get("source_hash"), "pages": pages}).encode(),
            "application/json",
        )

    db.set_doc_status(doc_id, "parsing", page_count=len(pages))
    title = (row.get("title") or "").strip() or paper_mod.guess_title(pages, doc_id)
    return {"pages": pages, "title": title}


@task(name="doc-embed-index", retries=2, retry_delay_seconds=60)
def t_embed_index_doc(doc_id: str, user_id: str, parsed: dict) -> int:
    """Page-aware chunk -> batched text embeddings -> idempotent upsert into
    the shared text collection (moments_text) — same collection videos'
    transcripts already index into."""
    db.set_doc_status(doc_id, "chunking")
    chunks = paper_mod.chunk_pages(parsed["pages"])
    if not chunks:
        raise RuntimeError("No text could be extracted from the document.")

    db.set_doc_status(doc_id, "embedding", progress=0.0, chunk_count=len(chunks))
    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, doc_id)  # drop stale points from a prior run

    total = 0
    for start in range(0, len(chunks), DOC_EMBED_BATCH):
        batch = chunks[start:start + DOC_EMBED_BATCH]
        vectors = embed_docs([c["text"] for c in batch])
        vector_store.upsert_doc_chunks(
            user_id, doc_id, vectors,
            payloads=[{"user_id": user_id, "video_id": doc_id, "source_id": doc_id,
                       "kind": "paper", "modality": "text",
                       "page": c["page"], "page_end": c["page_end"], "idx": c["idx"],
                       "text": c["text"], "title": parsed["title"],
                       "embed_version": TEXT_EMBED_VERSION}
                      for c in batch],
            start_idx=start,
        )
        total += len(batch)
        db.set_doc_progress(doc_id, total / len(chunks))

    # Crash-safe ordering (non-negotiable #3): 'indexed' is written ONLY after
    # every upsert above has returned (wait=True) — a worker killed before
    # this line leaves the row in 'embedding', not falsely 'indexed'.
    db.set_doc_status(doc_id, "indexed", chunk_count=total,
                      embed_version=TEXT_EMBED_VERSION, progress=1.0)
    return total


@flow(name="ms-ingest-document", log_prints=True, timeout_seconds=3600)
def ingest_document(doc_id: str, user_id: str) -> dict:
    attempt = db.bump_doc_attempts(doc_id)
    path: str | None = None
    try:
        path = t_fetch_doc(doc_id, user_id)
        if not path:  # duplicate — already marked 'skipped' by t_fetch_doc
            print(f"[ingest] {doc_id} skipped (duplicate content)")
            return {"doc_id": doc_id, "skipped": True}
        parsed = t_parse(doc_id, user_id, path)
        n = t_embed_index_doc(doc_id, user_id, parsed)
        print(f"[ingest] {doc_id} indexed: {n} chunks across {len(parsed['pages'])} "
              f"pages (attempt {attempt})")
        return {"doc_id": doc_id, "chunks": n, "pages": len(parsed["pages"])}
    except Exception as exc:
        db.set_doc_status(doc_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise  # Prefect marks the run Failed; full trace in the Cloud UI
    finally:
        if path:  # scratch only — durable copies live in object storage
            Path(path).unlink(missing_ok=True)
