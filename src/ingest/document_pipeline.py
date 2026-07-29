"""Per-document ingest pipeline — a Prefect flow of stage-tasks.

pending -> fetching -> parsing -> (captioning) -> chunking -> embedding -> indexed | skipped | failed

Stages:
  1. fetch    acquire the source file into worker scratch (URL download or
              bucket pull for uploads), hash it, skip duplicates. Archives the
              raw bytes to object storage so a retry never re-hits the origin.
  2. parse    kind="paper" -> pypdf page-by-page text; kind="deck" -> slide
              text/notes (PDF page-per-slide, or PPTX shapes). Cached to
              object storage (docs/{user}/{doc}/parsed.json) so a worker
              crash mid-ingest doesn't force a re-parse on resume.
  3. caption  DECK ONLY, no-op for papers. Slides with little/no extracted
              text (a chart or diagram drawn as vector shapes) are rendered
              to JPEG and captioned by the vision-capable LLM, so an
              image-only slide is still searchable. Cached to
              docs/{user}/{doc}/captions.json — this stage costs real LLM
              calls, so a resumed run must never re-pay for it.
  4. embed    kind="paper" -> chunk_pages() (never spans a page); kind="deck"
              -> chunk_slides() (never spans a slide, folds in captions) ->
              batched text embeddings -> idempotent upsert into the SAME
              Qdrant text collection videos' transcripts already live in.

Same orchestration shape as src/ingest/pipeline.py (video): Prefect Cloud
runs it, src/jobs.py triggers it, src/worker.py serves it. Each task carries
its own retry policy; Postgres (ms_documents) remains the business-status
source of truth. Both kinds ride the SAME task graph — papers pass through
t_caption as a no-op — so there is one status lifecycle and one Prefect UI
shape for both source types, not two flows.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
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
# imported below (fetch.py, paper.py, deck.py, db.py, ...) gets its OWN
# relative imports resolved normally, so only this entrypoint file needs the fix.
from src import db, storage
from src.config import (
    DECK_CAPTION_ENABLED,
    DECK_CAPTION_MAX_SLIDES,
    DOC_EMBED_BATCH,
    PAGE_RENDER_WIDTH,
    PAGE_THUMB_MAX_PAGES,
    PAGE_THUMB_WIDTH,
    TEXT_EMBED_VERSION,
)
from src.rag import vector_store
from src.rag.embeddings import embed_docs
from src.ingest import fetch as fetch_mod
from src.ingest import paper as paper_mod
from src.ingest import deck as deck_mod

_CAPTION_POOL = 3  # concurrent vision-LLM calls — keep low to avoid rate limits


@task(name="doc-fetch", retries=2, retry_delay_seconds=[30, 120])
def t_fetch_doc(doc_id: str, user_id: str) -> str:
    """Source file -> worker scratch file; duplicate check via source_hash.

    Returns "" when the content is a duplicate of an already-indexed document
    for this user (row marked 'skipped' — a plain outcome, not a retryable
    error), exactly like the video flow's t_fetch.
    """
    db.set_doc_status(doc_id, "fetching")
    row = db.get_document(doc_id)
    if row is None:
        raise ValueError(f"no manifest row for {doc_id}")

    ext = fetch_mod.doc_ext(row)
    archive_key = storage.doc_key(user_id, doc_id, ext)
    if storage.exists(archive_key):
        # A prior attempt already archived the raw bytes — reuse them instead
        # of re-hitting the origin URL on every retry/resume.
        path = storage.download_to(archive_key, fetch_mod.scratch_dir() / f"{doc_id}{ext}")
    elif row["source"] == "upload":
        path = fetch_mod.fetch_upload(row["storage_key"], doc_id)
    else:
        path = fetch_mod.fetch_document(row["uri"], doc_id, ext)

    source_hash = fetch_mod.sha256_file(path)
    db.set_doc_status(doc_id, "fetching", source_hash=source_hash)

    dup = db.find_duplicate_document(user_id, source_hash, exclude_id=doc_id)
    if dup:
        path.unlink(missing_ok=True)
        db.set_doc_status(doc_id, "skipped", error=f"duplicate of {dup['id']}")
        return ""

    if not storage.exists(archive_key):
        content_type = ("application/pdf" if ext == ".pdf" else
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation")
        storage.upload_file(path, archive_key, content_type)
    # Persist the archive location so citations/GET /api/documents/{id}/file
    # can serve these bytes back later. Written AFTER the archive is confirmed
    # to exist (freshly uploaded above, or already there from a prior attempt)
    # — never before — so a crash here just re-archives on retry instead of
    # pointing a citation at bytes that don't exist (crash-safe ordering,
    # AGENTS.md #3). For upload-sourced docs this is already set at
    # registration time (same key) — the write is a harmless no-op then.
    db.set_doc_status(doc_id, "fetching", storage_key=archive_key)
    return str(path)


@task(name="doc-parse", retries=2, retry_delay_seconds=60)
def t_parse(doc_id: str, user_id: str, path: str) -> dict:
    """File -> page/slide-numbered units. Cached so a resumed run (worker
    killed mid-ingest, flow re-entered) skips re-parsing a stage that already
    finished — the resilience gate cares about exactly this."""
    db.set_doc_status(doc_id, "parsing")
    row = db.get_document(doc_id) or {}
    kind = row.get("kind", "paper")
    cache_key = storage.doc_parsed_key(user_id, doc_id)

    cached = None
    if storage.exists(cache_key):
        try:
            payload = json.loads(storage.get_bytes(cache_key))
            if payload.get("source_hash") == row.get("source_hash") and payload.get("kind") == kind:
                cached = payload["units"]
        except Exception as exc:
            print(f"[doc] {doc_id}: cached parse unreadable ({exc}) — re-parsing")

    if cached is not None:
        print(f"[doc] {doc_id}: reusing cached parse ({len(cached)} units)")
        units = cached
    else:
        units = deck_mod.parse_deck(Path(path)) if kind == "deck" else paper_mod.parse_pdf(Path(path))
        storage.put_bytes(
            cache_key,
            json.dumps({"source_hash": row.get("source_hash"), "kind": kind, "units": units}).encode(),
            "application/json",
        )

    db.set_doc_status(doc_id, "parsing", page_count=len(units))
    title = (row.get("title") or "").strip()
    if not title:
        title = (deck_mod.guess_title(units, doc_id) if kind == "deck"
                 else paper_mod.guess_title(units, doc_id))
    return {"kind": kind, "units": units, "title": title}


@task(name="doc-caption", retries=2, retry_delay_seconds=60)
def t_caption(doc_id: str, user_id: str, path: str, parsed: dict) -> dict:
    """Deck-only: caption slides with little/no extracted text using the
    vision LLM, so an image-only chart/diagram slide is still searchable.
    No-op for papers (returns {} immediately) and when captioning is
    disabled or no model is configured — retrieval still works from slide
    text alone; this stage only makes it better.

    Cached to captions.json BEFORE returning, same rationale as t_parse's
    cache but higher stakes: this stage costs real LLM calls, and the
    resilience gate asserts a finished stage is never re-run.
    """
    if parsed["kind"] != "deck" or not DECK_CAPTION_ENABLED:
        return {}

    db.set_doc_status(doc_id, "captioning", progress=0.0)
    row = db.get_document(doc_id) or {}
    cache_key = storage.doc_captions_key(user_id, doc_id)

    if storage.exists(cache_key):
        try:
            payload = json.loads(storage.get_bytes(cache_key))
            if payload.get("source_hash") == row.get("source_hash"):
                print(f"[deck] {doc_id}: reusing {len(payload['captions'])} cached caption(s)")
                return {int(k): v for k, v in payload["captions"].items()}
        except Exception as exc:
            print(f"[deck] {doc_id}: cached captions unreadable ({exc}) — recaptioning")

    from src.rag.search import resolve_llm

    cfg, _source = resolve_llm(user_id)
    if cfg is None:
        print(f"[deck] {doc_id}: no model configured — skipping captions (text-only)")
        return {}

    from src import llm as llm_mod

    units = parsed["units"]
    todo = [u["slide"] for u in units if deck_mod.needs_caption(u)][:DECK_CAPTION_MAX_SLIDES]
    if not todo:
        storage.put_bytes(cache_key, json.dumps(
            {"source_hash": row.get("source_hash"), "captions": {}}).encode(), "application/json")
        return {}

    hints = {u["slide"]: u.get("text", "") for u in units}
    rendered = deck_mod.render_slides(Path(path), todo)

    captions: dict[int, str] = {}
    done = 0

    def _caption_one(slide: int) -> tuple[int, str | None]:
        jpeg = rendered.get(slide)
        if jpeg is None:
            return slide, None
        try:
            return slide, llm_mod.caption_image(cfg, jpeg, hint=hints.get(slide, ""))
        except Exception as exc:
            print(f"[deck] {doc_id}: slide {slide} caption failed "
                  f"({type(exc).__name__}: {exc}) — text-only for this slide")
            return slide, None

    with ThreadPoolExecutor(max_workers=_CAPTION_POOL) as ex:
        for slide, caption in ex.map(_caption_one, todo):
            if caption:
                captions[slide] = caption
                # Store a lighter thumbnail than the render fed to the vision
                # LLM — PAGE_RENDER_WIDTH is sized for caption quality,
                # PAGE_THUMB_WIDTH for what the citation UI actually displays.
                thumb = deck_mod.resize_jpeg(rendered[slide], PAGE_THUMB_WIDTH)
                storage.put_bytes(storage.slide_key(user_id, doc_id, slide), thumb, "image/jpeg")
            done += 1
            db.set_doc_progress(doc_id, done / len(todo))

    storage.put_bytes(
        cache_key,
        json.dumps({"source_hash": row.get("source_hash"), "captions": captions}).encode(),
        "application/json",
    )
    print(f"[deck] {doc_id}: captioned {len(captions)}/{len(todo)} slide(s)")
    return captions


def _render_citation_thumbnails(doc_id: str, user_id: str, path: str, kind: str,
                                 chunks: list[dict], captions: dict) -> None:
    """Render + store a page/slide thumbnail for every distinct locator that
    made it into the final chunk set, so every paper/deck citation has a real
    snippet to show — not just the deck slides t_caption already rendered
    (image-heavy ones it captioned). Capped at PAGE_THUMB_MAX_PAGES so a huge
    document doesn't rasterize hundreds of pages on first ingest; a citation
    past the cap just falls back to text in the UI. One page's render failure
    is logged and skipped inside render_pages/render_slides — never fails the
    whole ingest."""
    locator = "slide" if kind == "deck" else "page"
    wanted = sorted({c[locator] for c in chunks})[:PAGE_THUMB_MAX_PAGES]
    if not wanted:
        return
    if kind == "deck":
        already = set(captions.keys())  # t_caption already rendered+stored these
        todo = [p for p in wanted if p not in already]
        rendered = deck_mod.render_slides(Path(path), todo) if todo else {}
        key_fn = storage.slide_key
    else:
        rendered = paper_mod.render_pages(Path(path), wanted, width=PAGE_RENDER_WIDTH)
        key_fn = storage.page_key
    for p, jpeg in rendered.items():
        thumb = paper_mod.resize_jpeg(jpeg, PAGE_THUMB_WIDTH)
        storage.put_bytes(key_fn(user_id, doc_id, p), thumb, "image/jpeg")


@task(name="doc-embed-index", retries=2, retry_delay_seconds=60)
def t_embed_index_doc(doc_id: str, user_id: str, path: str, parsed: dict, captions: dict) -> int:
    """Page/slide-aware chunk -> batched text embeddings -> idempotent upsert
    into the shared text collection (moments_text) — same collection videos'
    transcripts and papers already index into."""
    db.set_doc_status(doc_id, "chunking")
    kind = parsed["kind"]
    if kind == "deck":
        chunks = deck_mod.chunk_slides(parsed["units"], captions)
    else:
        chunks = paper_mod.chunk_pages(parsed["units"])
    if not chunks:
        raise RuntimeError("No text could be extracted from the document.")

    _render_citation_thumbnails(doc_id, user_id, path, kind, chunks, captions)

    db.set_doc_status(doc_id, "embedding", progress=0.0, chunk_count=len(chunks))
    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, doc_id)  # drop stale points from a prior run

    total = 0
    for start in range(0, len(chunks), DOC_EMBED_BATCH):
        batch = chunks[start:start + DOC_EMBED_BATCH]
        vectors = embed_docs([c["text"] for c in batch])
        if kind == "deck":
            payloads = [{"user_id": user_id, "video_id": doc_id, "source_id": doc_id,
                        "kind": "deck", "modality": "text",
                        "slide": c["slide"], "slide_end": c["slide_end"], "idx": c["idx"],
                        "text": c["text"], "title": parsed["title"],
                        "embed_version": TEXT_EMBED_VERSION}
                       for c in batch]
        else:
            payloads = [{"user_id": user_id, "video_id": doc_id, "source_id": doc_id,
                        "kind": "paper", "modality": "text",
                        "page": c["page"], "page_end": c["page_end"], "idx": c["idx"],
                        "text": c["text"], "title": parsed["title"],
                        "embed_version": TEXT_EMBED_VERSION}
                       for c in batch]
        vector_store.upsert_doc_chunks(user_id, doc_id, vectors, payloads=payloads, start_idx=start)
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
        captions = t_caption(doc_id, user_id, path, parsed)  # no-op for papers
        n = t_embed_index_doc(doc_id, user_id, path, parsed, captions)
        print(f"[ingest] {doc_id} indexed: {n} chunks across {len(parsed['units'])} "
              f"{'slides' if parsed['kind'] == 'deck' else 'pages'} (attempt {attempt})")
        return {"doc_id": doc_id, "chunks": n, "units": len(parsed["units"])}
    except Exception as exc:
        db.set_doc_status(doc_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise  # Prefect marks the run Failed; full trace in the Cloud UI
    finally:
        if path:  # scratch only — durable copies live in object storage
            Path(path).unlink(missing_ok=True)
