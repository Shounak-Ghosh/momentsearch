"""Document registration API — papers (Part 1) and decks (Part 2), mirroring
src/api/videos.py's shape one-for-one so the two source types share one
mental model.

Two ways to point at a source, same as video registration:
  * uri="https://..."   the worker downloads it (URL flow, paper's main path)
  * uri="storage://..." an object already in our bucket (e.g. a browser
                        upload placed at docs/{user_id}/...) — the worker
                        pulls it from there instead

POST /api/documents inserts a `pending` row and schedules a queue run (or
leaves it `pending` for the fair dispatcher — src/dispatcher.py), returning
202 BEFORE any parsing happens. That's the whole point of the queue: a 60-page
PDF must never be parsed inside this request.

Every request is tenant-scoped by the X-User-Id header, same as videos.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from .. import config, db, jobs, storage
from ..config import ALLOWED_DOC_EXTS, DOC_KEY_PREFIX, DOC_KINDS, MAX_DOC_MB
from ..ingest.fetch import doc_ext
from ..rag import vector_store
from .videos import require_auth, user_id

router = APIRouter(prefix="/api/documents", tags=["documents"])

# A real file extension is alphabetic — ".pdf", ".pptx". Path.suffix is purely
# textual, so it reads the trailing segment of an extensionless paper URL as an
# "extension" too: https://arxiv.org/pdf/2312.10997 -> ".10997", and
# .../2312.10997v2 -> ".10997v2". Those are arXiv ids, not formats, and
# rejecting them would 400 the single most common way to register a paper.
_EXT_RE = re.compile(r"^\.[A-Za-z]{1,8}$")


def _ext_of(value: str) -> str:
    """The file extension of a URL or storage key, ignoring a query string.
    Returns "" when there isn't one — including for a version-numbered paper
    URL, which fetch.doc_ext then handles by defaulting to .pdf."""
    suffix = Path(value.split("?", 1)[0]).suffix.lower()
    return suffix if _EXT_RE.match(suffix) else ""


# ── Presign (browser -> bucket upload, for a local .pdf/.pptx) ────────────────
# Same shape as src/api/videos.py's /presign: mint the doc_id + key up front,
# the browser PUTs the bytes straight to object storage, then POST /api/documents
# with uri="storage://<key>" (the register() branch above already handles
# storage:// uris — this just gives a local deck/paper a key to point at).

class PresignDocRequest(BaseModel):
    filename: str
    content_type: str
    size: int


@router.post("/presign", dependencies=[Depends(require_auth)])
def presign(req: PresignDocRequest, uid: str = Depends(user_id)):
    if req.size > MAX_DOC_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_DOC_MB}MB limit.")
    ext = Path(req.filename or "").suffix.lower()
    if ext not in ALLOWED_DOC_EXTS:
        raise HTTPException(415, f"Only {ALLOWED_DOC_EXTS} are accepted.")
    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    key = storage.doc_key(uid, doc_id, ext)
    if not storage.presign_capable():
        # local-dev fallback: the API accepts the bytes itself
        return {"mode": "direct", "doc_id": doc_id, "key": key,
                "url": f"/api/documents/{doc_id}/content?key={key}",
                "headers": {"Content-Type": req.content_type}}
    signed = storage.presign_put(key, req.content_type)
    return {"mode": "presigned", "doc_id": doc_id, "key": key, **signed}


@router.put("/{doc_id}/content", dependencies=[Depends(require_auth)])
async def upload_direct(doc_id: str, key: str, request: Request,
                        uid: str = Depends(user_id)):
    """Dev-only direct upload (STORAGE_PROVIDER=local can't presign)."""
    if storage.presign_capable():
        raise HTTPException(400, "Use the presigned URL to upload.")
    if not key.startswith(f"{DOC_KEY_PREFIX}{uid}/{doc_id}"):
        raise HTTPException(403, "Key does not belong to this upload.")
    dest = storage.local_path(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dest.open("wb") as out:
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_DOC_MB * 1024 * 1024:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds the {MAX_DOC_MB}MB limit.")
            out.write(chunk)
    return {"ok": True, "key": key, "size": size}


# ── Register (returns 202 instantly; a worker does the heavy lifting) ─────────

class RegisterDocument(BaseModel):
    uri: str
    kind: str = "paper"
    title: str | None = None


def _enqueue(doc_id: str, uid: str) -> str:
    """Schedule the queue run, turning an unreachable Prefect into a 502 (an
    upstream failure) rather than a bare 500. Only reached with fair dispatch
    OFF — with it on (the default) nothing talks to Prefect in the request
    path at all, which is what keeps accept latency inside the SLA."""
    try:
        return jobs.enqueue_document(doc_id, uid)
    except Exception as exc:
        raise HTTPException(502, f"Could not reach the work queue: {exc}") from exc


@router.post("", status_code=202, dependencies=[Depends(require_auth)])
def register(req: RegisterDocument, uid: str = Depends(user_id)):
    kind = (req.kind or "").strip().lower()
    if kind not in DOC_KINDS:
        raise HTTPException(400, f"kind must be one of {DOC_KINDS}.")

    uri = (req.uri or "").strip()
    doc_id = f"doc_{uuid.uuid4().hex[:10]}"

    if uri.startswith("http://") or uri.startswith("https://"):
        # A recognizable extension is required for an upload (the presigned
        # key's own suffix), but many paper URLs (arXiv-style) carry none —
        # only reject a URL whose extension IS present and unsupported (e.g.
        # a .docx), not one with no extension at all (fetch.doc_ext falls
        # back to .pdf for those, matching the paper flow's prior behavior).
        ext = _ext_of(uri)
        if ext and ext not in ALLOWED_DOC_EXTS:
            raise HTTPException(400, f"Unsupported file type {ext!r}; must be one of {ALLOWED_DOC_EXTS}.")
        row = db.upsert_pending_document({
            "id": doc_id, "user_id": uid, "kind": kind, "source": "url",
            "uri": uri, "storage_key": None, "source_hash": None,
            "title": req.title,
        })
    elif uri.startswith("storage://"):
        key = uri.removeprefix("storage://")
        # Never trust the client's key blindly: it must live under this
        # user's own document prefix (same ownership rule as video uploads).
        if not key.startswith(f"{DOC_KEY_PREFIX}{uid}/"):
            raise HTTPException(403, "Key does not belong to this user.")
        if _ext_of(key) not in ALLOWED_DOC_EXTS:
            raise HTTPException(400, f"Unsupported file type; must be one of {ALLOWED_DOC_EXTS}.")
        meta = storage.head(key)
        if meta is None:
            raise HTTPException(404, "Object not found — did the upload finish?")
        if meta["size"] > MAX_DOC_MB * 1024 * 1024:
            raise HTTPException(413, f"Object exceeds the {MAX_DOC_MB}MB limit.")
        row = db.upsert_pending_document({
            "id": doc_id, "user_id": uid, "kind": kind, "source": "upload",
            "uri": None, "storage_key": key, "source_hash": None,
            "title": req.title,
        })
    else:
        raise HTTPException(400, "uri must be an http(s) URL or a storage:// key you own.")

    # Fair dispatch (WFQ): leave it `pending` — the dispatcher admits it in
    # fair order alongside videos (src/dispatcher.py). FIFO mode: enqueue now.
    if config.ENABLE_FAIR_DISPATCH:
        return {"id": row["id"], "status": "pending", "kind": kind}
    flow_run_id = _enqueue(row["id"], uid)
    return {"id": row["id"], "status": row["status"], "kind": kind, "flow_run_id": flow_run_id}


# ── Status / lifecycle ─────────────────────────────────────────────────────────

_PUBLIC_FIELDS = ("id", "kind", "source", "uri", "title", "status", "error",
                  "page_count", "chunk_count", "progress", "attempts",
                  "created_at", "updated_at")


def _public(row: dict) -> dict:
    return {k: row.get(k) for k in _PUBLIC_FIELDS}


@router.get("")
def list_documents(uid: str = Depends(user_id), status: str | None = None):
    return {"documents": [_public(r) for r in db.list_documents(uid, status=status)]}


@router.get("/{doc_id}")
def get_document(doc_id: str, uid: str = Depends(user_id)):
    row = db.get_document(doc_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Document not found.")
    return _public(row)


# ── File (the archived raw PDF/PPTX, same-origin) ─────────────────────────────
# Citations for papers/decks embed the ORIGINAL external `url` too, but that
# never works for uploads (no public URL) and silently blanks under some
# hosts' X-Frame-Options (arxiv.org included) — the browser gives no JS error
# when that happens, it just renders an empty iframe. Serving our own archived
# copy same-origin sidesteps both: no cross-origin framing policy applies, and
# uploads have bytes to serve. An <iframe>/<embed src> can't attach an auth
# header, so — same pattern as the frame/slide/page/video endpoints in
# api/search.py — the tenant rides in the `u` query param, not X-User-Id.
_USER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _media_uid(value: str | None) -> str:
    uid = (value or config.DEFAULT_USER_ID).strip()
    if not _USER_RE.match(uid):
        raise HTTPException(400, "Invalid user id.")
    return uid


@router.get("/{doc_id}/file")
def file(doc_id: str, u: str | None = None):
    uid = _media_uid(u)
    row = db.get_document(doc_id)
    # storage_key is only set once t_fetch_doc has confirmed the archive
    # exists (written after a successful upload/reuse — see
    # ingest/document_pipeline.py's t_fetch_doc). A document mid-ingest, or
    # ingested before that write-back landed, has none: 404 here, and the
    # citation's file_url is already None for it (search.py's
    # _doc_file_url), so the UI falls through to the external `url` instead.
    if row is None or row["user_id"] != uid or not row.get("storage_key"):
        raise HTTPException(404, "Document not found.")
    if storage.presign_capable():
        return RedirectResponse(storage.presign_get(row["storage_key"]), status_code=307)
    path = storage.local_path(row["storage_key"])
    if not path.exists():
        raise HTTPException(404, "Document file not found.")
    media_type = ("application/pdf" if doc_ext(row) == ".pdf" else
                 "application/vnd.openxmlformats-officedocument.presentationml.presentation")
    return FileResponse(path, media_type=media_type,
                        headers={"Content-Disposition": "inline",
                                 "Cache-Control": "public, max-age=86400"})


@router.post("/{doc_id}/retry", status_code=202, dependencies=[Depends(require_auth)])
def retry(doc_id: str, uid: str = Depends(user_id)):
    row = db.get_document(doc_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Document not found.")
    db.set_doc_status(doc_id, "pending", error=None)
    if config.ENABLE_FAIR_DISPATCH:
        return {"id": doc_id, "status": "pending"}  # dispatcher re-admits it fairly
    flow_run_id = _enqueue(doc_id, uid)
    return {"id": doc_id, "status": "pending", "flow_run_id": flow_run_id}


@router.delete("/{doc_id}", dependencies=[Depends(require_auth)])
def delete(doc_id: str, uid: str = Depends(user_id)):
    """Deleting a document purges everything: vectors, the archived PDF, the
    cached parse, and the manifest row."""
    row = db.get_document(doc_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Document not found.")
    vector_store.delete_video(uid, doc_id)  # shared purge fn, keyed by video_id/doc_id
    # parsed.json, captions.json, and rendered slide thumbnails all live under
    # this same prefix (storage.slide_key/doc_parsed_key/doc_captions_key).
    storage.delete_prefix(storage.doc_prefix(uid, doc_id))
    storage.delete_key(storage.doc_key(uid, doc_id, doc_ext(row)))  # archived raw file
    if row.get("storage_key"):
        storage.delete_key(row["storage_key"])
    db.delete_document(doc_id)
    return {"ok": True, "id": doc_id}
