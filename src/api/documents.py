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

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import config, db, jobs, storage
from ..config import DOC_KEY_PREFIX, DOC_KINDS, MAX_DOC_MB
from ..rag import vector_store
from .videos import require_auth, user_id

router = APIRouter(prefix="/api/documents", tags=["documents"])


# ── Register (returns 202 instantly; a worker does the heavy lifting) ─────────

class RegisterDocument(BaseModel):
    uri: str
    kind: str = "paper"
    title: str | None = None


@router.post("", status_code=202, dependencies=[Depends(require_auth)])
def register(req: RegisterDocument, uid: str = Depends(user_id)):
    kind = (req.kind or "").strip().lower()
    if kind not in DOC_KINDS:
        raise HTTPException(400, f"kind must be one of {DOC_KINDS}.")

    uri = (req.uri or "").strip()
    doc_id = f"doc_{uuid.uuid4().hex[:10]}"

    if uri.startswith("http://") or uri.startswith("https://"):
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
    flow_run_id = jobs.enqueue_document(row["id"], uid)
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


@router.post("/{doc_id}/retry", status_code=202, dependencies=[Depends(require_auth)])
def retry(doc_id: str, uid: str = Depends(user_id)):
    row = db.get_document(doc_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Document not found.")
    db.set_doc_status(doc_id, "pending", error=None)
    if config.ENABLE_FAIR_DISPATCH:
        return {"id": doc_id, "status": "pending"}  # dispatcher re-admits it fairly
    flow_run_id = jobs.enqueue_document(doc_id, uid)
    return {"id": doc_id, "status": "pending", "flow_run_id": flow_run_id}


@router.delete("/{doc_id}", dependencies=[Depends(require_auth)])
def delete(doc_id: str, uid: str = Depends(user_id)):
    """Deleting a document purges everything: vectors, the archived PDF, the
    cached parse, and the manifest row."""
    row = db.get_document(doc_id)
    if row is None or row["user_id"] != uid:
        raise HTTPException(404, "Document not found.")
    vector_store.delete_video(uid, doc_id)  # shared purge fn, keyed by video_id/doc_id
    storage.delete_prefix(storage.doc_prefix(uid, doc_id))  # parsed.json cache
    storage.delete_key(storage.doc_key(uid, doc_id))         # archived raw PDF
    if row.get("storage_key"):
        storage.delete_key(row["storage_key"])
    db.delete_document(doc_id)
    return {"ok": True, "id": doc_id}
