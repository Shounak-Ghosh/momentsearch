"""Admin surface — the assignment's contract, mounted over the existing routers.

`/api/videos/*` and `/api/documents/*` are the app's real write paths and stay
exactly as they are (non-negotiable #6: the provided video endpoints keep
working). This router is a THIN alias layer that publishes the shapes the
assignment specifies — `POST /admin/documents`, `POST /admin/videos`,
`GET /admin/sources` — by delegating straight to those handlers. It contains no
ingest logic of its own; the only genuinely new thing here is /admin/sources,
which unifies both manifests into one status view.

Why an alias rather than a rename: the UI, the seed path and every existing
integration call /api/*. Renaming would break them for cosmetic gain; delegating
costs one function call and keeps both contracts true at once.

Auth mirrors the routers it fronts: writes require the Bearer admin token, the
read (`GET /admin/sources`) is tenant-scoped by X-User-Id only — same as the
existing `GET /api/videos`, which the unauthenticated UI polls.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import db
from ..samples import is_sample
from . import documents as documents_api
from . import videos as videos_api
from .videos import require_auth, user_id

router = APIRouter(prefix="/admin", tags=["admin"])


# ── Overall completion percent ────────────────────────────────────────────────
# `progress` in Postgres is 0..1 WITHIN the current stage — it resets to 0 each
# time the pipeline moves on, so it can't be shown as an overall figure. These
# ladders map (status, progress) onto one 0..100 number: each stage owns a slice
# of the bar, and the stage's own progress fills that slice.
#
# The stage names are the literal strings the pipelines write via
# db.set_status / db.set_doc_status — keep these tables in sync with
# src/ingest/pipeline.py and src/ingest/document_pipeline.py.

#                     status: (base, span)
_VIDEO_LADDER = {
    "pending":   (0, 0),
    "queued":    (5, 0),
    "fetching":  (10, 0),
    "sampling":  (20, 30),   # 20 -> 50
    "embedding": (50, 45),   # 50 -> 95
    "indexed":   (100, 0),
    "skipped":   (100, 0),   # duplicate — nothing left to do
}

_DOC_LADDER = {
    "pending":    (0, 0),
    "queued":     (5, 0),
    "fetching":   (10, 0),
    "parsing":    (20, 0),
    "captioning": (30, 25),  # 30 -> 55 (decks only; papers skip it)
    "chunking":   (58, 0),
    "embedding":  (60, 38),  # 60 -> 98
    "indexed":    (100, 0),
    "skipped":    (100, 0),
}


def pct_of(kind: str, status: str, progress: float | None) -> int:
    """Overall 0..100 completion for one source.

    `failed` reports how far it had got before dying (0 — the row's status has
    already been overwritten with 'failed', so the stage it died in is gone);
    an unrecognized status degrades to 0 rather than raising, so adding a stage
    to a pipeline can never 500 this endpoint.
    """
    ladder = _VIDEO_LADDER if kind == "video" else _DOC_LADDER
    base, span = ladder.get(status, (0, 0))
    return int(round(base + span * (progress or 0.0)))


_SOURCE_FIELDS = ("id", "kind", "source", "uri", "title", "status", "error",
                  "unit_count", "chunk_count", "progress", "attempts",
                  "created_at", "updated_at")


def _public(row: dict) -> dict:
    out = {k: row.get(k) for k in _SOURCE_FIELDS}
    out["pct"] = pct_of(row["kind"], row["status"], row.get("progress"))
    # Samples are protected: selectable-yes, deletable-no. Only videos are ever
    # samples; the UI reads this to hide the ✕.
    out["is_sample"] = is_sample(row["id"]) if row["kind"] == "video" else False
    return out


def _is_document(source_id: str) -> bool:
    """Which manifest owns this id. Prefixes don't collide: documents mint
    doc_<hex>, videos yt_<id> / up_<hex> (see the two register handlers)."""
    return source_id.startswith("doc_")


# ── Register (both types, async — 202 before any parsing) ────────────────────

class RegisterDocumentRequest(BaseModel):
    uri: str
    kind: str = "paper"
    title: str | None = None


@router.post("/documents", status_code=202, dependencies=[Depends(require_auth)])
def register_document(req: RegisterDocumentRequest, uid: str = Depends(user_id)):
    """Insert a pending row + schedule a queue run, then return. Parsing,
    chunking, captioning and embedding all happen on a worker — never here."""
    return documents_api.register(
        documents_api.RegisterDocument(uri=req.uri, kind=req.kind, title=req.title), uid)


class RegisterVideoRequest(BaseModel):
    url: str | None = None
    video_id: str | None = None
    key: str | None = None
    title: str | None = None
    speaker: str | None = None   # the assignment's field name for a talk's title


@router.post("/videos", status_code=202, dependencies=[Depends(require_auth)])
def register_video(req: RegisterVideoRequest, uid: str = Depends(user_id)):
    out = videos_api.register(
        videos_api.RegisterRequest(url=req.url, video_id=req.video_id, key=req.key,
                                   title=req.title or req.speaker), uid)
    # The contract asks for `id`; /api/videos answers `video_id`. Return both so
    # either client reads it correctly.
    return {"id": out["video_id"], **out}


# ── Unified status (the one genuinely new endpoint) ──────────────────────────

@router.get("/sources")
def list_sources(uid: str = Depends(user_id),
                 status: str | None = None, kind: str | None = None):
    """Videos + documents in one list, newest first, each with its `kind` and an
    overall `pct`. This is what the library UI polls while an ingest runs."""
    return {"sources": [_public(r) for r in db.list_sources(uid, status=status, kind=kind)]}


@router.get("/sources/{source_id}")
def get_source(source_id: str, uid: str = Depends(user_id)):
    row = db.get_source(uid, source_id)
    if row is None:
        raise HTTPException(404, "Source not found.")
    return _public(row)


@router.post("/sources/{source_id}/retry", status_code=202,
             dependencies=[Depends(require_auth)])
def retry_source(source_id: str, uid: str = Depends(user_id)):
    if _is_document(source_id):
        return documents_api.retry(source_id, uid)
    return videos_api.retry(source_id, uid)


@router.delete("/sources/{source_id}", dependencies=[Depends(require_auth)])
def delete_source(source_id: str, uid: str = Depends(user_id)):
    if _is_document(source_id):
        return documents_api.delete(source_id, uid)
    return videos_api.delete(source_id, uid)
