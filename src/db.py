"""Postgres (Neon) access layer — the videos manifest, source of truth.

One row per (user's) video; `status` tracks the ingest lifecycle:
pending -> fetching -> sampling -> embedding -> indexed | skipped | failed
(skipped = duplicate (user_id, source_hash); indexed = searchable in Qdrant).
"""
from __future__ import annotations

import os
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import DATABASE_URL, INFLIGHT_STATUSES

_pool: ConnectionPool | None = None
_pool_pid: int | None = None


def pool() -> ConnectionPool:
    """Process-local pool. Prefect runs flows in subprocesses; a child must
    never reuse the parent's SSL connections (corrupts the TLS stream), so a
    fork gets a fresh pool."""
    global _pool, _pool_pid
    if _pool is None or _pool_pid != os.getpid():
        # check= pings each connection before lending it out — Neon silently
        # drops idle SSL connections, which otherwise 500s the first request
        # after a quiet period.
        _pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5,
                               check=ConnectionPool.check_connection,
                               kwargs={"row_factory": dict_row})
        _pool_pid = os.getpid()
    return _pool


SCHEMA = """
CREATE TABLE IF NOT EXISTS ms_videos (
    id           TEXT PRIMARY KEY,           -- yt_<id> | up_<uuid>
    user_id      TEXT NOT NULL,
    source       TEXT NOT NULL,              -- youtube | upload
    url          TEXT,                       -- YouTube URL (source=youtube)
    storage_key  TEXT,                       -- uploads/<user>/<id>.<ext> (source=upload)
    source_hash  TEXT,                       -- sha256 of the file / yt video id
    title        TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    error        TEXT,
    frame_count  INT,
    progress     REAL,                       -- 0..1 within the current stage
    attempts     INT NOT NULL DEFAULT 0,
    embed_version TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ms_videos_user_idx   ON ms_videos (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ms_videos_status_idx ON ms_videos (status);
CREATE INDEX IF NOT EXISTS ms_videos_hash_idx   ON ms_videos (user_id, source_hash);

-- Documents (papers, decks) — the second source type. One row per document;
-- `status` tracks the SAME shape of lifecycle as ms_videos, with two extra
-- document-only stages (parsing, chunking) in front of embedding.
CREATE TABLE IF NOT EXISTS ms_documents (
    id           TEXT PRIMARY KEY,           -- doc_<hex10>
    user_id      TEXT NOT NULL,
    kind         TEXT NOT NULL,              -- paper | deck
    source       TEXT NOT NULL,              -- url | upload
    uri          TEXT,                       -- https://... (source=url)
    storage_key  TEXT,                       -- docs/{user}/{doc_id}.pdf (source=upload,
                                              -- and also where a url source is archived)
    source_hash  TEXT,                       -- sha256 of the PDF bytes
    title        TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    error        TEXT,
    page_count   INT,
    chunk_count  INT,
    progress     REAL,                       -- 0..1 within the current stage
    attempts     INT NOT NULL DEFAULT 0,
    embed_version TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ms_documents_user_idx   ON ms_documents (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS ms_documents_status_idx ON ms_documents (status);
CREATE INDEX IF NOT EXISTS ms_documents_hash_idx   ON ms_documents (user_id, source_hash);

-- Bring-your-own-model: a tenant's hosted LLM endpoint (vLLM / Ollama / any
-- OpenAI-compatible server, NVIDIA NIM, or Anthropic). When a row exists the
-- read path answers with THIS model instead of the server's LLM_* env config.
CREATE TABLE IF NOT EXISTS ms_user_llms (
    user_id    TEXT PRIMARY KEY,
    provider   TEXT NOT NULL DEFAULT 'openai',  -- openai | nvidia | anthropic
    model      TEXT NOT NULL,
    base_url   TEXT,                            -- e.g. http://my-vllm:8000/v1
    api_key    TEXT,                            -- optional (vLLM often has none)
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def init_schema() -> None:
    with pool().connection() as conn:
        conn.execute(SCHEMA)


def upsert_pending(video: dict[str, Any]) -> dict:
    """Insert a video as pending; re-submitting an existing id resets it."""
    with pool().connection() as conn:
        row = conn.execute(
            """
            INSERT INTO ms_videos (id, user_id, source, url, storage_key, source_hash, title, status)
            VALUES (%(id)s, %(user_id)s, %(source)s, %(url)s, %(storage_key)s,
                    %(source_hash)s, %(title)s, 'pending')
            ON CONFLICT (id) DO UPDATE SET
                url = COALESCE(EXCLUDED.url, ms_videos.url),
                storage_key = COALESCE(EXCLUDED.storage_key, ms_videos.storage_key),
                source_hash = COALESCE(EXCLUDED.source_hash, ms_videos.source_hash),
                title = COALESCE(EXCLUDED.title, ms_videos.title),
                status = 'pending', error = NULL, progress = NULL, updated_at = now()
            RETURNING *
            """,
            video,
        ).fetchone()
    return row


def set_status(video_id: str, status: str, *, error: str | None = None,
               title: str | None = None, frame_count: int | None = None,
               source_hash: str | None = None, embed_version: str | None = None,
               progress: float | None = None) -> None:
    with pool().connection() as conn:
        conn.execute(
            """
            UPDATE ms_videos SET status = %s, error = %s,
                title = COALESCE(%s, title),
                frame_count = COALESCE(%s, frame_count),
                source_hash = COALESCE(%s, source_hash),
                embed_version = COALESCE(%s, embed_version),
                progress = %s,
                updated_at = now()
            WHERE id = %s
            """,
            (status, error, title, frame_count, source_hash, embed_version,
             progress, video_id),
        )


def set_progress(video_id: str, progress: float) -> None:
    with pool().connection() as conn:
        conn.execute("UPDATE ms_videos SET progress = %s, updated_at = now() WHERE id = %s",
                     (round(progress, 3), video_id))


def bump_attempts(video_id: str) -> int:
    with pool().connection() as conn:
        row = conn.execute(
            "UPDATE ms_videos SET attempts = attempts + 1, updated_at = now() WHERE id = %s RETURNING attempts",
            (video_id,),
        ).fetchone()
    return row["attempts"] if row else 0


def get_video(video_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute("SELECT * FROM ms_videos WHERE id = %s", (video_id,)).fetchone()


def find_duplicate(user_id: str, source_hash: str, exclude_id: str) -> dict | None:
    """An already-indexed video with the same content for the same user."""
    with pool().connection() as conn:
        return conn.execute(
            """
            SELECT * FROM ms_videos
            WHERE user_id = %s AND source_hash = %s AND id <> %s AND status = 'indexed'
            LIMIT 1
            """,
            (user_id, source_hash, exclude_id),
        ).fetchone()


def list_videos(user_id: str, status: str | None = None) -> list[dict]:
    q = "SELECT * FROM ms_videos WHERE user_id = %s"
    params: list = [user_id]
    if status:
        q += " AND status = %s"
        params.append(status)
    q += " ORDER BY created_at DESC"
    with pool().connection() as conn:
        return conn.execute(q, tuple(params)).fetchall()


def videos_by_ids(ids: list[str]) -> dict[str, dict]:
    """Metadata join for search citations (title/url/source live here, not in Qdrant)."""
    if not ids:
        return {}
    with pool().connection() as conn:
        rows = conn.execute("SELECT * FROM ms_videos WHERE id = ANY(%s)", (ids,)).fetchall()
    return {r["id"]: r for r in rows}


def delete_video(video_id: str) -> None:
    with pool().connection() as conn:
        conn.execute("DELETE FROM ms_videos WHERE id = %s", (video_id,))


# ── Documents (papers, decks) ────────────────────────────────────────────────
# Mirrors the ms_videos functions above one-for-one; kept as separate functions
# (rather than parameterizing the video ones by table) so the video path stays
# byte-for-byte untouched — non-negotiable #6.

def upsert_pending_document(doc: dict[str, Any]) -> dict:
    """Insert a document as pending; re-submitting an existing id resets it."""
    with pool().connection() as conn:
        row = conn.execute(
            """
            INSERT INTO ms_documents (id, user_id, kind, source, uri, storage_key,
                                      source_hash, title, status)
            VALUES (%(id)s, %(user_id)s, %(kind)s, %(source)s, %(uri)s,
                    %(storage_key)s, %(source_hash)s, %(title)s, 'pending')
            ON CONFLICT (id) DO UPDATE SET
                uri = COALESCE(EXCLUDED.uri, ms_documents.uri),
                storage_key = COALESCE(EXCLUDED.storage_key, ms_documents.storage_key),
                source_hash = COALESCE(EXCLUDED.source_hash, ms_documents.source_hash),
                title = COALESCE(EXCLUDED.title, ms_documents.title),
                status = 'pending', error = NULL, progress = NULL, updated_at = now()
            RETURNING *
            """,
            doc,
        ).fetchone()
    return row


def set_doc_status(doc_id: str, status: str, *, error: str | None = None,
                   title: str | None = None, page_count: int | None = None,
                   chunk_count: int | None = None, source_hash: str | None = None,
                   embed_version: str | None = None,
                   progress: float | None = None) -> None:
    with pool().connection() as conn:
        conn.execute(
            """
            UPDATE ms_documents SET status = %s, error = %s,
                title = COALESCE(%s, title),
                page_count = COALESCE(%s, page_count),
                chunk_count = COALESCE(%s, chunk_count),
                source_hash = COALESCE(%s, source_hash),
                embed_version = COALESCE(%s, embed_version),
                progress = %s,
                updated_at = now()
            WHERE id = %s
            """,
            (status, error, title, page_count, chunk_count, source_hash,
             embed_version, progress, doc_id),
        )


def set_doc_progress(doc_id: str, progress: float) -> None:
    with pool().connection() as conn:
        conn.execute("UPDATE ms_documents SET progress = %s, updated_at = now() WHERE id = %s",
                     (round(progress, 3), doc_id))


def bump_doc_attempts(doc_id: str) -> int:
    with pool().connection() as conn:
        row = conn.execute(
            "UPDATE ms_documents SET attempts = attempts + 1, updated_at = now() WHERE id = %s RETURNING attempts",
            (doc_id,),
        ).fetchone()
    return row["attempts"] if row else 0


def get_document(doc_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute("SELECT * FROM ms_documents WHERE id = %s", (doc_id,)).fetchone()


def find_duplicate_document(user_id: str, source_hash: str, exclude_id: str) -> dict | None:
    """An already-indexed document with the same content for the same user."""
    with pool().connection() as conn:
        return conn.execute(
            """
            SELECT * FROM ms_documents
            WHERE user_id = %s AND source_hash = %s AND id <> %s AND status = 'indexed'
            LIMIT 1
            """,
            (user_id, source_hash, exclude_id),
        ).fetchone()


def list_documents(user_id: str, status: str | None = None) -> list[dict]:
    q = "SELECT * FROM ms_documents WHERE user_id = %s"
    params: list = [user_id]
    if status:
        q += " AND status = %s"
        params.append(status)
    q += " ORDER BY created_at DESC"
    with pool().connection() as conn:
        return conn.execute(q, tuple(params)).fetchall()


def documents_by_ids(ids: list[str]) -> dict[str, dict]:
    """Metadata join for search citations (title/uri live here, not in Qdrant)."""
    if not ids:
        return {}
    with pool().connection() as conn:
        rows = conn.execute("SELECT * FROM ms_documents WHERE id = ANY(%s)", (ids,)).fetchall()
    return {r["id"]: r for r in rows}


def delete_document(doc_id: str) -> None:
    with pool().connection() as conn:
        conn.execute("DELETE FROM ms_documents WHERE id = %s", (doc_id,))


# ── Unified source listing (videos + documents) ──────────────────────────────
# GET /admin/sources needs ONE list across both tables. Same UNION ALL shape the
# fair claimer below already uses, but projected to a common column set so a
# caller can render a video, a paper and a deck from one row type. The per-table
# list_videos/list_documents above stay as they are — the /api/* responses they
# back must not change (non-negotiable #6).

_SOURCES_SQL = """
SELECT id, user_id, 'video' AS kind, source, url AS uri, title, status, error,
       progress, attempts, frame_count AS unit_count, NULL::int AS chunk_count,
       created_at, updated_at
  FROM ms_videos    WHERE user_id = %(uid)s
UNION ALL
SELECT id, user_id, kind, source, uri, title, status, error,
       progress, attempts, page_count AS unit_count, chunk_count,
       created_at, updated_at
  FROM ms_documents WHERE user_id = %(uid)s
"""


def list_sources(user_id: str, status: str | None = None,
                 kind: str | None = None) -> list[dict]:
    """Every source this user owns, newest first, videos and documents pooled.

    `kind` is the literal 'video' for a video row and the document's own
    'paper'/'deck' — so filtering by kind works uniformly across both tables.
    """
    q = f"SELECT * FROM ({_SOURCES_SQL}) s"
    params: dict[str, Any] = {"uid": user_id}
    where = []
    if status:
        where.append("s.status = %(status)s")
        params["status"] = status
    if kind:
        where.append("s.kind = %(kind)s")
        params["kind"] = kind
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY s.created_at DESC"
    with pool().connection() as conn:
        return conn.execute(q, params).fetchall()


def get_source(user_id: str, source_id: str) -> dict | None:
    """One of this user's sources by id, from whichever table owns it (id
    prefixes don't collide: yt_/up_ are videos, doc_ are documents)."""
    with pool().connection() as conn:
        return conn.execute(
            f"SELECT * FROM ({_SOURCES_SQL}) s WHERE s.id = %(id)s",
            {"uid": user_id, "id": source_id},
        ).fetchone()


# ── Fair scheduling (WFQ) ────────────────────────────────────────────────────
# Fair across BOTH source types: a user backfilling 40 papers can't starve a
# user adding one video, and vice versa — the round-robin partitions by
# user_id across the union of both tables' pending rows.

def count_inflight() -> int:
    """How many videos+documents currently occupy execution capacity."""
    with pool().connection() as conn:
        row = conn.execute(
            """
            SELECT (SELECT count(*) FROM ms_videos WHERE status = ANY(%s)) +
                   (SELECT count(*) FROM ms_documents WHERE status = ANY(%s)) AS n
            """,
            (list(INFLIGHT_STATUSES), list(INFLIGHT_STATUSES)),
        ).fetchone()
    return row["n"] if row else 0


def wfq_claim_all(limit: int) -> list[dict]:
    """Atomically claim up to `limit` pending videos+documents in FAIR
    (round-robin across users, irrespective of source type) order, flipping
    them pending -> queued. Returns [{id, user_id, entity, kind}].

    Fairness: rank each user's pending rows (videos AND documents together) by
    age, then order by that rank first — so we take everyone's oldest item
    first, then everyone's 2nd, ... The two per-table UPDATE ... WHERE
    status='pending' RETURNING statements are the atomic claim: if two
    dispatchers race, each row is handed out once.
    """
    if limit <= 0:
        return []
    with pool().connection() as conn:
        picked = conn.execute(
            """
            SELECT id, user_id, entity FROM (
                SELECT id, user_id, 'video' AS entity, created_at FROM ms_videos
                WHERE status = 'pending'
                UNION ALL
                SELECT id, user_id, 'document' AS entity, created_at FROM ms_documents
                WHERE status = 'pending'
            ) u
            ORDER BY row_number() OVER (PARTITION BY user_id ORDER BY created_at, id), id
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        video_ids = [r["id"] for r in picked if r["entity"] == "video"]
        doc_ids = [r["id"] for r in picked if r["entity"] == "document"]
        claimed: list[dict] = []
        if video_ids:
            rows = conn.execute(
                """
                UPDATE ms_videos SET status = 'queued', updated_at = now()
                WHERE id = ANY(%s) AND status = 'pending'
                RETURNING id, user_id
                """,
                (video_ids,),
            ).fetchall()
            claimed.extend({**r, "entity": "video", "kind": "video"} for r in rows)
        if doc_ids:
            rows = conn.execute(
                """
                UPDATE ms_documents SET status = 'queued', updated_at = now()
                WHERE id = ANY(%s) AND status = 'pending'
                RETURNING id, user_id, kind
                """,
                (doc_ids,),
            ).fetchall()
            claimed.extend({**r, "entity": "document"} for r in rows)
        return claimed


# ── Bring-your-own-model (per-tenant LLM endpoint) ───────────────────────────

def get_user_llm(user_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute("SELECT * FROM ms_user_llms WHERE user_id = %s",
                            (user_id,)).fetchone()


def set_user_llm(user_id: str, *, provider: str, model: str,
                 base_url: str | None, api_key: str | None) -> dict:
    """Upsert a tenant's model endpoint. An empty api_key keeps the stored one
    (so users can change model/URL without re-pasting their secret)."""
    with pool().connection() as conn:
        return conn.execute(
            """
            INSERT INTO ms_user_llms (user_id, provider, model, base_url, api_key)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                base_url = EXCLUDED.base_url,
                api_key = COALESCE(NULLIF(EXCLUDED.api_key, ''), ms_user_llms.api_key),
                updated_at = now()
            RETURNING *
            """,
            (user_id, provider, model, base_url, api_key),
        ).fetchone()


def delete_user_llm(user_id: str) -> None:
    with pool().connection() as conn:
        conn.execute("DELETE FROM ms_user_llms WHERE user_id = %s", (user_id,))
