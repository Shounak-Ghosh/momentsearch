"""Acquiring the source video into worker scratch space.

Two sources, one contract — a local temp file the sampler can read:
  * upload  — the browser already PUT the file to object storage via a
              presigned URL; we stream it down from the bucket.
  * youtube — yt-dlp downloads a small <=480p video-only stream (CLIP
              downsizes to 224px anyway; 4K would just waste bandwidth).

The temp file is worker scratch, deleted after the run — durable copies live
only in object storage ("nothing on local").
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from .. import storage


def scratch_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "momentsearch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_upload(storage_key: str, video_id: str) -> Path:
    suffix = Path(storage_key).suffix or ".mp4"
    dest = scratch_dir() / f"{video_id}{suffix}"
    return storage.download_to(storage_key, dest)


# ── Documents (papers, decks: PDF or PPTX) ───────────────────────────────────

# Magic bytes for the two formats we accept — an arXiv abstract page or a 404
# HTML page can still return HTTP 200, so this catches a wrong content type
# here instead of failing obscurely in pypdf/python-pptx later. PPTX is a ZIP
# container (docx/xlsx share the same signature), hence "PK\x03\x04".
_MAGIC = {".pdf": b"%PDF-", ".pptx": b"PK\x03\x04"}


def doc_ext(row: dict) -> str:
    """Which file type a document row's bytes are — the suffix of its
    storage_key (uploads) or its uri (URL registrations), lowercased,
    defaulting to .pdf (papers never carry an extension-bearing key today).
    One helper so fetch/archive/delete never disagree about the format."""
    for field in ("storage_key", "uri"):
        value = row.get(field) or ""
        suffix = Path(value.split("?", 1)[0]).suffix.lower()
        if suffix in _MAGIC:
            return suffix
    return ".pdf"


def fetch_document(uri: str, doc_id: str, ext: str = ".pdf") -> Path:
    """Acquire a document's source file (paper PDF, or PDF/PPTX deck) into
    worker scratch.

    Two sources, one contract — same pattern as fetch_upload/fetch_youtube:
      * an http(s):// URL     — streamed down directly (no yt-dlp involved)
      * a storage:// / bare key — already in our bucket (a browser upload via
        presign), pulled the same way fetch_upload pulls a video
    """
    from ..config import MAX_DOC_MB

    dest = scratch_dir() / f"{doc_id}{ext}"
    if uri.startswith("storage://"):
        return storage.download_to(uri.removeprefix("storage://"), dest)
    if not (uri.startswith("http://") or uri.startswith("https://")):
        raise ValueError(f"Unsupported document URI scheme: {uri!r}")
    _download_http(uri, dest, MAX_DOC_MB * 1024 * 1024, ext)
    return dest


def fetch_pdf(uri: str, doc_id: str) -> Path:
    """Back-compat alias for the paper path (always .pdf)."""
    return fetch_document(uri, doc_id, ".pdf")


def _download_http(url: str, dest: Path, max_bytes: int, ext: str = ".pdf") -> None:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "momentsearch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        size = 0
        with dest.open("wb") as out:
            while chunk := resp.read(1 << 16):
                size += len(chunk)
                if size > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise ValueError(f"Document exceeds the {max_bytes // (1024*1024)}MB limit.")
                out.write(chunk)
    magic = _MAGIC.get(ext, b"%PDF-")
    with dest.open("rb") as fh:
        head = fh.read(len(magic))
    if head != magic:
        dest.unlink(missing_ok=True)
        raise ValueError(f"URL did not return a {ext} file (got {head!r}): {url}")


_cookie_path: str | None = None


def _cookiefile() -> str | None:
    """Resolve cookies from a mounted file (YT_COOKIES_FILE) or a base64 secret
    (YT_COOKIES_B64, written to a temp file once). Same code, local or cloud."""
    global _cookie_path
    from ..config import YT_COOKIES_B64, YT_COOKIES_FILE

    # Prefer a real mounted file; if the path is set but missing (e.g. the
    # local YT_COOKIES_FILE got imported to Fly where ./data isn't mounted),
    # fall through to the base64 secret instead of handing yt-dlp a dead path.
    if YT_COOKIES_FILE and Path(YT_COOKIES_FILE).exists():
        return YT_COOKIES_FILE
    if YT_COOKIES_B64:
        if _cookie_path is None:
            import base64
            p = scratch_dir() / "yt_cookies.txt"
            p.write_bytes(base64.b64decode(YT_COOKIES_B64))
            _cookie_path = str(p)
        return _cookie_path
    return None


def _yt_opts(video_id: str, clients: list[str]) -> dict:
    from ..config import YT_JS_RUNTIMES, YT_PROXY_URL, YT_REMOTE_COMPONENTS

    opts = {
        # We only sample frames — audio and resolution don't matter, smaller is
        # better. Prefer a <=480p video-only stream, but fall back through ANY
        # video-only stream (the tv/android/ios clients mostly return adaptive
        # video-only formats) and finally ANY format at all, so this never
        # errors "Requested format is not available".
        "format": ("bestvideo[height<=480][ext=mp4]/bestvideo[height<=480]/"
                   "best[height<=480][ext=mp4]/best[height<=480]/"
                   "bestvideo[ext=mp4]/bestvideo/best"),
        "outtmpl": str(scratch_dir() / f"{video_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    # Only override the player client when asked (empty = yt-dlp's default,
    # which is best once a JS runtime is available). Forcing tv/android is a
    # fallback for when the default fails.
    if clients:
        opts["extractor_args"] = {"youtube": {"player_client": clients}}
    # JS runtime + EJS solver — required by modern yt-dlp to extract YouTube
    # formats at all (see config). Node ships in the Docker image.
    if YT_JS_RUNTIMES:
        opts["js_runtimes"] = {r: {} for r in YT_JS_RUNTIMES}
    if YT_REMOTE_COMPONENTS:
        opts["remote_components"] = list(YT_REMOTE_COMPONENTS)
    # Cookies are the durable fix when the IP itself is blocked (datacenter);
    # a residential proxy is the alternative. See .env.example.
    cookies = _cookiefile()
    if cookies:
        opts["cookiefile"] = cookies
    if YT_PROXY_URL:
        opts["proxy"] = YT_PROXY_URL
    return opts


def _yt_download(url: str, video_id: str, clients: list[str]) -> tuple[Path, str]:
    import yt_dlp

    with yt_dlp.YoutubeDL(_yt_opts(video_id, clients)) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    return path, (info.get("title") or video_id)


def fetch_youtube(url: str, video_id: str) -> tuple[Path, str]:
    """Download via yt-dlp. Returns (path, title).

    Uses a robust multi-client list on the first try (see _yt_opts). If it
    still fails, retries once with the wider fallback set. Persistent failure
    across ALL videos usually means the IP is blocked (datacenter deploy) —
    set YT_COOKIES_FILE or YT_PROXY_URL then (see .env.example)."""
    from ..config import YT_PLAYER_CLIENTS, YT_FALLBACK_CLIENTS

    try:
        return _yt_download(url, video_id, YT_PLAYER_CLIENTS)
    except Exception as exc:
        extra = [c for c in YT_FALLBACK_CLIENTS if c not in YT_PLAYER_CLIENTS]
        if extra:
            print(f"[fetch] {video_id}: {str(exc)[:80]}… — retrying with "
                  f"{YT_PLAYER_CLIENTS + extra}")
            return _yt_download(url, video_id, YT_PLAYER_CLIENTS + extra)
        raise
