"""Paper parsing + page-aware chunking — pure functions, no Prefect/DB/Qdrant.

Two steps, mirroring the video pipeline's shape (fetch -> sample -> embed)
one level up: a document is fetch -> parse -> chunk -> embed -> index (the
Prefect flow lives in document_pipeline.py; this module only turns PDF bytes
into page-numbered chunks, so it's testable without any infrastructure).

  parse_pdf()   PDF -> [{page, text}], 1-indexed, page numbers preserved
  chunk_pages() pages -> [{text, page, page_end, idx}], NEVER spans a page
  render_pages() pages -> {page: jpeg} — rasterized page (shared with deck)
  resize_jpeg()  downscale a JPEG to citation thumbnail width

Why chunks never cross a page: the citation locator IS the page number
(A3_README's contract: a paper cites a page, not a passage). A chunker that
merges the tail of page 4 with the head of page 5 would point a citation at
the wrong page for whichever half it didn't quote. Losing a little context at
a page boundary is a better trade than a wrong locator.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

from ..config import PAPER_CHUNK_CHARS, PAPER_CHUNK_OVERLAP, PAPER_MIN_CHUNK_CHARS

_HYPHEN_BREAK = re.compile(r"-\n(\w)")   # "retrie-\nval" -> "retrieval"
_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def parse_pdf(path: Path) -> list[dict]:
    """Extract text per page, 1-indexed. Returns one entry per page — including
    empty ones — so page numbering never drifts from the physical PDF."""
    import pypdf

    reader = pypdf.PdfReader(str(path))
    pages: list[dict] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # a single malformed page shouldn't fail the whole doc
            print(f"[paper] page {i}: extract failed ({type(exc).__name__}: {exc})")
            raw = ""
        pages.append({"page": i, "text": _clean(raw)})
    return pages


def _clean(text: str) -> str:
    """Rejoin line-end hyphenation and collapse whitespace. Academic PDFs
    hyphenate across line breaks; left alone, that splits words apart in the
    embedding input and hurts retrieval quality."""
    text = _HYPHEN_BREAK.sub(r"\1", text)
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_long(text: str, span: int) -> list[str]:
    """Fallback for a single paragraph that alone exceeds `span` — common in
    scanned/OCR'd or poorly-formatted PDFs with no paragraph breaks at all.
    Split on sentence boundaries; if even one sentence outruns `span`, hard-cut
    it by characters as a last resort. Without this, chunk_pages would silently
    emit one oversized chunk instead of respecting PAPER_CHUNK_CHARS."""
    parts: list[str] = []
    buf = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        candidate = f"{buf} {sentence}".strip() if buf else sentence
        if len(candidate) > span and buf:
            parts.append(buf)
            buf = sentence
        else:
            buf = candidate
        while len(buf) > span:  # a single sentence longer than span itself
            parts.append(buf[:span])
            buf = buf[span:]
    if buf:
        parts.append(buf)
    return parts


def chunk_pages(pages: list[dict], *, chunk_chars: int | None = None,
                overlap: int | None = None, min_chars: int | None = None) -> list[dict]:
    """Page-aware chunking: accumulate paragraphs up to ~chunk_chars, carrying
    `overlap` chars of trailing context into the next chunk — but ONLY within
    the same page. A chunk's `page` is therefore always exact.

    Returns [{text, page, page_end, idx}]. `page_end` equals `page` today
    (chunks don't span pages); the field exists so a future spanning mode
    doesn't need a payload shape change.
    """
    span = chunk_chars or PAPER_CHUNK_CHARS
    lap = overlap if overlap is not None else PAPER_CHUNK_OVERLAP
    floor = min_chars if min_chars is not None else PAPER_MIN_CHUNK_CHARS

    chunks: list[dict] = []
    for page in pages:
        paras = _paragraphs(page["text"])
        if not paras:
            continue
        # Pre-split any paragraph that alone exceeds the chunk size (no \n\n
        # breaks to lean on) so the accumulate loop below never has to.
        pieces: list[str] = []
        for p in paras:
            pieces.extend(_split_long(p, span) if len(p) > span else [p])
        buf = ""
        for piece in pieces:
            candidate = f"{buf}\n\n{piece}" if buf else piece
            if len(candidate) > span and buf:
                _emit(chunks, buf, page["page"], floor)
                # carry the tail of the just-emitted chunk as overlap context
                tail = buf[-lap:] if lap > 0 else ""
                buf = f"{tail}\n\n{piece}" if tail else piece
            else:
                buf = candidate
        if buf:
            _emit(chunks, buf, page["page"], floor)

    for i, c in enumerate(chunks):
        c["idx"] = i
    return chunks


def _emit(chunks: list[dict], text: str, page: int, floor: int) -> None:
    text = text.strip()
    if len(text) < floor:  # header/footer/page-number fragments — not worth a vector
        return
    chunks.append({"text": text, "page": page, "page_end": page})


def guess_title(pages: list[dict], fallback: str) -> str:
    """First non-trivial line of page 1 — good enough when the caller didn't
    supply a title; the request's `title` always wins over this."""
    for page in pages:
        for line in page["text"].splitlines():
            line = line.strip()
            if len(line) >= 8:
                return line[:200]
    return fallback


# ── Render (citation thumbnails; shared with deck.py's PDF-deck case) ────────

def render_pages(path: Path, pages: list[int], *, width: int | None = None) -> dict[int, bytes]:
    """Page numbers -> rasterized JPEG (pypdfium2). A deck backed by a PDF is
    one-page-per-slide, so deck.py's render_slides() calls straight through to
    this for its PDF branch — one rasterizer, not two."""
    import pypdfium2 as pdfium

    from ..config import PAGE_RENDER_WIDTH

    if not pages:
        return {}
    w = width or PAGE_RENDER_WIDTH
    out: dict[int, bytes] = {}
    doc = pdfium.PdfDocument(str(path))
    try:
        for p in pages:
            idx = p - 1
            if idx < 0 or idx >= len(doc):
                continue
            page = doc[idx]
            try:
                page_width_pt = page.get_size()[0]
                scale = (w / page_width_pt) if page_width_pt else 1.0
                bitmap = page.render(scale=scale)
                try:
                    img = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                out[p] = buf.getvalue()
            except Exception as exc:
                print(f"[paper] page {p}: render failed ({type(exc).__name__}: {exc})")
            finally:
                page.close()
    finally:
        doc.close()
    return out


def resize_jpeg(jpeg: bytes, width: int) -> bytes:
    """Downscale an already-rendered page/slide JPEG to a smaller citation
    thumbnail width without rasterizing a second time."""
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg))
    if max(img.size) <= width:
        return jpeg
    img = img.convert("RGB")
    img.thumbnail((width, width))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
