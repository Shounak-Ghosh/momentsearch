"""Multimodal LLM — cited answer synthesis from frames, per-tenant switchable.

Every call takes an LLMConfig. Where it comes from (resolved in
src/rag/search.py):
  1. the user's own hosted model (ms_user_llms row — a vLLM/Ollama/LM Studio/
     Together/OpenRouter endpoint via base_url, NVIDIA NIM, or Anthropic), or
  2. the server-wide LLM_* env config as the fallback.

The two multimodal calls are where latency and cost actually live (retrieval
is milliseconds), so frames are downscaled to LLM_IMAGE_MAX_PX before they are
sent and only TOP_K of them ever reach the model.

Providers:
  * "openai"    — Chat Completions; covers every OpenAI-compatible server
                  (vLLM, Ollama, LM Studio, Together, Groq, OpenRouter, ...)
                  via base_url.
  * "nvidia"    — NVIDIA NIM / build.nvidia.com hosted vision models.
                  OpenAI-compatible, same client with NVIDIA's endpoint.
  * "anthropic" — the Anthropic Messages API.

Provider SDKs are imported lazily — only the one you use.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from . import config

# NVIDIA's hosted inference endpoint (OpenAI-compatible).
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

PROVIDERS = ("openai", "nvidia", "anthropic")

SYSTEM = (
    "You answer a user's question using the numbered moments provided as your "
    "evidence, drawn from indexed sources — talks (video), research papers, and "
    "slide decks. Each moment carries a locator: a VIDEO moment has a timestamp "
    "and may include a FRAME (what was shown on screen) and/or a TRANSCRIPT "
    "excerpt (what was said out loud); a PAPER moment has a page number and its "
    "text; a DECK moment has a slide number and its text. Use whichever evidence "
    "answers the question: for what someone SAID or a paper/deck's text, read "
    "that excerpt; for what is SHOWN on screen, read the frame.\n"
    "Rules:\n"
    "1. Read the question carefully and answer exactly what is asked. Start with a "
    "one-line direct answer, then explain in short paragraphs — ONE paragraph per "
    "distinct point. Keep it focused, don't pad. No preamble, don't restate the "
    "question.\n"
    "2. Ground every claim in the moments and cite the moment number(s) in square "
    "brackets, e.g. [1] or [2, 3]. Quote transcript/text excerpts accurately — "
    "keep the actual wording and numbers, don't alter or round them. Cite each "
    "moment's locator (timestamp/page/slide) exactly as given below; never state "
    "a page, slide, or timestamp that doesn't appear in the moments.\n"
    "3. Group the relevant moments by the point they make:\n"
    "   - Moments that make the SAME point (especially several from the same "
    "source) belong TOGETHER in ONE paragraph, cited together, e.g. [1, 2]. Do not "
    "split one shared point across separate paragraphs.\n"
    "   - Moments that make DIFFERENT points, or come from different sources, go "
    "in SEPARATE paragraphs, each with its own citation.\n"
    "   Cover every distinct relevant point — don't merge unrelated ones and don't "
    "drop any.\n"
    "4. Don't use outside knowledge or invent details that aren't in the moments.\n"
    "5. Abstain ONLY as a last resort: if — and only if — none of the moments are "
    "relevant to the question at all, reply with a single sentence saying you "
    "couldn't find it in the indexed sources. If even one moment is relevant, "
    "ANSWER from it; do not refuse just because the match is partial."
)


CAPTION_SYSTEM = (
    "You describe a single slide from a presentation deck for a search index. "
    "Transcribe ALL visible text verbatim, in reading order — titles, labels, "
    "axis names, legend entries, numbers. Then describe what the image/chart/"
    "diagram shows factually: what kind of visual it is, what it depicts, and "
    "any values or relationships it makes visible. Be concrete and literal. "
    "Do not speculate about what the speaker might be saying, do not add "
    "commentary, and do not describe anything not actually visible in the "
    "image. Keep it to a few sentences."
)


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 1024


def env_config() -> LLMConfig | None:
    """The server-wide fallback model from LLM_* env vars, if configured."""
    if not config.llm_configured():
        return None
    return LLMConfig(provider=config.LLM_PROVIDER, model=config.LLM_MODEL,
                     api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL,
                     max_tokens=config.LLM_MAX_TOKENS)


def from_row(row: dict) -> LLMConfig:
    """A tenant's own hosted model (ms_user_llms row)."""
    return LLMConfig(provider=row.get("provider") or "openai",
                     model=row.get("model") or "",
                     api_key=row.get("api_key") or "",
                     base_url=row.get("base_url") or "",
                     max_tokens=config.LLM_MAX_TOKENS)


def _intro(question: str, n: int) -> str:
    return (
        f"QUESTION: {question}\n\n"
        f"Answer this question using the {n} moments below (numbered 1 to {n}). "
        "Each comes from a video, paper, or deck and carries its own locator "
        "(timestamp/page/slide) plus a frame and/or a text excerpt. If the "
        "question is about what was said or written, use that excerpt. Give a "
        "direct answer grounded in the relevant moment(s), cited as [n] with its "
        "locator. Only say you couldn't find it if none of the moments are "
        "relevant."
    )


def _downscale(jpeg: bytes) -> bytes:
    """Shrink a frame before it becomes LLM image tokens."""
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg))
    if max(img.size) <= config.LLM_IMAGE_MAX_PX:
        return jpeg
    img.thumbnail((config.LLM_IMAGE_MAX_PX, config.LLM_IMAGE_MAX_PX))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def answer(question: str, moments: list[dict], cfg: LLMConfig) -> str:
    """Synthesize a cited answer from retrieved moments with `cfg`'s model.

    moments: [{"image": bytes|None, "transcript": str|None, "timestamp": str,
               "kind": "video"|"paper"|"deck" (optional, defaults "video"),
               "title": str|None, "page": int|None, "slide": int|None}]
    — each may carry a frame, a text excerpt, or both."""
    if cfg.provider == "anthropic":
        return _answer_anthropic(cfg, question, moments)
    return _answer_openai(cfg, question, moments)


def answer_stream(question: str, moments: list[dict], cfg: LLMConfig):
    """Same contract as answer(), but yields text deltas as they arrive.

    Used by the SSE /ask_stream route so citations can reach the client before
    a single answer token — the caller still runs _validate_citations on the
    fully assembled text afterward, so streaming is a UX layer over the same
    grounding guarantee, never a bypass of it."""
    if cfg.provider == "anthropic":
        yield from _answer_stream_anthropic(cfg, question, moments)
    else:
        yield from _answer_stream_openai(cfg, question, moments)


def ping(cfg: LLMConfig) -> str:
    """Connectivity + vision check: one tiny image, one word back. Raises with
    the provider's error on failure (surfaced to the settings UI)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (220, 40, 40)).save(buf, format="JPEG")
    return answer("Reply with the dominant color of moment 1, one word.",
                  [{"image": buf.getvalue(), "transcript": None, "timestamp": "00:00"}], cfg)


def _base_url(cfg: LLMConfig) -> str | None:
    if cfg.base_url:
        return cfg.base_url
    if cfg.provider == "nvidia":
        return NVIDIA_BASE_URL
    return None


_KIND_TAG = {"video": "VIDEO", "paper": "PAPER", "deck": "DECK"}


def _label(i: int, m: dict) -> str:
    # `kind` is absent for callers that pass bare moments (e.g. ping()'s
    # connectivity check) — treat that as a video moment, matching the
    # original (pre-cross-source) behavior exactly.
    kind = m.get("kind") or "video"
    tag = _KIND_TAG.get(kind, "VIDEO")
    if kind == "paper":
        where = f"page {m.get('page') or m.get('timestamp', '')}"
    elif kind == "deck":
        where = f"slide {m.get('slide') or m.get('timestamp', '')}"
    else:
        where = f"@ {m.get('timestamp', '')}"
    title = f' "{m["title"]}"' if m.get("title") else ""
    line = f"[{i}] {tag}{title} {where}"
    if m.get("transcript"):
        field = "transcript" if kind == "video" else "text"
        line += f' {field}: "{m["transcript"]}"'
    if kind == "video" and m.get("image") is None:
        line += " (transcript only, no frame)"
    return line


def _build_blocks(question: str, moments: list[dict]) -> list[dict]:
    """Provider-agnostic content blocks: intro, then per-moment label +
    optional image. Both the blocking and streaming paths for both providers
    build from this one place, so the prompt can never drift between them."""
    blocks: list[dict] = [{"kind": "text", "text": _intro(question, len(moments))}]
    for i, m in enumerate(moments, 1):
        blocks.append({"kind": "text", "text": _label(i, m)})
        if m.get("image"):
            blocks.append({"kind": "image", "data": _downscale(m["image"])})
    return blocks


def _openai_content(blocks: list[dict]) -> list[dict]:
    out = []
    for b in blocks:
        if b["kind"] == "text":
            out.append({"type": "text", "text": b["text"]})
        else:
            uri = f"data:image/jpeg;base64,{base64.b64encode(b['data']).decode()}"
            out.append({"type": "image_url", "image_url": {"url": uri}})
    return out


def _anthropic_content(blocks: list[dict]) -> list[dict]:
    out = []
    for b in blocks:
        if b["kind"] == "text":
            out.append({"type": "text", "text": b["text"]})
        else:
            out.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(b["data"]).decode()}})
    return out


def _answer_openai(cfg: LLMConfig, question: str, moments: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key or "not-needed", base_url=_base_url(cfg))
    content = _openai_content(_build_blocks(question, moments))
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=cfg.max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _answer_anthropic(cfg: LLMConfig, question: str, moments: list[dict]) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.api_key, base_url=cfg.base_url or None)
    content = _anthropic_content(_build_blocks(question, moments))
    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _answer_stream_openai(cfg: LLMConfig, question: str, moments: list[dict]):
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key or "not-needed", base_url=_base_url(cfg))
    content = _openai_content(_build_blocks(question, moments))
    stream = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=cfg.max_tokens,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _answer_stream_anthropic(cfg: LLMConfig, question: str, moments: list[dict]):
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.api_key, base_url=cfg.base_url or None)
    content = _anthropic_content(_build_blocks(question, moments))
    with client.messages.stream(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        yield from stream.text_stream


# ── Slide captioning (deck ingestion — Part 2) ────────────────────────────────
# A slide with little/no extractable text (a chart, diagram, or screenshot
# drawn as vector shapes) is otherwise invisible to the text index. One vision
# call per such slide turns it into a searchable, grounded caption. Separate
# from answer() because it's a one-image describe task, not a numbered-moment
# Q&A — the "moments" framing/citation validation doesn't apply here.

def _caption_prompt(hint: str) -> str:
    base = "Describe this slide for a search index."
    if hint.strip():
        # Whatever text WAS extracted (a title, a label) — steer the model to
        # describe the REST (the figure/chart), not re-transcribe what we
        # already have.
        base += (f" Text already extracted from this slide: \"{hint.strip()}\". "
                 "Focus your description on the image/chart/diagram content, "
                 "not on repeating that text.")
    return base


def caption_image(cfg: LLMConfig, jpeg: bytes, hint: str = "") -> str:
    """One factual, grounded caption for a rendered slide image. `hint` is
    whatever text extraction already found (may be empty for an image-only
    slide) — passed so the model describes the figure, not the title again."""
    max_tokens = min(cfg.max_tokens, 300)  # this runs per slide; keep it cheap
    if cfg.provider == "anthropic":
        return _caption_anthropic(cfg, jpeg, hint, max_tokens)
    return _caption_openai(cfg, jpeg, hint, max_tokens)


def _caption_openai(cfg: LLMConfig, jpeg: bytes, hint: str, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key or "not-needed", base_url=_base_url(cfg))
    uri = f"data:image/jpeg;base64,{base64.b64encode(_downscale(jpeg)).decode()}"
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": CAPTION_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": _caption_prompt(hint)},
                {"type": "image_url", "image_url": {"url": uri}},
            ]},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _caption_anthropic(cfg: LLMConfig, jpeg: bytes, hint: str, max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=cfg.api_key, base_url=cfg.base_url or None)
    resp = client.messages.create(
        model=cfg.model,
        max_tokens=max_tokens,
        system=CAPTION_SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": _caption_prompt(hint)},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(_downscale(jpeg)).decode()}},
        ]}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()
