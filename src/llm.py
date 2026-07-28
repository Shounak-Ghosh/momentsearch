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
    "You answer a user's question about a video using the numbered moments "
    "provided as your evidence. Each moment has a timestamp and may include a "
    "video FRAME (what was shown on screen) and/or a TRANSCRIPT excerpt (what was "
    "said out loud). Use BOTH kinds of evidence: for a question about what someone "
    "SAID or talked about, read the transcript text; for a question about what is "
    "SHOWN, read the frame.\n"
    "Rules:\n"
    "1. Read the question carefully and answer exactly what is asked. Start with a "
    "one-line direct answer, then explain in short paragraphs — ONE paragraph per "
    "distinct point. Keep it focused, don't pad. No preamble, don't restate the "
    "question.\n"
    "2. Ground every claim in the moments and cite the moment number(s) in square "
    "brackets, e.g. [1] or [2, 3]. When the question is about what was said, quote "
    "the transcript accurately — keep the actual wording and numbers, don't alter "
    "or round them.\n"
    "3. Group the relevant moments by the point they make:\n"
    "   - Moments that make the SAME point (especially several from the same "
    "video) belong TOGETHER in ONE paragraph, cited together, e.g. [1, 2]. Do not "
    "split one shared point across separate paragraphs.\n"
    "   - Moments that make DIFFERENT points, or come from different videos, go in "
    "SEPARATE paragraphs, each with its own citation.\n"
    "   Cover every distinct relevant point — don't merge unrelated ones and don't "
    "drop any.\n"
    "4. Don't use outside knowledge or invent details that aren't in the moments.\n"
    "5. Abstain ONLY as a last resort: if — and only if — none of the moments are "
    "relevant to the question at all, reply with a single sentence saying you "
    "couldn't find it in the video. If even one moment is relevant, ANSWER from "
    "it; do not refuse just because the match is partial."
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
        "Each has a timestamp and a video frame and/or a transcript excerpt. If "
        "the question is about what was said, use the transcript text. Give a "
        "direct answer grounded in the relevant moment(s), cited as [n]. Only say "
        "you couldn't find it if none of the moments are relevant."
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

    moments: [{"image": bytes|None, "transcript": str|None, "timestamp": str}]
    — each may carry a frame, a transcript excerpt, or both."""
    if cfg.provider == "anthropic":
        return _answer_anthropic(cfg, question, moments)
    return _answer_openai(cfg, question, moments)


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


def _label(i: int, m: dict) -> str:
    line = f"[{i}] @ {m.get('timestamp', '')}"
    if m.get("transcript"):
        line += f' transcript: "{m["transcript"]}"'
    if m.get("image") is None:
        line += " (transcript only, no frame)"
    return line


def _answer_openai(cfg: LLMConfig, question: str, moments: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key or "not-needed", base_url=_base_url(cfg))
    content: list[dict] = [{"type": "text", "text": _intro(question, len(moments))}]
    for i, m in enumerate(moments, 1):
        content.append({"type": "text", "text": _label(i, m)})
        if m.get("image"):
            uri = f"data:image/jpeg;base64,{base64.b64encode(_downscale(m['image'])).decode()}"
            content.append({"type": "image_url", "image_url": {"url": uri}})
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
    blocks: list[dict] = [{"type": "text", "text": _intro(question, len(moments))}]
    for i, m in enumerate(moments, 1):
        blocks.append({"type": "text", "text": _label(i, m)})
        if m.get("image"):
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(_downscale(m["image"])).decode()}})
    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system=SYSTEM,
        messages=[{"role": "user", "content": blocks}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


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
