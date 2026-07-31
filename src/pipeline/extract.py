"""Multimodal extraction: selected frames + transcript -> validated JSON.

Two things make this reliable rather than a demo trick:

  * the response schema is the Pydantic model itself, so the API cannot return
    a shape the caller does not expect;
  * the prompt forbids inference. A vision model asked "what is this product"
    will happily invent a price. Asked to report only what is legible or spoken,
    it returns null instead — which is the answer an e-commerce pipeline can
    actually act on.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List

from openai import OpenAI

from ..config import settings
from ..schemas import VideoInsight
from .audio import Segment
from .frames import Frame

SYSTEM = """You extract structured data from short-form video.

You receive frames sampled at scene changes, each labelled with its timestamp,
plus the transcript with segment timings.

Rules:
- Report only what is legible on screen or spoken. Never infer a brand, price,
  or attribute that is not present. Use null rather than guessing.
- Timestamps must come from the labels given to you, never estimated.
- on_screen_text is verbatim, including casing and currency symbols.
- hooks are verbatim lines from the transcript, not paraphrases.
- Set confidence below 0.6 when frames are blurry, text is unreadable, or the
  audio carries no usable information.
"""


def _data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def extract(frames: List[Frame], segments: List[Segment]) -> VideoInsight:
    client = OpenAI(api_key=settings.openai_api_key)

    transcript = "\n".join(f"[{s.start:.1f}-{s.end:.1f}] {s.text}" for s in segments)
    content: list = [{
        "type": "text",
        "text": f"Transcript with segment timings:\n{transcript or '(no speech detected)'}",
    }]
    for f in frames:
        content.append({"type": "text", "text": f"Frame at t={f.timestamp:.2f}s"})
        content.append({
            "type": "image_url",
            # "low" detail is one fixed tile per image. For frames already
            # downscaled to 720p it costs a fraction of "high" and loses nothing
            # that matters here; raise it only when reading fine print.
            "image_url": {"url": _data_url(f.path), "detail": "low"},
        })

    last_error: Exception | None = None
    for _ in range(settings.max_retries):
        try:
            resp = client.chat.completions.parse(
                model=settings.vision_model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": content}],
                response_format=VideoInsight,
                temperature=0,
            )
            parsed = resp.choices[0].message.parsed
            if parsed is None:
                raise RuntimeError("model refused or returned no parsable object")
            if not parsed.transcript:
                parsed.transcript = " ".join(s.text for s in segments)
            return parsed
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"extraction failed after {settings.max_retries} attempts: {last_error}")


def estimate_cost(n_frames: int, transcript_chars: int, duration_sec: float) -> dict:
    """Rough per-video cost, logged on every run.

    Knowing this number per video is the difference between an MVP that scales
    and one that quietly runs up a bill — it is the first thing to watch once
    volume climbs.
    """
    image_tokens = n_frames * 85            # one low-detail tile each
    text_tokens = transcript_chars / 4 + 400
    vision_usd = (image_tokens + text_tokens) / 1_000_000 * 2.50 + 1200 / 1_000_000 * 10.0
    whisper_usd = duration_sec / 60 * 0.006
    return {
        "frames": n_frames,
        "vision_usd": round(vision_usd, 4),
        "whisper_usd": round(whisper_usd, 4),
        "total_usd": round(vision_usd + whisper_usd, 4),
    }
