"""Audio extraction and transcription.

Whisper caps uploads at 25 MB. Rather than discovering that in production on a
long video, the audio is always normalised to mono 16 kHz at a low bitrate —
which is both under the limit for ~100 minutes of speech and exactly the format
the model was trained on. Longer inputs are split on time and stitched back with
the offset applied to every segment timestamp.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from openai import OpenAI

from ..config import make_client, settings

CHUNK_SECONDS = 20 * 60
MAX_UPLOAD_BYTES = 24 * 1024 * 1024


@dataclass
class Segment:
    start: float
    end: float
    text: str


def extract_audio(video: Path, workdir: Path) -> Path:
    out = workdir / "audio.mp3"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vn", "-ac", "1", "-ar", "16000",
           "-c:a", "libmp3lame", "-b:a", settings.audio_bitrate,
           str(out), "-y"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{proc.stderr[-2000:]}")
    return out


def transcribe(audio: Path, duration: float) -> List[Segment]:
    client = make_client()

    if audio.stat().st_size <= MAX_UPLOAD_BYTES and duration <= CHUNK_SECONDS:
        return _transcribe_one(client, audio, offset=0.0)

    segments: List[Segment] = []
    for i in range(math.ceil(duration / CHUNK_SECONDS)):
        offset = i * CHUNK_SECONDS
        part = audio.with_name(f"audio_{i:03d}.mp3")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-ss", str(offset), "-t", str(CHUNK_SECONDS), "-i", str(audio),
             "-c", "copy", str(part), "-y"],
            check=True, capture_output=True,
        )
        segments += _transcribe_one(client, part, offset=offset)
    return segments


def _transcribe_one(client: OpenAI, audio: Path, offset: float) -> List[Segment]:
    last_error: Exception | None = None
    for attempt in range(settings.max_retries):
        try:
            with audio.open("rb") as fh:
                resp = client.audio.transcriptions.create(
                    model=settings.asr_model,
                    file=fh,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            return [
                Segment(s.start + offset, s.end + offset, s.text.strip())
                for s in (resp.segments or [])
            ]
        except Exception as exc:  # transport / rate limit / transient 5xx
            last_error = exc
    raise RuntimeError(f"transcription failed after {settings.max_retries} attempts: {last_error}")


def to_srt(segments: List[Segment]) -> str:
    """Segment timestamps are what make burned-in subtitles possible — the same
    alignment step the production video system this demo is modelled on uses."""
    def ts(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"

    return "\n".join(
        f"{i}\n{ts(sg.start)} --> {ts(sg.end)}\n{sg.text}\n"
        for i, sg in enumerate(segments, 1)
    )
