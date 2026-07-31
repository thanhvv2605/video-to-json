"""Frame selection.

Cost and accuracy of a video-to-JSON pipeline are decided here, not at the model
call. Sending every frame of a 60 s clip to a vision model means ~1500 images;
sending one per second still means 60 near-identical ones. Three stages cut that
to 8-15 genuinely distinct frames:

  1. ffmpeg scene detection  — keep only frames where the picture actually changed
  2. perceptual-hash dedup   — drop the near-duplicates scene detection lets through
  3. even-sample cap         — hard ceiling so a pathological video cannot blow the budget
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image

from ..config import settings


@dataclass
class Frame:
    path: Path
    timestamp: float


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def _dhash(path: Path, size: int = 8) -> int:
    """Difference hash — robust to compression noise, sensitive to real change."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(img.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | int(row[c] > row[c + 1])
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def extract_frames(video: Path, workdir: Path) -> List[Frame]:
    workdir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info"]
    if settings.hwaccel:
        cmd += ["-hwaccel", settings.hwaccel]
    cmd += [
        "-i", str(video),
        # eq(n,0) guarantees the opening frame survives even in a static video
        "-vf", (f"select='gt(scene,{settings.scene_threshold})+eq(n,0)',"
                f"scale=-2:{settings.frame_height},showinfo"),
        "-vsync", "vfr", "-q:v", "3",
        str(workdir / "frame_%04d.jpg"), "-y",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{proc.stderr[-2000:]}")

    # showinfo writes one line per emitted frame carrying its source timestamp
    times = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]
    files = sorted(workdir.glob("frame_*.jpg"))
    frames = [Frame(p, times[i] if i < len(times) else 0.0)
              for i, p in enumerate(files)]

    frames = _dedupe(frames)
    return _cap(frames, settings.max_frames)


def _dedupe(frames: List[Frame]) -> List[Frame]:
    kept: List[Frame] = []
    hashes: List[int] = []
    for f in frames:
        h = _dhash(f.path)
        if any(_hamming(h, k) <= settings.dhash_distance for k in hashes):
            f.path.unlink(missing_ok=True)
            continue
        kept.append(f)
        hashes.append(h)
    return kept


def _cap(frames: List[Frame], limit: int) -> List[Frame]:
    """Keep the ceiling but stay evenly spread across the timeline, so a video
    with a busy opening does not spend its whole budget in the first 5 seconds."""
    if len(frames) <= limit:
        return frames
    step = len(frames) / limit
    keep = {int(i * step) for i in range(limit)}
    for i, f in enumerate(frames):
        if i not in keep:
            f.path.unlink(missing_ok=True)
    return [f for i, f in enumerate(frames) if i in keep]
