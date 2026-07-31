"""End-to-end orchestration for one video."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

from .pipeline import audio as audio_mod
from .pipeline import extract as extract_mod
from .pipeline import frames as frames_mod


def process_video(video: Path, keep_workdir: bool = False) -> dict:
    workdir = Path(tempfile.mkdtemp(prefix="v2j-"))
    started = time.time()
    try:
        duration = frames_mod.probe_duration(video)

        frames = frames_mod.extract_frames(video, workdir / "frames")
        audio_path = audio_mod.extract_audio(video, workdir)
        segments = audio_mod.transcribe(audio_path, duration)

        insight = extract_mod.extract(frames, segments)

        return {
            "source": video.name,
            "duration_sec": round(duration, 2),
            "elapsed_sec": round(time.time() - started, 2),
            "cost": extract_mod.estimate_cost(
                len(frames), len(insight.transcript), duration
            ),
            "srt": audio_mod.to_srt(segments),
            "insight": insight.model_dump(),
        }
    finally:
        # Temp files are the classic way a media pipeline fills a disk at 3am.
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
