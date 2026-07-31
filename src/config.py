import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Read .env from the project root regardless of where the process was started —
# running `python -m src.cli` from a parent directory should not silently lose
# the API key.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Any OpenAI-compatible endpoint works — the official API, a self-hosted
    # gateway, vLLM, or a regional reseller. Keeping this configurable means the
    # provider can change without touching a line of pipeline code, and it makes
    # the engine deployable where api.openai.com is slow or blocked.
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    vision_model: str = os.getenv("VISION_MODEL", "gpt-4o")
    asr_model: str = os.getenv("ASR_MODEL", "whisper-1")

    # Language of the generated prose fields (summary, scene descriptions).
    # "source" keeps whatever the video is in — the right default for a catalogue
    # that has to stay searchable in the market it sells to. Set a BCP-47 code
    # such as "en" to normalise everything into one language instead.
    output_language: str = os.getenv("OUTPUT_LANGUAGE", "source")

    # Frame selection — the single biggest cost lever in this pipeline.
    # Scene threshold 0.30 is a good default for short-form social video;
    # lower it for slow product shots, raise it for fast-cut ads.
    scene_threshold: float = float(os.getenv("SCENE_THRESHOLD", "0.30"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "720"))
    max_frames: int = int(os.getenv("MAX_FRAMES", "15"))
    dhash_distance: int = int(os.getenv("DHASH_DISTANCE", "6"))

    # Decode on GPU when the host has NVDEC. Without an explicit output format
    # ffmpeg downloads decoded frames back to system memory, so every software
    # filter downstream keeps working — safe to leave on where supported.
    hwaccel: str = os.getenv("HWACCEL", "")  # "cuda" | "videotoolbox" | ""

    # Whisper's upload limit is 25 MB; mono 16 kHz @ 32 kbps keeps ~100 min
    # under it while staying well above what the model needs.
    audio_bitrate: str = os.getenv("AUDIO_BITRATE", "32k")

    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))


settings = Settings()


def make_client():
    """Single place the API endpoint is decided, so swapping providers is a
    config change rather than a search-and-replace across the pipeline."""
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
