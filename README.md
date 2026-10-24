# video-to-json

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A small, runnable reference implementation of a video → structured-JSON pipeline:
scene-aware frame extraction (FFmpeg), transcription with segment timestamps
(Whisper), and multimodal extraction into a schema-validated object (GPT-4o
structured outputs).

It is deliberately compact — the point is the engineering decisions, not the
line count.

```
video ─┬─► FFmpeg scene detect ─► perceptual-hash dedup ─► cap ─► 8-15 frames ─┐
       │                                                                       ├─► GPT-4o ─► VideoInsight (Pydantic-validated JSON)
       └─► FFmpeg extract audio ─► Whisper (segment timestamps) ─► transcript ─┘
                                              └─► SRT (burn-in ready)
```

## Run it

Needs Python 3.9+ and `ffmpeg`/`ffprobe` on PATH (`brew install ffmpeg` on macOS,
`apt install ffmpeg` on Debian/Ubuntu).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY
python -m src.cli samples/demo.mp4 --out result.json --srt demo.srt
```

As an async service:

```bash
uvicorn src.api:app --reload
curl -F file=@samples/demo.mp4 localhost:8000/jobs      # -> {"job_id": "...", "status": "queued"}
curl localhost:8000/jobs/<job_id>
```

Or `docker build -t video-to-json . && docker run -e OPENAI_API_KEY=... -p 8000:8000 video-to-json`.

## The four decisions that matter

**1. Frame selection is the cost lever, not the model call.**
A naive pipeline sends one frame per second: 60 near-identical images for a
60-second clip. Here, FFmpeg scene detection keeps only frames where the picture
actually changed, a difference hash drops the near-duplicates that slip through,
and a hard cap with even time-spreading guarantees a pathological video cannot
blow the budget. Typical short-form clip: **8–15 frames**. `src/pipeline/frames.py`

**2. The schema is the contract.**
The Pydantic model *is* the response format passed to the API, so a malformed
shape is impossible by construction. The prompt then forbids inference — a vision
model asked "what is this product" will invent a price; asked to report only what
is legible or spoken it returns `null`, which is the answer a downstream
e-commerce system can actually act on. `src/schemas.py`, `src/pipeline/extract.py`

**3. Audio is normalised before it becomes a problem.**
Whisper caps uploads at 25 MB. Audio is always transcoded to mono 16 kHz at a low
bitrate — under the limit for ~100 minutes and exactly the format the model was
trained on — with time-based chunking and offset stitching beyond that.
Segment-level timestamps are kept, which is what makes burned-in subtitles
possible downstream. `src/pipeline/audio.py`

**4. Long work does not live in a request.**
Submit returns `202` with a job id; the caller polls. Swapping the in-process
executor for Redis and a worker pool is a configuration change, not a rewrite,
because the job record is already the unit of work. `src/api.py`

## Provider is configuration, not code

`OPENAI_BASE_URL` points the whole pipeline at any OpenAI-compatible endpoint —
the official API, a self-hosted gateway, vLLM, or a regional provider. One place
in `src/config.py` builds the client, so switching providers never touches
pipeline code:

```bash
OPENAI_BASE_URL=https://api.openai.com/v1     # default
OPENAI_BASE_URL=http://localhost:8000/v1      # vLLM / LM Studio
```

`OUTPUT_LANGUAGE` works the same way: `source` (default) keeps the video's own
language, which is what a catalogue serving a local market wants; a code such as
`en` normalises everything instead.

One warning learned the hard way: an output-language instruction placed only in
the system prompt is ignored. The field descriptions inside the JSON schema win,
and everything comes back in English. The rule has to appear in the system
prompt, in the relevant field descriptions, *and* as the last thing the model
reads. Structured outputs makes the schema the strongest instruction in the
request — useful to know before debugging it as a model failure.

That matters more than it looks: it keeps the engine deployable where
api.openai.com is slow or unreachable, and it makes cost renegotiation a
one-line change rather than a migration.

## Operational notes

- Every run logs an estimated per-video cost (frames, vision, Whisper). Knowing
  that number per video is the difference between an MVP that scales and one that
  quietly runs up a bill.
- `HWACCEL=cuda` moves decode onto NVDEC where the host supports it. Because no
  explicit output format is set, frames are downloaded back to system memory and
  every software filter downstream keeps working — safe to leave on.
- Temp directories are removed in a `finally`. Filling a disk with orphaned
  frame dumps is the classic failure mode of a media pipeline under load.
- `confidence < 0.6` is the signal to route a video to human review rather than
  trust the extracted fields.

## Where this comes from

The design mirrors a production AI video system I built and operate that has
rendered **several hundred thousand videos**, with GPU-accelerated FFmpeg and
Whisper-driven subtitle alignment. This repository is the extraction half of that
architecture, stripped to something you can read in ten minutes and run in one.

— Thanh
