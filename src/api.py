"""Async job API.

Video processing takes tens of seconds to minutes, so it cannot live inside a
request. Submit returns immediately with an id; the caller polls. Swapping the
in-process executor below for Redis + a worker pool is a configuration change,
not a rewrite — the job record is already the unit of work.

    uvicorn src.api:app --reload
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from .run import process_video

app = FastAPI(title="AI Media Extraction Engine")
_pool = ThreadPoolExecutor(max_workers=2)
_jobs: Dict[str, dict] = {}


class JobRef(BaseModel):
    job_id: str
    status: str


@app.post("/jobs", response_model=JobRef, status_code=202)
async def submit(file: UploadFile, background: BackgroundTasks) -> JobRef:
    job_id = uuid.uuid4().hex
    dest = Path(tempfile.gettempdir()) / f"{job_id}-{file.filename}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    _jobs[job_id] = {"status": "queued", "result": None, "error": None}
    background.add_task(_run, job_id, dest)
    return JobRef(job_id=job_id, status="queued")


def _run(job_id: str, video: Path) -> None:
    _jobs[job_id]["status"] = "processing"
    try:
        _jobs[job_id]["result"] = process_video(video)
        _jobs[job_id]["status"] = "done"
    except Exception as exc:
        _jobs[job_id]["error"] = str(exc)
        _jobs[job_id]["status"] = "failed"
    finally:
        video.unlink(missing_ok=True)


@app.get("/jobs/{job_id}")
async def status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {"job_id": job_id, **job}


@app.get("/health")
async def health() -> dict:
    return {"ok": True}
