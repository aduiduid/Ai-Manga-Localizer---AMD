from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from .config import AppPaths
from .pipeline import LocalizerPipeline, PipelineRequest


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds_between(start: str | None, end: str) -> float:
    """Seconds from an ISO stamp to ``end``; 0 when the stamp is unusable."""
    if not start:
        return 0.0
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except ValueError:
        return 0.0


def _close_pause(job: dict, now: str) -> None:
    """Fold an open pause window into ``paused_seconds`` so it stops counting."""
    if job.get("paused_at"):
        job["paused_seconds"] = round(
            job.get("paused_seconds", 0.0) + _seconds_between(job["paused_at"], now), 3
        )
        job["paused_at"] = None


class JobCancelled(Exception):
    """Raised inside the worker once the user asked to end a job."""


# Statuses whose worker thread is still alive inside this process.
LIVE_STATUSES = {"queued", "running", "paused"}

# Statuses after which the elapsed clock stops.
TERMINAL_STATUSES = {"complete", "failed", "cancelled"}


class _JobGate:
    """Pause/cancel latch shared between the API thread and the worker.

    The pipeline has no suspend points of its own, so the worker blocks inside
    the progress callback: that fires between pages (OCR, render) and between
    chunks (translate), which is the smallest resumable unit.  Blocking here
    keeps the loaded models and the cached pages intact, so resuming continues
    instead of restarting.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._paused = False
        self._cancelled = False

    @property
    def paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def cancelled(self) -> bool:
        with self._condition:
            return self._cancelled

    def pause(self) -> bool:
        with self._condition:
            if self._cancelled or self._paused:
                return False
            self._paused = True
            return True

    def resume(self) -> bool:
        with self._condition:
            if not self._paused:
                return False
            self._paused = False
            self._condition.notify_all()
            return True

    def cancel(self) -> None:
        with self._condition:
            self._cancelled = True
            self._paused = False
            self._condition.notify_all()

    def wait_while_paused(self) -> None:
        with self._condition:
            while self._paused and not self._cancelled:
                self._condition.wait()
            if self._cancelled:
                raise JobCancelled()


class JobStore:
    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, job_id: str) -> Path:
        return self.folder / f"{job_id}.json"

    def create(self, request: PipelineRequest, total: int = 0) -> dict:
        job_id = uuid.uuid4().hex[:12]
        now = utc_now()
        job = {
            "id": job_id,
            "status": "queued",
            "phase": "queued",
            "progress": 0,
            "current": 0,
            "total": total,
            "message": "等待开始",
            "error": None,
            "created_at": now,
            "updated_at": now,
            # Wall-clock bookkeeping for the sidebar timer.  started_at is set
            # by the first "running" save; finished_at by the first terminal
            # one, so paused_seconds can be subtracted from the elapsed time.
            "started_at": None,
            "finished_at": None,
            "paused_seconds": 0.0,
            "paused_at": None,
            "request": {
                "source": str(request.source),
                "output": str(request.output),
                "target_language": request.target_language,
                "story_context": request.story_context,
                "context_pages": request.context_pages,
                "preserve_sfx": request.preserve_sfx,
                "quality_profile": request.quality_profile,
                "output_format": request.output_format,
                "reviewed_transcript": (
                    str(request.reviewed_transcript)
                    if request.reviewed_transcript
                    else None
                ),
                "device": request.device,
                "inference_backend": request.inference_backend,
                "ocr_backend": request.ocr_backend,
                "ollama_base_url": request.ollama_base_url,
                "ollama_model": request.ollama_model,
                "ollama_ocr_model": request.ollama_ocr_model,
                "online_base_url": request.online_base_url,
                "online_model": request.online_model,
                "glossary": request.glossary or {},
            },
            "result": None,
        }
        self.save(job)
        return job

    def save(self, job: dict) -> None:
        with self._lock:
            now = utc_now()
            job["updated_at"] = now
            # Every state change funnels through here, so the clock bookkeeping
            # lives here too: one guard instead of a call at every transition.
            status = job.get("status")
            if status == "running" and not job.get("started_at"):
                job["started_at"] = now
            if status == "paused":
                if not job.get("paused_at"):
                    job["paused_at"] = now
            else:
                _close_pause(job, now)
            if status in TERMINAL_STATUSES and not job.get("finished_at"):
                job["finished_at"] = now
            self._path(job["id"]).write_text(
                json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def get(self, job_id: str) -> dict | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        jobs = []
        for path in self.folder.glob("*.json"):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)


class JobService:
    def __init__(self, paths: AppPaths, pipeline_factory=LocalizerPipeline):
        self.paths = paths.ensure()
        self.store = JobStore(paths.jobs)
        self.pipeline_factory = pipeline_factory
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="manga-localizer"
        )
        self._gates: dict[str, _JobGate] = {}
        self._gate_lock = threading.RLock()
        self._adopt_orphans()

    def _adopt_orphans(self) -> None:
        """A live status without a gate belongs to a previous process."""
        for job in self.store.list():
            if job.get("status") in LIVE_STATUSES:
                job.update(
                    status="cancelled",
                    phase="cancelled",
                    message="服务重启，任务已中断",
                    error=None,
                )
                self.store.save(job)

    def create_job(self, request: PipelineRequest, total: int = 0) -> dict:
        """Write the queued record and wire its gate, without starting work.

        Split out of submit() so a batch can show its whole queue up front and
        hand each record to the worker only when that folder's turn comes.
        """
        # Store the page count before the worker can start, avoiding an API-side
        # follow-up save that could race and overwrite live progress fields.
        job = self.store.create(request, total=total)
        with self._gate_lock:
            self._gates[job["id"]] = _JobGate()
        return job

    def start(self, job_id: str, request: PipelineRequest):
        """Hand an already-created record to the single worker thread."""
        return self.executor.submit(self._run, job_id, request)

    def discard(self, job_id: str) -> None:
        """Forget the gate of a record that will never be started.

        A queued job dropped by its batch never reaches _run(), which is the one
        place that would otherwise pop the gate.
        """
        with self._gate_lock:
            self._gates.pop(job_id, None)

    def submit(self, request: PipelineRequest, total: int = 0) -> dict:
        job = self.create_job(request, total=total)
        self.start(job["id"], request)
        return job

    def pause(self, job_id: str) -> dict | None:
        """Freeze the worker at its next progress tick.

        Models stay loaded and every page already written to the work folder
        stays reusable, so resuming continues from the same page.
        """
        job = self.store.get(job_id)
        if job is None:
            return None
        with self._gate_lock:
            gate = self._gates.get(job_id)
        if gate is None or not gate.pause():
            return job
        job.update(status="paused", message="已暂停")
        self.store.save(job)
        return job

    def resume(self, job_id: str) -> dict | None:
        job = self.store.get(job_id)
        if job is None:
            return None
        with self._gate_lock:
            gate = self._gates.get(job_id)
        if gate is None or not gate.resume():
            return job
        job.update(status="running", message="继续运行")
        self.store.save(job)
        return job

    def cancel(self, job_id: str) -> dict | None:
        """Ask the worker to stop at its next progress tick.

        The pipeline has no safe suspend point, so this is cooperative: the flag
        is polled inside the progress callback, which fires at least once per
        page.  Cached OCR survives, so a later run resumes instead of restarting.
        """
        job = self.store.get(job_id)
        if job is None:
            return None
        if job["status"] not in LIVE_STATUSES:
            return job
        with self._gate_lock:
            gate = self._gates.get(job_id)
        if gate is not None:
            gate.cancel()
        job.update(status="cancelled", phase="cancelled", message="已结束")
        self.store.save(job)
        return job

    def _run(self, job_id: str, request: PipelineRequest) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        with self._gate_lock:
            gate = self._gates.get(job_id) or _JobGate()
        # Cancelled before the worker got here (an abandoned batch entry): stop
        # now, so no OCR or render work is started for a job nobody wants.
        if gate.cancelled:
            job.update(status="cancelled", phase="cancelled", message="已结束")
            self.store.save(job)
            with self._gate_lock:
                self._gates.pop(job_id, None)
            return
        # A pause issued while the job was still queued must survive the start.
        if not gate.paused:
            job.update(status="running", phase="prepare", message="准备本地化")
            self.store.save(job)
        else:
            job.update(status="paused", message="已暂停")
            self.store.save(job)

        def progress(phase: str, current: int, total: int, message: str) -> None:
            # Blocking here is the pause: it is the narrowest point between
            # resumable units, so nothing half-rendered is left behind.
            gate.wait_while_paused()
            active = self.store.get(job_id)
            if active is None:
                return
            value = round(current / total * 100) if total else 0
            # Each phase occupies a stable part of the overall progress bar.
            offsets = {"ocr": (0, 45), "translate": (45, 25), "render": (70, 30)}
            if phase in offsets:
                base, span = offsets[phase]
                value = base + round((current / total if total else 0) * span)
            elif phase == "complete":
                value = 100
            active.update(
                status="running" if phase != "complete" else "complete",
                phase=phase,
                progress=value,
                current=current,
                total=total,
                message=message,
            )
            self.store.save(active)

        try:
            result = self.pipeline_factory(self.paths).run(request, progress)
            job = self.store.get(job_id) or job
            job.update(
                status="complete",
                phase="complete",
                progress=100,
                message="本地化完成",
                result=result,
            )
        except JobCancelled:
            # cancel() already wrote the cancelled state; keep its wording.
            job = self.store.get(job_id) or job
            job.update(
                status="cancelled", phase="cancelled", message="已结束", error=None
            )
        except Exception as exc:
            job = self.store.get(job_id) or job
            job.update(
                status="failed", phase="failed", error=str(exc), message="任务失败"
            )
        finally:
            with self._gate_lock:
                self._gates.pop(job_id, None)
        self.store.save(job)
