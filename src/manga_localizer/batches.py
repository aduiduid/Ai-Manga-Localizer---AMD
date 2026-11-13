"""Queue several folders as one batch.

One folder is the unit the pipeline understands, but a reader usually owns a
volume set.  A batch is therefore a list of folders plus a single consumer
thread that runs them one after another.  It deliberately reuses JobService for
the per-folder work: every folder still gets its own job record (so preview,
pause, cancel and the history list keep working), and anything that already
knows how to talk about a job keeps working inside a batch too.

Why a thread instead of asyncio: the pipeline is synchronous, CPU/GPU bound and
already runs on JobService's single worker thread.  The consumer simply submits
one job and blocks on its future, which is the strongest possible guarantee that
two folders never touch the GPU at once.
"""

from __future__ import annotations

import gc
import json
import queue
import sys
import threading
import uuid
from copy import deepcopy
from pathlib import Path

from .config import AppPaths, UserSettings
from .jobs import LIVE_STATUSES, TERMINAL_STATUSES, JobService, utc_now
from .llama_server import (
    hold_llama_servers,
    running_llama_slots,
    stop_llama_servers,
)
from .pipeline import PipelineRequest
from .translator import release_ollama_model

# A batch is live while its consumer thread still owns it.  There is no paused
# batch: pausing freezes the current folder, not the queue.
BATCH_LIVE_STATUSES = {"queued", "running"}
BATCH_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def make_task(index: int, source: Path, output: Path, images: int) -> dict:
    """One queued folder, with the same status words a job uses."""
    return {
        "index": index,
        "name": source.name,
        "source": str(source),
        "output": str(output),
        "job_id": None,
        "status": "queued",
        "phase": "queued",
        "progress": 0,
        "current": 0,
        "total": images,
        "message": "等待开始",
        "error": None,
        "started_at": None,
        "finished_at": None,
    }


def summarize(batch: dict) -> None:
    """Recount finished folders from the task statuses, the source of truth."""
    tasks = batch.get("tasks") or []
    batch["completed"] = sum(1 for task in tasks if task.get("status") == "complete")
    batch["failed"] = sum(1 for task in tasks if task.get("status") == "failed")
    batch["cancelled"] = sum(1 for task in tasks if task.get("status") == "cancelled")


def batch_progress(tasks: list[dict], total: int) -> int:
    """Whole-batch percentage: finished folders plus the running one's share.

    Folders dropped from the queue are not counted as done: doing so would jump
    the bar forward for work that was never performed.
    """
    if total <= 0:
        return 0
    done = 0.0
    for task in tasks:
        status = task.get("status")
        if status in {"complete", "failed"}:
            done += 1
        elif status == "running":
            done += min(max(int(task.get("progress") or 0), 0), 100) / 100
    return round(done / total * 100)


def queued_tasks(batch: dict) -> list[dict]:
    """The folders that still count as part of the run.

    A folder the reader dropped from the queue is cancelled before it ever
    starts, so it is not work the progress bar or the counter should wait for.
    Dropping one folder therefore shrinks the denominator from, say, 4 to 3
    instead of leaving a permanently unfinished quarter.
    """
    return [task for task in batch.get("tasks") or [] if task.get("status") != "cancelled"]


def queue_position(tasks: list[dict], index: int) -> int:
    """1-based position of folder ``index`` among the folders still queued.

    The UI names the running folder "1/3", "2/3", so the number has to skip the
    rows the reader dropped rather than reuse the raw list index.
    """
    position = 0
    for task in tasks:
        if task.get("status") == "cancelled":
            continue
        position += 1
        if task.get("index") == index:
            return position
    return position


def batch_current_task(batch: dict, tasks: list[dict]) -> dict | None:
    """The folder the batch card should speak about right now.

    While a batch runs that is the folder being processed, so the UI can show
    its phase and page count instead of only the queue position.  Once the
    batch is over, a failed folder is kept as the subject so its error stays
    visible after the run; a fully successful batch has nobody to talk about.
    """
    status = batch.get("status")
    index = batch.get("current_index")
    if status == "running" and isinstance(index, int) and 0 <= index < len(tasks):
        return tasks[index]
    if status in BATCH_TERMINAL_STATUSES:
        for task in reversed(tasks):
            if task.get("status") == "failed":
                return task
    return None


def release_runtime(settings: UserSettings | None) -> list[str]:
    """Give back what a finished batch was holding, and report what was freed.

    Ollama keeps a model resident for five minutes after its last request, and
    the pipeline's Paddle/torch objects only disappear at the next garbage
    collection, so a batch that just processed several folders would otherwise
    keep gigabytes allocated until the app is closed.  Best effort throughout:
    releasing for a status line must never fail the batch.

    llama.cpp is the opposite problem: it holds the VRAM as long as its process
    lives, so a batch that just finished would pin the whole GPU until the app
    is closed.  Stopping the servers here is what hands the VRAM back, and it
    runs for the cancelled case too, because that is exactly when the user is
    looking at the GPU and found it still busy.

    ``torch`` is only touched when it is already imported, because importing it
    here could land before Paddle and break the DLL load order elsewhere.
    """
    released: list[str] = []
    if settings is not None:
        # Either stage can be the one that started a llama-server: OCR reads
        # pages through it too, so the translation backend alone is not the test.
        if settings.inference_backend in {"llama", "llama_7b"} or settings.ocr_backend == "llama":
            live = running_llama_slots()
            for slot in live:
                released.append(f"{slot.alias}@{slot.port}")
            stop_llama_servers()
        models: list[str] = []
        if settings.inference_backend == "ollama":
            models.append(settings.ollama_model)
        if settings.ocr_backend in {"hybrid", "ollama"}:
            models.append(settings.ollama_ocr_model)
        for model in dict.fromkeys(models):
            if model and release_ollama_model(settings.ollama_base_url, model):
                released.append(model)
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return released


class BatchStore:
    """One JSON file per batch, next to the per-folder job records."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, batch_id: str) -> Path:
        return self.folder / f"{batch_id}.json"

    def create(self, tasks: list[dict], duplicates: int = 0, output_root: str = "") -> dict:
        batch_id = uuid.uuid4().hex[:12]
        now = utc_now()
        batch = {
            "id": batch_id,
            "kind": "batch",
            # Kept so a reloaded page can show the same source folders and the
            # same output root the batch was queued with. Without it the queue
            # would come back but the output root would look empty, and the next
            # batch would silently write beside the sources.
            "request": {
                "source": [task["source"] for task in tasks],
                "output": output_root,
            },
            "status": "queued",
            "message": "等待开始",
            "progress": 0,
            "total": len(tasks),
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "current_index": None,
            "duplicates_removed": duplicates,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "released": [],
            "tasks": tasks,
        }
        self.save(batch)
        return batch

    def save(self, batch: dict) -> None:
        with self._lock:
            now = utc_now()
            batch["updated_at"] = now
            # Same clock bookkeeping as JobStore.save: one guard here instead of
            # a call at every transition.
            status = batch.get("status")
            if status == "running" and not batch.get("started_at"):
                batch["started_at"] = now
            if status in BATCH_TERMINAL_STATUSES and not batch.get("finished_at"):
                batch["finished_at"] = now
            self._path(batch["id"]).write_text(
                json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def get(self, batch_id: str) -> dict | None:
        path = self._path(batch_id)
        if not path.exists():
            return None
        with self._lock:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None

    def list(self) -> list[dict]:
        batches = []
        for path in self.folder.glob("*.json"):
            try:
                batches.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(
            batches, key=lambda item: item.get("created_at", ""), reverse=True
        )


class BatchService:
    """Serial consumer that turns one batch into a sequence of jobs."""

    def __init__(self, paths: AppPaths, jobs: JobService):
        self.paths = paths.ensure()
        self.jobs = jobs
        self.store = BatchStore(paths.batches)
        self._queue: queue.Queue[str | None] = queue.Queue()
        # Per-folder request options and the settings for the release step.  Kept
        # in memory only: the batch record is written to disk and must never
        # carry the online API key.
        self._templates: dict[str, dict] = {}
        self._current: dict[str, str] = {}
        self._lock = threading.RLock()
        self._adopt_orphans()
        self._worker = threading.Thread(
            target=self._consume, name="manga-localizer-batch", daemon=True
        )
        self._worker.start()

    def _adopt_orphans(self) -> None:
        """A live batch without a consumer thread belongs to a previous process."""
        for batch in self.store.list():
            if batch.get("status") not in BATCH_LIVE_STATUSES:
                continue
            for task in batch.get("tasks") or []:
                if task.get("status") in LIVE_STATUSES:
                    task.update(
                        status="cancelled",
                        phase="cancelled",
                        message="服务重启，批次已中断",
                        error=None,
                    )
            batch.update(
                status="cancelled",
                message="服务重启，批次已中断",
                current_index=None,
            )
            self.store.save(batch)

    def submit(
        self,
        tasks: list[dict],
        options: dict,
        settings: UserSettings,
        duplicates: int = 0,
        output_root: str = "",
    ) -> dict:
        """Queue a validated batch; returns the record for the API response."""
        batch = self.store.create(tasks, duplicates=duplicates, output_root=output_root)
        with self._lock:
            self._templates[batch["id"]] = {
                "options": options,
                "settings": settings,
            }
        self._queue.put(batch["id"])
        return self.view(batch)

    def view(self, batch: dict) -> dict:
        """Copy of the record with each folder's live progress merged in.

        The job records are the source of truth while a folder runs, so the
        batch file only has to remember the queue, not mirror live progress.
        """
        merged = deepcopy(batch)
        tasks = merged.get("tasks") or []
        for task in tasks:
            job_id = task.get("job_id")
            if not job_id:
                continue
            job = self.jobs.store.get(job_id)
            if not job:
                continue
            for field in (
                "status",
                "phase",
                "progress",
                "current",
                "total",
                "message",
                "error",
                "started_at",
                "finished_at",
            ):
                if field in job:
                    task[field] = job[field]
        summarize(merged)
        merged["total"] = len(tasks)
        merged["effective_total"] = len(queued_tasks(merged))
        merged["finished"] = merged["completed"] + merged["failed"]
        merged["queue_position"] = (
            queue_position(tasks, merged["current_index"])
            if isinstance(merged.get("current_index"), int)
            else None
        )
        merged["progress"] = batch_progress(tasks, merged["effective_total"])
        current = batch_current_task(merged, tasks)
        merged["current_task"] = deepcopy(current) if current is not None else None
        merged["current_batch_index"] = current.get("index") if current else None
        merged["total_batch_size"] = len(tasks)
        return merged

    def get(self, batch_id: str) -> dict | None:
        batch = self.store.get(batch_id)
        return self.view(batch) if batch else None

    def list(self) -> list[dict]:
        return [self.view(batch) for batch in self.store.list()]

    def cancel(self, batch_id: str) -> dict | None:
        """Stop the folder that is running and drop the ones still queued."""
        batch = self.store.get(batch_id)
        if batch is None:
            return None
        if batch.get("status") in BATCH_TERMINAL_STATUSES:
            return self.view(batch)
        with self._lock:
            job_id = self._current.get(batch_id)
        if job_id:
            # Cooperative: the job raises JobCancelled at its next progress tick,
            # which is what actually stops the OCR/render work on the GPU.
            self.jobs.cancel(job_id)
        for task in batch.get("tasks") or []:
            if task.get("job_id") == job_id:
                continue
            if task.get("status") in LIVE_STATUSES:
                task.update(
                    status="cancelled",
                    phase="cancelled",
                    message="已取消",
                    error=None,
                )
        batch.update(status="cancelled", message="已取消", current_index=None)
        self.store.save(batch)
        return self.view(batch)

    def cancel_task(self, batch_id: str, index: int) -> dict | None:
        """Drop one folder that has not started yet.

        Refused for the folder that is already running: mid-pipeline cancellation
        has to go through the job's cooperative gate, which is what ``cancel``
        does.  Removing a waiting folder is only a status flip, and the consumer
        loop re-reads the record before every step, so the row is simply skipped.

        The read-modify-write happens under the service lock because the
        consumer thread writes the same record when a folder finishes; without
        it a cancel landing in that window would be silently undone.
        """
        with self._lock:
            batch = self.store.get(batch_id)
            if batch is None:
                return None
            if batch.get("status") in BATCH_TERMINAL_STATUSES:
                raise ValueError("批次已结束，无法取消排队中的任务")
            tasks = batch.get("tasks") or []
            task = next((item for item in tasks if item.get("index") == index), None)
            if task is None:
                raise ValueError("任务不存在")
            status = task.get("status") or "queued"
            if status == "running":
                raise ValueError("无法取消正在运行的任务，请取消整批")
            if status != "queued":
                raise ValueError(f"任务状态为 {status}，无需取消")
            task.update(
                status="cancelled",
                phase="cancelled",
                progress=0,
                current=0,
                message="已取消排队",
                error=None,
                finished_at=utc_now(),
            )
            # ``total`` keeps counting every row ever queued; the view derives
            # ``effective_total`` so the counter and the bar forget this row.
            summarize(batch)
            self.store.save(batch)
            return self.view(batch)

    def pause(self, batch_id: str) -> dict | None:
        batch = self.store.get(batch_id)
        if batch is None:
            return None
        with self._lock:
            job_id = self._current.get(batch_id)
        if job_id:
            self.jobs.pause(job_id)
        batch["message"] = "已暂停"
        self.store.save(batch)
        return self.view(batch)

    def resume(self, batch_id: str) -> dict | None:
        batch = self.store.get(batch_id)
        if batch is None:
            return None
        with self._lock:
            job_id = self._current.get(batch_id)
        if job_id:
            self.jobs.resume(job_id)
        batch["message"] = "继续运行"
        self.store.save(batch)
        return self.view(batch)

    def _consume(self) -> None:
        while True:
            batch_id = self._queue.get()
            if batch_id is None:
                return
            try:
                self._run_batch(batch_id)
            except Exception as exc:  # pragma: no cover - defensive
                batch = self.store.get(batch_id)
                if batch is not None:
                    batch.update(status="failed", message=f"批次失败：{exc}")
                    self.store.save(batch)
            finally:
                self._queue.task_done()

    def _run_batch(self, batch_id: str) -> None:
        # Holding the servers for the whole queue keeps the 7 GB model loaded
        # between folders; the per-job release then becomes a no-op instead of
        # stopping and restarting the server for every single folder.
        with hold_llama_servers():
            self._run_batch_locked(batch_id)

    def _run_batch_locked(self, batch_id: str) -> None:
        batch = self.store.get(batch_id)
        if batch is None or batch.get("status") in BATCH_TERMINAL_STATUSES:
            return
        with self._lock:
            template = self._templates.pop(batch_id, None)
        batch.update(status="running", message="开始处理")
        self.store.save(batch)
        index = 0
        while index < len(batch.get("tasks") or []):
            batch = self.store.get(batch_id)
            if batch is None or batch.get("status") == "cancelled":
                break
            if (batch["tasks"][index].get("status") or "queued") == "queued":
                self._run_task(batch_id, index, template or {})
            index += 1
        batch = self.store.get(batch_id)
        if batch is None:
            return
        if batch.get("status") != "cancelled":
            summarize(batch)
            dropped = batch["cancelled"]
            if not batch["completed"] and not batch["failed"] and dropped:
                # Every folder was dropped from the queue before it started, so
                # there is nothing to report as success or failure.
                batch["status"] = "cancelled"
                batch["message"] = "已取消"
            else:
                # A folder that failed is recorded and skipped; the batch only
                # counts as failed when nothing came out of it.
                batch["status"] = "failed" if not batch["completed"] else "completed"
                batch["message"] = (
                    f"成功 {batch['completed']} 个 / 失败 {batch['failed']} 个"
                )
                if dropped:
                    batch["message"] += f" / 已取消 {dropped} 个"
            batch["current_index"] = None
        settings = (template or {}).get("settings")
        batch["released"] = release_runtime(settings)
        self.store.save(batch)

    def _run_task(self, batch_id: str, index: int, template: dict) -> None:
        batch = self.store.get(batch_id)
        if batch is None:
            return
        task = batch["tasks"][index]
        options = dict(template.get("options") or {})
        request = PipelineRequest(
            source=Path(task["source"]),
            output=Path(task["output"]),
            **options,
        )
        job = self.jobs.create_job(request, total=task["total"])
        task.update(
            job_id=job["id"],
            status="running",
            phase="prepare",
            message="准备本地化",
            started_at=utc_now(),
        )
        batch["current_index"] = index
        # Count only the folders that are still part of the run: rows the reader
        # dropped from the queue must not leave a gap in "1/3, 2/3".
        batch["message"] = (
            f"正在处理 {queue_position(batch['tasks'], index)}"
            f"/{len(queued_tasks(batch))}：{task['name']}"
        )
        self.store.save(batch)
        with self._lock:
            self._current[batch_id] = job["id"]
        try:
            self.jobs.start(job["id"], request).result()
        except Exception as exc:  # pragma: no cover - the worker records its own
            final = {"status": "failed", "error": str(exc), "message": "任务失败"}
        else:
            final = self.jobs.store.get(job["id"]) or {}
        finally:
            with self._lock:
                self._current.pop(batch_id, None)
        # Re-read: a cancel issued while this folder ran must not be undone.
        batch = self.store.get(batch_id) or batch
        task = batch["tasks"][index]
        task.update(
            status=final.get("status") or "failed",
            phase=final.get("phase") or "failed",
            progress=final.get("progress", task.get("progress", 0)),
            current=final.get("current", task.get("current", 0)),
            total=final.get("total") or task.get("total", 0),
            message=final.get("message") or "任务失败",
            error=final.get("error"),
            finished_at=final.get("finished_at") or utc_now(),
        )
        self.store.save(batch)
