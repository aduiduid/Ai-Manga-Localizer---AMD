from __future__ import annotations

import asyncio
import os
import platform
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .batches import BatchService, make_task
from .config import AppPaths, UserSettings
from .hardware import detect_cpu, normalize_cpu_threads
from .jobs import JobService
from .model_manager import ModelManager, OllamaRuntime, ollama_tags
from .ocr import DEFAULT_OLLAMA_VISION_MODEL, list_images
from .pipeline import PipelineRequest
from .translator import (
    DEFAULT_LOCAL_TRANSLATION_MODEL,
    available_remote_models,
    validate_local_model_size,
)


class FolderRequest(BaseModel):
    initial: str | None = None
    output: str | None = None


class SettingsPayload(BaseModel):
    target_language: str = "简体中文"
    source_language: Literal["auto", "en", "ja"] = "auto"
    story_context: bool = True
    context_pages: int = Field(6, ge=0, le=12)
    preserve_sfx: bool = False
    quality_profile: Literal["quality", "fast"] = "quality"
    output_format: Literal["webp", "png"] = "webp"
    prefer_modelscope: bool = True
    device: str = "auto"
    # 0 means "let each library use every logical core", which is the behaviour
    # the app had before the thread budget existed.  The upper bound is only a
    # sanity limit; /api/system publishes the real choices for this machine.
    cpu_threads: int = Field(0, ge=0, le=256)
    notify_on_complete: bool = True
    notify_on_failure: bool = True
    # Percent, so a hand-edited settings.json cannot ask for a gain the Web Audio
    # graph would clip.
    notify_volume: int = Field(70, ge=0, le=100)
    ocr_backend: Literal["builtin", "hybrid", "ollama"] = "hybrid"
    inference_backend: Literal["builtin", "ollama", "online"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = DEFAULT_LOCAL_TRANSLATION_MODEL
    ollama_ocr_model: str = DEFAULT_OLLAMA_VISION_MODEL
    online_base_url: str = "https://api.openai.com/v1"
    online_model: str = ""
    online_api_key: str | None = Field(default=None, max_length=2048)


class JobPayload(SettingsPayload):
    # A single folder keeps the original single-job behaviour; a list queues a
    # batch and runs the folders one after another.
    source: str | list[str]
    # For one folder this is the destination itself; for a batch it is the root
    # holding one "<name>--已汉化" folder per queued source, and may be empty to
    # place each result next to its own source.
    output: str = ""
    reviewed_transcript: str | None = None


class BootstrapPayload(BaseModel):
    model_ids: list[str] = Field(
        default_factory=lambda: ["paddleocr", "manga-ocr", "lama", "hy-mt2"]
    )


class InferenceCheckPayload(BaseModel):
    inference_backend: Literal["builtin", "ollama", "online"]
    ocr_backend: Literal["builtin", "hybrid", "ollama"] = "hybrid"
    quality_profile: Literal["quality", "fast"] = "quality"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = DEFAULT_LOCAL_TRANSLATION_MODEL
    ollama_ocr_model: str = DEFAULT_OLLAMA_VISION_MODEL
    online_base_url: str = "https://api.openai.com/v1"
    online_model: str = ""
    online_api_key: str | None = Field(default=None, max_length=2048)


def _pick_folder(initial: str | None = None) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(initialdir=initial or str(Path.home()))
        root.destroy()
        return chosen
    except Exception as exc:
        raise RuntimeError(f"Native folder picker is unavailable: {exc}") from exc


def _queued_targets(
    sources: list[str], output_root: str
) -> tuple[list[dict], list[str], int]:
    """Validate queued folders and work out where each one writes.

    A path that cannot be processed is reported here rather than halfway
    through a long batch, so the whole queue is checked before any GPU work
    starts.  Returns the tasks, the problems that block the batch, and how many
    duplicate paths were dropped.
    """
    problems: list[str] = []
    tasks: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    for raw in sources:
        text = (raw or "").strip().strip('"')
        if not text:
            continue
        source = Path(text).expanduser()
        # Windows paths are case insensitive, so "D:\Manga" and "d:\manga" are
        # the same folder and must not be translated twice.
        key = str(source).rstrip("\\/").casefold()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        if not source.is_dir():
            problems.append(f"源文件夹不存在：{source}")
            continue
        try:
            images = list_images(source)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if not images:
            problems.append(f"未找到支持的图片：{source}")
            continue
        resolved = source.resolve()
        base = (
            Path(output_root).expanduser() if output_root.strip() else resolved.parent
        )
        output = base / f"{resolved.name}--已汉化"
        if output.resolve() == resolved:
            problems.append(f"输出目录与源文件夹相同：{output}")
            continue
        tasks.append(make_task(len(tasks), resolved, output, len(images)))
    return tasks, problems, duplicates


def create_app(paths: AppPaths | None = None, pipeline_factory=None) -> FastAPI:
    paths = (paths or AppPaths.from_env()).ensure()
    models = ModelManager(paths)
    jobs = (
        JobService(paths, pipeline_factory) if pipeline_factory else JobService(paths)
    )
    batches = BatchService(paths, jobs)
    app = FastAPI(title="Manga Localizer Studio", version=__version__)
    tasks: dict[str, dict] = {}
    task_lock = threading.RLock()
    task_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-bootstrap")
    secrets = {"online_api_key": os.environ.get("MLS_ONLINE_API_KEY", "")}

    def public_settings(active: UserSettings) -> dict:
        return {
            **active.__dict__,
            "online_api_key_configured": bool(secrets["online_api_key"]),
        }

    def save_settings_payload(payload: SettingsPayload) -> UserSettings:
        if payload.online_api_key is not None:
            secrets["online_api_key"] = payload.online_api_key.strip()
        values = payload.model_dump(exclude={"online_api_key"})
        # Clamp to this machine: a settings file copied from a 32-thread desktop
        # must not ask a 4-core laptop for 32 threads.
        values["cpu_threads"] = normalize_cpu_threads(values.get("cpu_threads"))
        active = UserSettings(**values)
        active.save(paths.settings)
        return active

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/system")
    def system_info():
        active = UserSettings.load(paths.settings)
        cpu = detect_cpu()
        return {
            "version": __version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "home": str(paths.root),
            "local_only": active.inference_backend != "online",
            # The settings panel builds its thread dropdown from this, so the
            # numbers come from the machine rather than a hard-coded list.
            "cpu": {
                "name": cpu.name,
                "physical_cores": cpu.physical_cores,
                "logical_cores": cpu.logical_cores,
                "thread_choices": cpu.choices,
            },
            "settings": public_settings(active),
        }

    @app.get("/api/models")
    def model_status(refresh: bool = Query(False)):
        del refresh
        active = UserSettings.load(paths.settings)
        runtime = OllamaRuntime.from_settings(
            active,
            # Only probe the service when it is actually the inference backend;
            # a 3 s dead-socket wait on every page load is not worth it.
            ollama_tags(active.ollama_base_url)
            if active.inference_backend == "ollama"
            else None,
        )
        return {
            "models": models.status(
                active.inference_backend,
                active.quality_profile,
                active.ocr_backend,
                runtime,
            ),
            "source_policy": "ModelScope first",
        }

    def bootstrap(task_id: str, model_ids: list[str]):
        try:
            for index, model_id in enumerate(model_ids):
                models.download(
                    model_id,
                    lambda value, message, idx=index: _update_task(
                        task_id,
                        status="running",
                        progress=round((idx + value / 100) / len(model_ids) * 100),
                        message=message,
                    ),
                )
            _update_task(
                task_id, status="complete", progress=100, message="全部模型已就绪"
            )
        except Exception as exc:
            _update_task(
                task_id, status="failed", error=str(exc), message="模型准备失败"
            )

    def _update_task(task_id: str, **values):
        with task_lock:
            tasks.setdefault(task_id, {}).update(values)

    @app.post("/api/models/bootstrap")
    def start_bootstrap(payload: BootstrapPayload):
        task_id = uuid.uuid4().hex[:12]
        tasks[task_id] = {
            "id": task_id,
            "status": "queued",
            "progress": 0,
            "message": "等待下载",
            "error": None,
        }
        task_pool.submit(bootstrap, task_id, payload.model_ids)
        return tasks[task_id]

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        with task_lock:
            task = tasks.get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        return task

    @app.get("/api/settings")
    def get_settings():
        return public_settings(UserSettings.load(paths.settings))

    @app.put("/api/settings")
    def put_settings(payload: SettingsPayload):
        return public_settings(save_settings_payload(payload))

    @app.post("/api/inference/check")
    def check_inference(payload: InferenceCheckPayload):
        if payload.inference_backend == "builtin" and payload.ocr_backend == "builtin":
            ready = models.is_ready("hy-mt2")
            return {
                "ok": ready,
                "backend": "builtin",
                "message": "Hy-MT2 已就绪" if ready else "Hy-MT2 尚未下载",
                "models": ["hy-mt2"] if ready else [],
            }
        api_key = (
            payload.online_api_key.strip()
            if payload.online_api_key is not None
            else secrets["online_api_key"]
        )
        if (
            payload.ocr_backend in {"hybrid", "ollama"}
            or payload.inference_backend == "ollama"
        ):
            try:
                if payload.inference_backend == "ollama":
                    validate_local_model_size(payload.ollama_model)
                if payload.ocr_backend in {"hybrid", "ollama"}:
                    validate_local_model_size(payload.ollama_ocr_model)
            except ValueError as exc:
                return {
                    "ok": False,
                    "backend": "ollama",
                    "message": str(exc),
                    "models": [],
                }
            try:
                ollama_available = available_remote_models(
                    "ollama", payload.ollama_base_url, ""
                )
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
            required_ollama = []
            if payload.ocr_backend in {"hybrid", "ollama"}:
                required_ollama.append(payload.ollama_ocr_model)
            if payload.inference_backend == "ollama":
                required_ollama.append(payload.ollama_model)
            missing = [
                model for model in required_ollama if model not in ollama_available
            ]
            if missing:
                return {
                    "ok": False,
                    "backend": "ollama",
                    "message": f"已连接，但未找到模型 {', '.join(missing)}",
                    "models": ollama_available,
                }
            if payload.inference_backend != "online":
                required_local = []
                if payload.inference_backend == "builtin" or (
                    payload.inference_backend == "ollama"
                    and payload.quality_profile == "quality"
                ):
                    required_local.append("hy-mt2")
                if payload.ocr_backend != "ollama":
                    required_local.extend(["paddleocr", "manga-ocr"])
                missing_local = [
                    model_id
                    for model_id in required_local
                    if not models.is_ready(model_id)
                ]
                builtin_ready = not missing_local
                return {
                    "ok": builtin_ready,
                    "backend": "ollama",
                    "message": (
                        f"已连接，发现 {len(ollama_available)} 个模型"
                        if builtin_ready
                        else f"Ollama 已连接，但本地组件未就绪：{', '.join(missing_local)}"
                    ),
                    "models": ollama_available,
                }
        base_url = payload.online_base_url
        try:
            available = available_remote_models(
                payload.inference_backend, base_url, api_key
            )
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        selected = (
            payload.ollama_model
            if payload.inference_backend == "ollama"
            else payload.online_model
        )
        return {
            "ok": bool(available) and (not selected or selected in available),
            "backend": payload.inference_backend,
            "message": (
                f"已连接，发现 {len(available)} 个模型"
                if not selected or selected in available
                else f"已连接，但未找到模型 {selected}"
            ),
            "models": available,
        }

    @app.post("/api/dialog/folder")
    async def folder_dialog(payload: FolderRequest):
        try:
            selected = await asyncio.to_thread(_pick_folder, payload.initial)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
        return {"path": selected}

    @app.post("/api/paths/derive-output")
    def derive_output(payload: FolderRequest):
        if not payload.initial:
            return {"path": ""}
        source = Path(payload.initial).expanduser()
        base = Path(payload.output).expanduser() if payload.output else source.parent
        return {"path": str(base / f"{source.name}--已汉化")}

    @app.get("/api/paths/children")
    def list_children(path: str):
        """Sub-folders of ``path`` with their image counts, for bulk adding."""
        folder = Path(path).expanduser()
        if not folder.is_dir():
            raise HTTPException(422, f"文件夹不存在：{folder}")
        try:
            entries = sorted(
                (item for item in folder.iterdir() if item.is_dir()),
                key=lambda item: item.name.casefold(),
            )
        except OSError as exc:
            raise HTTPException(422, f"无法读取文件夹：{folder}（{exc}）") from exc
        folders = []
        for child in entries:
            try:
                images = len(list_images(child))
            except ValueError:
                images = 0
            folders.append(
                {"name": child.name, "path": str(child.resolve()), "images": images}
            )
        return {"path": str(folder.resolve()), "folders": folders}

    def _activate(payload: JobPayload) -> tuple[UserSettings, str]:
        """Persist the submitted settings and resolve the online API key."""
        settings = SettingsPayload(
            **payload.model_dump(exclude={"source", "output", "reviewed_transcript"})
        )
        active = save_settings_payload(settings)
        api_key = payload.online_api_key
        if api_key is None:
            api_key = secrets["online_api_key"]
        return active, api_key or ""

    def _request_options(
        payload: JobPayload, active: UserSettings, api_key: str
    ) -> dict:
        """Everything a PipelineRequest needs except the two folder paths."""
        return {
            "target_language": payload.target_language,
            "source_language": payload.source_language,
            "story_context": payload.story_context,
            "context_pages": payload.context_pages,
            "preserve_sfx": payload.preserve_sfx,
            "quality_profile": payload.quality_profile,
            "output_format": payload.output_format,
            "device": payload.device,
            "cpu_threads": active.cpu_threads,
            "inference_backend": active.inference_backend,
            "ocr_backend": active.ocr_backend,
            "ollama_base_url": active.ollama_base_url,
            "ollama_model": active.ollama_model,
            "ollama_ocr_model": active.ollama_ocr_model,
            "online_base_url": active.online_base_url,
            "online_model": active.online_model,
            "online_api_key": api_key,
            "reviewed_transcript": (
                Path(payload.reviewed_transcript).expanduser().resolve()
                if payload.reviewed_transcript
                else None
            ),
        }

    def _create_batch(payload: JobPayload, sources: list[str]) -> dict:
        if payload.reviewed_transcript:
            raise HTTPException(
                422, "已审校 transcript 只能用于单个文件夹，请先只保留一个源文件夹"
            )
        tasks, problems, duplicates = _queued_targets(sources, payload.output)
        if problems:
            raise HTTPException(422, "；".join(problems))
        if not tasks:
            raise HTTPException(422, "没有可用的文件夹：请至少添加一个包含图片的文件夹")
        active, api_key = _activate(payload)
        return batches.submit(
            tasks,
            _request_options(payload, active, api_key),
            active,
            duplicates=duplicates,
            output_root=payload.output.strip(),
        )

    @app.post("/api/jobs")
    def create_job(payload: JobPayload):
        # A list of folders is a batch; a single path keeps the original
        # response shape, so existing callers do not have to change.
        if isinstance(payload.source, list):
            return _create_batch(payload, payload.source)
        if not payload.output.strip():
            raise HTTPException(422, "Output folder is required")
        source = Path(payload.source).expanduser().resolve()
        output = Path(payload.output).expanduser().resolve()
        if not source.is_dir():
            raise HTTPException(422, "Source folder does not exist")
        try:
            image_count = len(list_images(source))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if image_count == 0:
            raise HTTPException(422, "No supported images found")
        if source == output:
            raise HTTPException(422, "Output folder must differ from source")
        active, api_key = _activate(payload)
        request = PipelineRequest(
            source=source,
            output=output,
            **_request_options(payload, active, api_key),
        )
        return jobs.submit(request, total=image_count)

    @app.post("/api/batches")
    def create_batch(payload: JobPayload):
        sources = (
            payload.source if isinstance(payload.source, list) else [payload.source]
        )
        return _create_batch(payload, sources)

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": jobs.store.list()}

    @app.get("/api/batches")
    def list_batches():
        return {"batches": batches.list()}

    @app.get("/api/batches/{batch_id}")
    def get_batch(batch_id: str):
        batch = batches.get(batch_id)
        if not batch:
            raise HTTPException(404, "Batch not found")
        return batch

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        # Batches share this poll endpoint so the UI can keep one refresh loop
        # and does not need to know which kind of id it is holding.
        batch = batches.get(job_id)
        if batch:
            return batch
        job = jobs.store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/api/jobs/{job_id}/pause")
    def pause_job(job_id: str):
        if batches.store.get(job_id):
            return batches.pause(job_id)
        job = jobs.pause(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: str):
        if batches.store.get(job_id):
            return batches.resume(job_id)
        job = jobs.resume(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        if batches.store.get(job_id):
            return batches.cancel(job_id)
        job = jobs.cancel(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.get("/api/jobs/{job_id}/preview/{kind}/{page_number}")
    def preview(job_id: str, kind: str, page_number: int):
        job = jobs.store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        source = Path(job["request"]["source"])
        output = Path(job["request"]["output"])
        try:
            files = list_images(source)
        except ValueError as error:
            # The source folder can be deleted after the job ran; the UI shows
            # the message instead of a bare 500 in the console.
            raise HTTPException(404, str(error)) from error
        if page_number < 1 or page_number > len(files):
            raise HTTPException(404, "Page not found")
        source_file = files[page_number - 1]
        if kind == "source":
            path = source_file
        elif kind == "output":
            suffix = (
                ".webp"
                if job["request"].get("output_format", "webp") == "webp"
                else ".png"
            )
            path = output / f"{source_file.stem}{suffix}"
        else:
            raise HTTPException(422, "Preview kind must be source or output")
        if not path.exists():
            raise HTTPException(404, "Preview is not ready")
        return FileResponse(path)

    @app.post("/api/shutdown")
    def shutdown():
        """Stop the local server.

        Active jobs are cancelled first: the executor threads are non-daemon,
        so a running pipeline would otherwise keep the process alive after
        uvicorn stops.
        """
        for batch in batches.store.list():
            if batch.get("status") in {"queued", "running"}:
                batches.cancel(batch["id"])
        for job in jobs.store.list():
            if job["status"] in {"queued", "running", "paused"}:
                jobs.cancel(job["id"])
        server = getattr(app.state, "server", None)
        if server is None:
            raise HTTPException(409, "Server handle unavailable")
        # Reply first, then stop: should_exit is checked on the next loop tick.
        threading.Thread(
            target=lambda: (time.sleep(0.4), setattr(server, "should_exit", True)),
            daemon=True,
        ).start()
        return {"ok": True}

    @app.middleware("http")
    async def revalidate_web_assets(request, call_next):
        # The bundled UI has no build step and no content hashes, so browsers
        # heuristically cache app.js/i18n.js and keep rendering raw i18n keys
        # after an upgrade.  Make them revalidate instead.
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.endswith((".js", ".css", ".html")):
            response.headers["Cache-Control"] = "no-cache, must-revalidate"
        return response

    web_dir = Path(__file__).resolve().parent / "web"
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    app.state.paths = paths
    app.state.models = models
    app.state.jobs = jobs
    app.state.batches = batches
    return app


app = create_app()
