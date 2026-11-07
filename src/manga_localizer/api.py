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
    ocr_backend: Literal["builtin", "hybrid", "ollama"] = "hybrid"
    inference_backend: Literal["builtin", "ollama", "online"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = DEFAULT_LOCAL_TRANSLATION_MODEL
    ollama_ocr_model: str = DEFAULT_OLLAMA_VISION_MODEL
    online_base_url: str = "https://api.openai.com/v1"
    online_model: str = ""
    online_api_key: str | None = Field(default=None, max_length=2048)


class JobPayload(SettingsPayload):
    source: str
    output: str
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


def create_app(paths: AppPaths | None = None, pipeline_factory=None) -> FastAPI:
    paths = (paths or AppPaths.from_env()).ensure()
    models = ModelManager(paths)
    jobs = (
        JobService(paths, pipeline_factory) if pipeline_factory else JobService(paths)
    )
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

    @app.post("/api/jobs")
    def create_job(payload: JobPayload):
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
        settings = SettingsPayload(
            **payload.model_dump(exclude={"source", "output", "reviewed_transcript"})
        )
        active = save_settings_payload(settings)
        api_key = payload.online_api_key
        if api_key is None:
            api_key = secrets["online_api_key"]
        request = PipelineRequest(
            source=source,
            output=output,
            target_language=payload.target_language,
            source_language=payload.source_language,
            story_context=payload.story_context,
            context_pages=payload.context_pages,
            preserve_sfx=payload.preserve_sfx,
            quality_profile=payload.quality_profile,
            output_format=payload.output_format,
            device=payload.device,
            cpu_threads=active.cpu_threads,
            inference_backend=active.inference_backend,
            ocr_backend=active.ocr_backend,
            ollama_base_url=active.ollama_base_url,
            ollama_model=active.ollama_model,
            ollama_ocr_model=active.ollama_ocr_model,
            online_base_url=active.online_base_url,
            online_model=active.online_model,
            online_api_key=api_key or "",
            reviewed_transcript=(
                Path(payload.reviewed_transcript).expanduser().resolve()
                if payload.reviewed_transcript
                else None
            ),
        )
        return jobs.submit(request, total=image_count)

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": jobs.store.list()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/api/jobs/{job_id}/pause")
    def pause_job(job_id: str):
        job = jobs.pause(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: str):
        job = jobs.resume(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
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
    return app


app = create_app()
