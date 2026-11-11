"""Run llama.cpp's ``llama-server`` as the local inference service.

Ollama wraps llama.cpp and keeps a model resident for five minutes; the wrapped
build also hides the process behind its own scheduler.  This module drives
``llama-server`` directly so the app owns the lifecycle: one process per model,
an explicit port, and a stop that really frees the VRAM.

Two facts about the Ollama-bundled build shape this module and cost a long time
to find, so they are recorded where the code that depends on them lives:

* ``ggml-vulkan.dll`` sits in a ``vulkan`` subdirectory and is loaded by Ollama
  under a name of its own.  Run ``llama-server.exe`` from that directory and it
  reports ``no usable GPU found`` and silently falls back to CPU, so the launch
  below stages a private directory whose root holds the exe *and* that DLL.
* The bundled build has no router mode (``--models-dir`` fails with
  ``subprocess is not enabled on this build``), so each model gets its own
  process and its own port.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# llama-server's own default port is 9931, which would collide with nothing the
# user expects; the documented ports are used instead because the UI shows them.
DEFAULT_MAIN_PORT = 8080
DEFAULT_REVIEW_PORT = 8081
# Vision OCR needs its own port: the main slot is busy translating, and a page
# read cannot share a slot that serves one request at a time (``-np 1``).
DEFAULT_VISION_PORT = 8082

# One slot per server, so this is the whole context one request may use.  64K
# matches the Ollama ceiling this migration replaces, and the server advertises
# 262K but a bigger KV cache buys nothing for page-sized prompts.
DEFAULT_CONTEXT_TOKENS = 65_536

# How long a cold start may take.  A 7 GB Q6_K model off an SSD needs about 30 s
# to load on this machine, and a page file under load can double that.
START_TIMEOUT_S = 300.0
STOP_TIMEOUT_S = 15.0


class LlamaServerError(RuntimeError):
    """A server could not be started, or was not there when it was needed."""


@dataclass(frozen=True)
class OllamaModelFiles:
    """The GGUF blobs one Ollama tag resolves to."""

    tag: str
    weights: Path
    projector: Path | None = None


@dataclass(frozen=True)
class LlamaSlot:
    """One running ``llama-server`` instance."""

    role: str
    alias: str
    base_url: str
    port: int
    context: int
    weights: Path
    projector: Path | None = None


def _manifest_blob(blobs: Path, digest: str) -> Path | None:
    name = digest.replace(":", "-")
    candidate = blobs / name
    return candidate if candidate.exists() else None


def discover_ollama_models(manifest_root: Path, blobs_root: Path) -> dict[str, OllamaModelFiles]:
    """Map every Ollama tag under ``manifest_root`` to its GGUF blobs.

    The manifests are the only reliable index: blob filenames are content
    hashes, and the params/config layers next to the weights carry no file name
    the loader could use.
    """
    found: dict[str, OllamaModelFiles] = {}
    if not manifest_root.is_dir():
        return found
    for manifest in manifest_root.rglob("*"):
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        tag = f"{manifest.parent.name}:{manifest.name}"
        weights = None
        projector = None
        for layer in payload.get("layers") or []:
            media = str(layer.get("mediaType") or "")
            blob = _manifest_blob(blobs_root, str(layer.get("digest") or ""))
            if blob is None:
                continue
            if media.endswith(".model"):
                weights = blob
            elif media.endswith(".projector"):
                projector = blob
        if weights is not None:
            found[tag] = OllamaModelFiles(tag=tag, weights=weights, projector=projector)
    return found


def _port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _server_alias(base_url: str, timeout: float = 2.0) -> str | None:
    """The alias served at ``base_url``, or None when nothing answers there."""
    try:
        with urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, HTTPError, URLError):
        return None
    rows = payload.get("data") or []
    if not rows:
        return None
    return str(rows[0].get("id") or "") or None


def _healthy(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as response:
            return response.status == 200
    except (OSError, HTTPError, URLError):
        return False


def _find_free_port(start: int, span: int = 40) -> int:
    for port in range(start, start + span):
        if _port_is_free(port):
            return port
    raise LlamaServerError(
        f"端口 {start}-{start + span - 1} 都被占用，无法启动 llama-server"
    )


class LlamaServerManager:
    """Owns the ``llama-server`` processes for one application data root.

    ``hold()`` brackets a unit of work: the servers stay up while the counter is
    positive and are stopped when the last holder leaves, which is exactly the
    "free the VRAM when the batch ends" rule without each caller tracking which
    other jobs are still queued.
    """

    def __init__(
        self,
        models_root: Path,
        ollama_root: Path | None = None,
        executable: Path | None = None,
    ):
        self.models_root = models_root
        self.ollama_root = ollama_root
        self.executable = executable
        self._lock = threading.RLock()
        self._slots: dict[str, tuple[LlamaSlot, subprocess.Popen]] = {}
        self._notes: dict[str, str] = {}
        self._holders = 0
        self._discovered: dict[str, OllamaModelFiles] | None = None

    # -- discovery ---------------------------------------------------------
    def _ollama_dirs(self) -> tuple[Path, Path]:
        root = self.ollama_root
        if root is not None:
            return root / "manifests-v2" / "ollama.com" / "library", root / "blobs"
        return Path(""), Path("")

    def models(self) -> dict[str, OllamaModelFiles]:
        with self._lock:
            if self._discovered is None:
                manifests, blobs = self._ollama_dirs()
                self._discovered = discover_ollama_models(manifests, blobs)
            return self._discovered

    def resolve(self, tag: str) -> OllamaModelFiles | None:
        """The blobs for ``tag``, tolerating a missing ``:latest`` suffix.

        ``ensure_ollama_model`` treats tags strictly, and settings written by an
        older version carry the suffix; accepting both spellings here keeps that
        mismatch from failing a migration.
        """
        models = self.models()
        if tag in models:
            return models[tag]
        stripped = tag[: -len(":latest")] if tag.endswith(":latest") else tag
        for candidate in (f"{tag}:latest", stripped, f"{stripped}:q8", f"{stripped}:latest"):
            if candidate in models:
                return models[candidate]
        return None

    def find_executable(self) -> Path:
        """The ``llama-server`` binary, wherever this install keeps it.

        ``ollama_root`` points at the model store (``<app>\\ollama-models``), but
        the binary itself sits in the unpacked Ollama runtime
        (``<app>\\_dl\\ollama\\lib\\ollama``), so both layouts are checked before
        giving up.
        """
        if self.executable is not None and self.executable.exists():
            return self.executable
        roots: list[Path] = []
        if self.ollama_root is not None:
            base = Path(self.ollama_root)
            roots.extend(
                [
                    base,
                    base / "_dl" / "ollama",
                    base.parent / "_dl" / "ollama",
                    base.parent,
                ]
            )
        for root in roots:
            for candidate in (
                root / "lib" / "ollama" / "llama-server.exe",
                root / "llama-server.exe",
            ):
                if candidate.exists():
                    return candidate
        located = shutil.which("llama-server")
        if located:
            return Path(located)
        raise LlamaServerError(
            "找不到 llama-server.exe：请在设置里指定路径，或把 Ollama 的 "
            "_dl\\ollama\\lib\\ollama 目录留在原处"
        )

    # -- process lifecycle -------------------------------------------------
    def _stage_directory(self, executable: Path) -> Path:
        """A directory holding the exe plus the GPU backend it needs.

        Ollama ships ``ggml-vulkan.dll`` in a ``vulkan`` subdirectory and loads
        it under its own name.  llama.cpp only looks next to the executable, so
        running the bundled binary in place silently yields a CPU-only server.
        """
        stage = self.models_root / "runtime"
        stage.mkdir(parents=True, exist_ok=True)
        for item in executable.parent.iterdir():
            if item.is_file() and item.suffix.lower() in {".dll", ".exe"}:
                self._link(item, stage / item.name)
        backend_dir = executable.parent / "vulkan"
        if backend_dir.is_dir():
            for item in backend_dir.iterdir():
                if item.is_file():
                    self._link(item, stage / item.name)
        return stage

    @staticmethod
    def _link(source: Path, target: Path) -> None:
        """Hard-link ``source`` to ``target``; overwrite a stale link."""
        if target.exists():
            try:
                if target.stat().st_size == source.stat().st_size:
                    return
            except OSError:
                pass
            try:
                target.unlink()
            except OSError:
                return
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)

    def _model_path(self, files: OllamaModelFiles) -> tuple[Path, Path | None]:
        """Hard-linked, human-readable copies next to the runtime directory.

        llama.cpp does not care about the name, but a directory full of
        ``sha256-…`` files is impossible to inspect when a start fails.
        """
        self.models_root.mkdir(parents=True, exist_ok=True)
        stem = files.tag.replace(":", "-").replace("/", "-")
        weights = self.models_root / f"{stem}.gguf"
        self._link(files.weights, weights)
        projector = None
        if files.projector is not None:
            projector = self.models_root / f"{stem}-mmproj.gguf"
            self._link(files.projector, projector)
        return weights, projector

    def _args(
        self,
        staged: Path,
        weights: Path,
        projector: Path | None,
        alias: str,
        port: int,
        context: int,
    ) -> list[str]:
        args = [
            str(staged / "llama-server.exe"),
            "-m",
            str(weights),
            "-c",
            str(context),
            "-ngl",
            "99",
            "-fa",
            "on",
            # Qwen3.5 exposes a reasoning channel that swallows the whole budget
            # and leaves ``content`` empty; this is the server-side twin of
            # Ollama's ``think: false`` and the migration fails without it.
            "-rea",
            "off",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--alias",
            alias,
            "-np",
            "1",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--no-webui",
        ]
        if projector is not None:
            args += ["-mm", str(projector)]
        return args

    def start(
        self,
        role: str,
        files: OllamaModelFiles,
        *,
        port: int,
        context: int,
        portable_dir: Path | None = None,
        timeout: float = START_TIMEOUT_S,
    ) -> LlamaSlot:
        """Start (or reuse) the server for ``role`` and wait until it answers."""
        alias = files.tag.replace(":", "-").replace("/", "-")
        with self._lock:
            existing = self._slots.get(role)
            if existing is not None:
                slot, process = existing
                if process.poll() is None and _healthy(slot.base_url):
                    return slot
                self._stop_locked(role)

            executable = self.find_executable()
            staged = self._stage_directory(executable)
            weights, projector = self._model_path(files)

            # Port policy, in order: reuse a server that is already serving this
            # exact alias; otherwise take the requested port when it is free;
            # otherwise move to the next free one and record why.  A port held
            # by an unrelated program is never stolen and never silently
            # ignored -- the move shows up in ``port_notes`` so the UI can say
            # something instead of leaving the user pointing at a dead URL.
            requested = port
            base_url = f"http://127.0.0.1:{port}"
            if _server_alias(base_url) == alias:
                slot = LlamaSlot(
                    role=role,
                    alias=alias,
                    base_url=base_url,
                    port=port,
                    context=context,
                    weights=weights,
                    projector=projector,
                )
                self._slots[role] = (slot, _StubProcess())
                return slot
            if not _port_is_free(port):
                port = _find_free_port(port + 1)
                self._notes[role] = (
                    f"端口 {requested} 已被其他程序占用，{role} 改用 {port}"
                )
                base_url = f"http://127.0.0.1:{port}"
            log_dir = (portable_dir or self.models_root) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"llama-{role}.log"
            handle = open(log_path, "wb")
            creation = 0
            if os.name == "nt":
                creation = 0x00000008 | 0x00000200  # DETACHED | NEW_PROCESS_GROUP
            process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                self._args(staged, weights, projector, alias, port, context),
                cwd=str(staged),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=creation,
            )
            slot = LlamaSlot(
                role=role,
                alias=alias,
                base_url=base_url,
                port=port,
                context=context,
                weights=weights,
                projector=projector,
            )
            self._slots[role] = (slot, process)
            try:
                self._wait_until_ready(slot, process, log_path, timeout)
            except LlamaServerError:
                self._stop_locked(role)
                raise
            return slot

    def _wait_until_ready(
        self,
        slot: LlamaSlot,
        process: subprocess.Popen,
        log_path: Path,
        timeout: float,
    ) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if process.poll() is not None:
                raise LlamaServerError(
                    f"llama-server（{slot.role}）启动后立即退出，日志：\n"
                    f"{_tail(log_path)}"
                )
            if _healthy(slot.base_url):
                if _server_alias(slot.base_url) == slot.alias:
                    return
            time.sleep(0.5)
        raise LlamaServerError(
            f"llama-server（{slot.role}）在 {int(timeout)} 秒内没有就绪，"
            f"日志：\n{_tail(log_path)}"
        )

    def _stop_locked(self, role: str) -> None:
        entry = self._slots.pop(role, None)
        if entry is None:
            return
        _, process = entry
        _terminate(process)

    def stop(self, role: str) -> None:
        with self._lock:
            self._stop_locked(role)

    def stop_all(self) -> None:
        with self._lock:
            for role in list(self._slots):
                self._stop_locked(role)

    def running(self) -> list[LlamaSlot]:
        with self._lock:
            return [
                slot
                for slot, process in self._slots.values()
                if isinstance(process, _StubProcess) or process.poll() is None
            ]

    def slot_for_tag(self, tag: str) -> LlamaSlot | None:
        """A live slot already serving ``tag``, whatever role started it.

        Vision OCR usually needs the same multimodal model as translation, and a
        7 GB duplicate on a second port would not fit next to the first one, so
        the caller asks for the tag instead of blindly starting one more server.
        """
        alias = tag.replace(":", "-").replace("/", "-")
        with self._lock:
            # ``_slots`` is keyed by role, so the tuple has to be unpacked rather
            # than taken from ``items()``.
            for slot, process in self._slots.values():
                live = isinstance(process, _StubProcess) or process.poll() is None
                if live and slot.alias == alias:
                    return slot
        return None

    def port_notes(self) -> dict[str, str]:
        """Port moves recorded during the last start, one line per role.

        A configured port that another program holds is not fatal -- the server
        moves to the next free one -- but the user asked for 8080 and is not
        getting it, so the reason has to be visible somewhere.
        """
        with self._lock:
            return dict(self._notes)

    # -- work bracket ------------------------------------------------------
    @contextlib.contextmanager
    def hold(self):
        """Keep the servers alive until the outermost holder exits."""
        with self._lock:
            self._holders += 1
        try:
            yield self
        finally:
            with self._lock:
                self._holders = max(0, self._holders - 1)
                if self._holders == 0:
                    self.stop_all()


class _StubProcess:
    """Stands in for a server this process started earlier and does not own."""

    def poll(self):
        return None

    def terminate(self) -> None:  # pragma: no cover - nothing to stop
        return None

    def kill(self) -> None:  # pragma: no cover - nothing to stop
        return None

    def wait(self, timeout=None):  # pragma: no cover - nothing to await
        return 0


def _tail(path: Path, lines: int = 12) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(无日志)"
    return "\n".join(text.strip().splitlines()[-lines:])


def _terminate(process, timeout: float = STOP_TIMEOUT_S) -> None:
    """Stop a server and everything it may have spawned."""
    if isinstance(process, _StubProcess):
        return
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=timeout)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        process.kill()
        process.wait(timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        pass


_MANAGERS: dict[str, LlamaServerManager] = {}
_MANAGERS_LOCK = threading.RLock()


def manager_for(
    models_root: Path, ollama_root: Path | None = None
) -> LlamaServerManager:
    """The process-wide manager for one data root.

    A module-level registry rather than an argument threaded through
    ``LocalizerPipeline``: the pipeline factory is called with just ``paths``
    from the job worker, and the shutdown path (which may run on another thread)
    has to reach the same processes.
    """
    key = str(models_root)
    with _MANAGERS_LOCK:
        manager = _MANAGERS.get(key)
        if manager is None:
            manager = LlamaServerManager(models_root, ollama_root)
            _MANAGERS[key] = manager
        return manager


def running_llama_slots() -> list[LlamaSlot]:
    """Every llama-server slot this process is currently keeping alive."""
    with _MANAGERS_LOCK:
        managers = list(_MANAGERS.values())
    slots: list[LlamaSlot] = []
    for manager in managers:
        slots.extend(manager.running())
    return slots


def stop_llama_servers() -> None:
    """Best-effort shutdown of every llama-server this process started."""
    with _MANAGERS_LOCK:
        managers = list(_MANAGERS.values())
    for manager in managers:
        try:
            manager.stop_all()
        except Exception:  # noqa: BLE001 - cleanup must never raise
            continue


def restart_llama_server(base_url: str) -> str | None:
    """Restart the slot answering at ``base_url``; its new URL, or ``None``.

    A long-uptime server can collapse into repeating one token for the whole
    budget, and no request setting brings it back: the sampler still holds the
    state that produced the collapse, so the answer is the same ``@@@@...``
    however many times it is asked.  Restarting the process is the only lever
    the client has.  ``None`` means "not ours, or the restart failed", and the
    caller must then report the failure rather than retry.
    """
    with _MANAGERS_LOCK:
        managers = list(_MANAGERS.values())
    for manager in managers:
        with manager._lock:
            entry = next(
                (
                    (role, slot, process)
                    for role, (slot, process) in manager._slots.items()
                    if slot.base_url == base_url
                ),
                None,
            )
            if entry is None:
                continue
            role, slot, process = entry
            if isinstance(process, _StubProcess):
                # Left over from an earlier session; killing and restarting it
                # would fight whoever owns it, so report instead.
                return None
            files = next(
                (
                    candidate
                    for tag, candidate in manager.models().items()
                    if tag.replace(":", "-").replace("/", "-") == slot.alias
                ),
                None,
            )
            if files is None:
                return None
            port, context = slot.port, slot.context
            manager._stop_locked(role)
        # Outside the lock: start() takes it again.
        try:
            return manager.start(role, files, port=port, context=context).base_url
        except (LlamaServerError, OSError):
            return None
    return None


# A batch runs one folder after another; loading the 7 GB model again between
# folders would add half a minute each time, so the servers outlive a single
# job while a batch is holding them.  The count, not a boolean, because a
# single job may also be released while a batch above it is still running.
_HOLD_LOCK = threading.RLock()
_HOLD_COUNT = 0


@contextlib.contextmanager
def hold_llama_servers():
    """Keep the servers alive until the outermost holder exits."""
    global _HOLD_COUNT
    with _HOLD_LOCK:
        _HOLD_COUNT += 1
    try:
        yield
    finally:
        with _HOLD_LOCK:
            _HOLD_COUNT = max(0, _HOLD_COUNT - 1)
            drained = _HOLD_COUNT == 0
        if drained:
            stop_llama_servers()


def release_llama_servers() -> None:
    """Stop the servers unless a wider scope is still holding them.

    Called when one job finishes.  Inside a batch this is a no-op because the
    batch holds the servers for its whole queue; standalone it frees the VRAM
    the moment the run ends, which is the defect the old Ollama path had.
    """
    with _HOLD_LOCK:
        idle = _HOLD_COUNT == 0
    if idle:
        stop_llama_servers()


# Requests carry the size of the reply, so a timeout keyed to that size keeps a
# short pass responsive without cutting a long structured reply in half.
def llama_timeout(token_budget: int, ceiling: int = 65_536) -> int:
    scaled = 1_200 * min(1.0, max(2_048, token_budget) / max(1, ceiling))
    return max(180, int(scaled))


def probe(base_url: str) -> bool:
    """True when a server answers its health endpoint at ``base_url``."""
    return _healthy(base_url, timeout=1.5)


def ollama_root_for(paths, settings) -> Path:
    """Where the Ollama blob store lives for this installation.

    ``start.bat`` puts it next to the data directory as ``ollama-models``, but
    someone may run the app from a checkout without that script, so the setting
    wins when set and the sibling directory is only a guess.
    """
    configured = (getattr(settings, "llama_ollama_root", "") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(paths.root).expanduser().parent / "ollama-models"


def _executable_for(settings) -> Path | None:
    configured = (getattr(settings, "llama_server_path", "") or "").strip()
    return Path(configured).expanduser() if configured else None


def manager_from_settings(paths, settings) -> LlamaServerManager:
    """The shared manager for this data root, configured from ``settings``."""
    root = ollama_root_for(paths, settings)
    manager = manager_for(Path(paths.llama_models), root)
    executable = _executable_for(settings)
    if executable is not None:
        manager.executable = executable
    return manager


def open_slots(manager: LlamaServerManager, settings, emit=None) -> dict[str, LlamaSlot]:
    """Start the editor, and in quality mode the review candidate too.

    Both are needed before the first translation request, and each one costs a
    model load, so the caller reports progress instead of leaving the UI silent
    for half a minute.
    """
    editor_tag = (getattr(settings, "llama_model", "") or "").strip()
    if not editor_tag:
        raise LlamaServerError("llama.cpp 主模型未配置（设置里的模型名是空的）")
    editor_files = manager.resolve(editor_tag)
    if editor_files is None:
        raise LlamaServerError(
            f"找不到模型 {editor_tag} 的 GGUF 文件："
            f"{manager.ollama_root} 下没有同名 manifest。"
            "请先在模型页下载该模型，或把设置里的模型名改成已下载的模型。"
        )
    slots: dict[str, LlamaSlot] = {}
    existing = manager.slot_for_tag(editor_tag)
    if existing is not None:
        # Vision OCR already loaded this very model for the OCR pass.  Starting
        # a second copy would put two 7 GB weights on one GPU for no gain, so
        # the running server is adopted instead.
        slots["editor"] = existing
    else:
        if emit:
            emit(f"启动 llama.cpp 主模型 {editor_tag}")
        slots["editor"] = manager.start(
            "editor",
            editor_files,
            port=int(getattr(settings, "llama_main_port", DEFAULT_MAIN_PORT) or DEFAULT_MAIN_PORT),
            context=int(
                getattr(settings, "llama_context_tokens", 0) or DEFAULT_CONTEXT_TOKENS
            ),
        )
    if getattr(settings, "quality_profile", "quality") == "quality":
        review_tag = (getattr(settings, "llama_review_model", "") or "").strip()
        if review_tag and review_tag != editor_tag:
            review_files = manager.resolve(review_tag)
            if review_files is None:
                raise LlamaServerError(
                    f"找不到审校模型 {review_tag} 的 GGUF 文件："
                    f"{manager.ollama_root} 下没有同名 manifest。"
                )
            if emit:
                emit(f"启动 llama.cpp 审校模型 {review_tag}")
            slots["review"] = manager.start(
                "review",
                review_files,
                port=int(
                    getattr(settings, "llama_review_port", DEFAULT_REVIEW_PORT)
                    or DEFAULT_REVIEW_PORT
                ),
                # The 1.8B candidate never sees a whole book; a smaller window
                # keeps its KV cache out of the 9B's way.
                context=min(8_192, int(getattr(settings, "llama_context_tokens", 0) or 8_192)),
            )
        else:
            # Same tag for both roles: one process serves both passes, which is
            # what the Ollama path does with a single resident model.
            slots["review"] = slots["editor"]
    return slots


def open_vision_slot(manager: LlamaServerManager, settings, emit=None) -> LlamaSlot:
    """The server that reads page images for OCR.

    The model is the OCR tag from settings, which is the same multimodal 9B the
    translator uses -- that is the whole point of routing OCR through llama.cpp
    too, and it means the OCR pass and the translation pass can share one loaded
    weight set.  The context is the main window for the same reason: whichever of
    the two passes starts the process, the other adopts it by tag and must not
    inherit a window that is too small for a book-sized prompt.
    """
    tag = (getattr(settings, "ollama_ocr_model", "") or "").strip()
    if not tag:
        tag = (getattr(settings, "llama_model", "") or "").strip()
    if not tag:
        raise LlamaServerError("llama.cpp 视觉 OCR 需要一个模型名，请先在设置里填写 OCR 模型")
    files = manager.resolve(tag)
    if files is None:
        raise LlamaServerError(
            f"找不到视觉 OCR 模型 {tag} 的 GGUF 文件："
            f"{manager.ollama_root} 下没有同名 manifest。"
            "请先在模型页下载该模型，或把设置里的 OCR 模型名改成已下载的模型。"
        )
    if files.projector is None:
        raise LlamaServerError(
            f"模型 {tag} 没有 mmproj 多模态投影器，无法用 llama.cpp 做视觉 OCR。"
            "请换一个带视觉能力的模型，或把 OCR 后端改回 hybrid / builtin。"
        )
    existing = manager.slot_for_tag(tag)
    if existing is not None:
        # Already loaded for this very tag; a second copy of a 7 GB weight set
        # would not fit beside the first one.
        return existing
    if emit:
        emit(f"启动 llama.cpp 视觉 OCR 模型 {tag}")
    return manager.start(
        "vision",
        files,
        port=int(
            getattr(settings, "llama_vision_port", 0) or DEFAULT_VISION_PORT
        ),
        context=int(
            getattr(settings, "llama_context_tokens", 0) or DEFAULT_CONTEXT_TOKENS
        ),
    )


__all__ = [
    "DEFAULT_MAIN_PORT",
    "DEFAULT_REVIEW_PORT",
    "DEFAULT_VISION_PORT",
    "DEFAULT_CONTEXT_TOKENS",
    "LlamaServerError",
    "LlamaServerManager",
    "LlamaSlot",
    "OllamaModelFiles",
    "discover_ollama_models",
    "hold_llama_servers",
    "llama_timeout",
    "manager_for",
    "manager_from_settings",
    "ollama_root_for",
    "open_slots",
    "open_vision_slot",
    "probe",
    "release_llama_servers",
    "restart_llama_server",
    "running_llama_slots",
    "stop_llama_servers",
]
