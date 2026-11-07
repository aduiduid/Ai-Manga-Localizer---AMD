"""CPU discovery and thread budgeting.

The pipeline runs three CPU-bound recognizers back to back (Paddle's detector,
Paddle's recognition pass, and Manga OCR's torch model).  Each of them sizes its
own thread pool from the machine's logical CPU count, so on a 6-core/12-thread
desktop they all try to claim 12 threads and end up fighting for the same cores.
Letting the user pick a number lets them leave headroom for the browser, or pin
the whole job to a couple of cores on a laptop.

``cpu_threads`` is stored as an integer where ``0`` means "let every library use
its own default", which is the historical behaviour and the safe fallback when
discovery does not work.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

# The thread budget is applied through each library's runtime setter rather than
# through OMP_NUM_THREADS / MKL_NUM_THREADS.  Paddle's Windows wheel reacts to a
# non-default OMP_NUM_THREADS by printing "It will fail if this PaddlePaddle
# binary is compiled with OpenBlas", which reads like a fatal error in the UI
# log for what is only a tuning hint.  torch.set_num_threads() and
# paddle's core.set_num_threads() are both documented runtime APIs and are what
# actually govern the intra-op pools here.


@dataclass(frozen=True)
class CpuInfo:
    """What the settings panel needs to describe this machine's CPU."""

    name: str
    physical_cores: int
    logical_cores: int

    @property
    def choices(self) -> list[int]:
        """Every thread count from ``N`` down to 1, largest first.

        Every value in the range is offered rather than only multiples or only
        the physical core count: a user offloading a long batch may want 3 of 12
        threads on one machine and 10 of 16 on another, and a coarse ladder
        would silently deny both.  ``N`` is the logical count because that is the
        true ceiling of the thread pools this setting drives.
        """
        ceiling = max(1, self.logical_cores)
        return list(range(ceiling, 0, -1))


def _windows_cpu_name() -> str:
    """Read the marketing CPU name from the registry.

    ``platform.processor()`` only yields the family/model string ("AMD64 Family
    25 Model 80"), which is useless in a UI.  The string the vendor advertises
    lives under the first logical processor's registry key.
    """
    try:
        import winreg
    except ImportError:
        return ""
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        return ""


def _windows_physical_cores() -> int:
    """Count physical cores via GetLogicalProcessorInformationEx.

    The registry exposes one subkey per *logical* processor, so it cannot answer
    this.  psutil is not a declared dependency (it only arrives transitively with
    some ML packages), so the Win32 call is the primary probe and psutil is the
    fallback rather than the other way round.
    """
    try:
        import ctypes
        from ctypes import wintypes

        relation_processor_core = 0
        buffer_length = wintypes.DWORD(0)
        kernel32 = ctypes.windll.kernel32
        # First call sizes the buffer; it always "fails" with a length hint.
        kernel32.GetLogicalProcessorInformationEx(
            relation_processor_core, None, ctypes.byref(buffer_length)
        )
        if buffer_length.value <= 0:
            return 0
        buffer = ctypes.create_string_buffer(buffer_length.value)
        if not kernel32.GetLogicalProcessorInformationEx(
            relation_processor_core, buffer, ctypes.byref(buffer_length)
        ):
            return 0
        # Each record starts with Relationship (DWORD) + Size (DWORD); walk them.
        cores = 0
        offset = 0
        raw = bytes(buffer)
        while offset + 8 <= len(raw):
            record_size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
            if record_size < 8:
                break
            cores += 1
            offset += record_size
        return cores
    except Exception:
        return 0


def _physical_cores() -> int:
    cores = _windows_physical_cores() if platform.system() == "Windows" else 0
    if cores > 0:
        return cores
    try:
        import psutil

        return int(psutil.cpu_count(logical=False) or 0)
    except Exception:
        return 0


def detect_cpu() -> CpuInfo:
    """Describe the local CPU, degrading to sane numbers when probes fail."""
    logical = int(os.cpu_count() or 1)
    physical = _physical_cores()
    name = _windows_cpu_name() if platform.system() == "Windows" else ""
    if not name:
        name = platform.processor() or platform.machine() or "CPU"
    # psutil returns None on some virtualized hosts; a machine always has at
    # least one core, and Windows reports logical >= physical, so never let the
    # headline number exceed the real thread count.
    if physical <= 0 or physical > logical:
        physical = logical
    return CpuInfo(name=name, physical_cores=physical, logical_cores=logical)


def normalize_cpu_threads(value: int | None, cpu: CpuInfo | None = None) -> int:
    """Clamp a requested thread count to something this machine can run."""
    info = cpu or detect_cpu()
    try:
        requested = int(value or 0)
    except (TypeError, ValueError):
        return 0
    if requested <= 0:
        return 0
    return max(1, min(requested, info.logical_cores))


def apply_cpu_threads(value: int | None, cpu: CpuInfo | None = None) -> int:
    """Apply a thread budget to torch and Paddle.

    Returns the count actually in effect (``0`` means "library defaults").  Each
    library is imported defensively: the CLI imports this module before any ML
    runtime, the web server may have already loaded one, and on Windows a torch
    import after paddle's DLLs are mapped raises OSError.  A failure to import
    is not an error here -- it just means that backend is not in play.
    """
    resolved = normalize_cpu_threads(value, cpu)
    try:
        import torch

        if resolved:
            torch.set_num_threads(resolved)
        elif hasattr(torch, "set_num_threads"):
            # 0 means "leave it alone"; restore the library default only when a
            # previous job narrowed it and the user has now cleared the setting.
            torch.set_num_threads(max(1, os.cpu_count() or 1))
    except Exception:
        pass
    try:
        from paddle.base import core

        if resolved:
            core.set_num_threads(resolved)
    except Exception:
        pass
    return resolved
