from __future__ import annotations

from pathlib import Path

import numpy as np

from .model_manager import ModelDependencyError


class LaMaInpainter:
    """Lazy, native-resolution wrapper around the pinned Big-LaMa TorchScript model.

    Only masked pixels are accepted from the network.  Unmasked source pixels
    are copied back byte-for-byte after inference, which makes the edit boundary
    an enforceable contract instead of a best-effort promise.
    """

    def __init__(
        self, model_path: Path, device: str = "auto", graph_path: Path | None = None
    ):
        self.model_path = Path(model_path)
        # Only a raw checkpoint needs a graph to run in; a scripted `.pt` brings
        # its own, so this stays None for the pinned Big-LaMa file.
        self.graph_path = Path(graph_path) if graph_path is not None else None
        self.requested_device = device
        self._torch = None
        self._model = None
        self._device = None

    def _load_weights_into_graph(self, torch):
        """Run a raw state_dict checkpoint on the scripted graph that shipped.

        The manga checkpoint holds the same 989 tensor names the pinned
        TorchScript wrapper exposes, so the graph is reused and only the
        weights are swapped -- no second copy of the network to maintain.
        """
        if self.graph_path is None or not self.graph_path.exists():
            raise FileNotFoundError(
                f"LaMa graph is missing: {self.graph_path}. Run `manga-localizer models download lama`."
            )
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ModelDependencyError(
                "lama-manga requires the safetensors package. Install "
                "manga-localizer-studio[ml]."
            ) from exc
        graph = torch.jit.load(str(self.graph_path), map_location=self._device)
        graph.model.generator.load_state_dict(load_file(str(self.model_path)))
        return graph

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LaMa weights are missing: {self.model_path}. Run `manga-localizer models download lama`."
            )
        try:
            import torch
        except ImportError as exc:
            raise ModelDependencyError(
                "LaMa requires PyTorch. Install manga-localizer-studio[ml]."
            ) from exc
        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.requested_device.startswith("gpu") or self.requested_device == "cuda":
            # The UI always offers "GPU 0".  On a machine without CUDA (this
            # box runs the CPU torch wheel on an AMD card) asking torch for
            # cuda raises "Torch not compiled with CUDA enabled", so honour
            # the request only when it can actually be met.
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"
        self._torch = torch
        self._device = torch.device(device)
        # `.pt` here is a TorchScript archive that carries its own graph;
        # `.safetensors` is a bare state_dict, so it borrows one.
        if self.model_path.suffix.lower() == ".safetensors":
            model = self._load_weights_into_graph(torch)
        else:
            model = torch.jit.load(str(self.model_path), map_location=self._device)
        model.eval()
        self._model = model

    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("LaMa input must be an HxWx3 RGB image")
        if mask.shape != rgb.shape[:2]:
            raise ValueError("LaMa mask must match the image dimensions")
        binary = mask > 0
        if not binary.any():
            return rgb.copy()
        self._load()
        torch = self._torch
        height, width = rgb.shape[:2]
        pad_height = (-height) % 8
        pad_width = (-width) % 8
        image_tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(
            2, 0, 1
        )[None]
        mask_tensor = torch.from_numpy(binary.astype(np.float32))[None, None]
        if pad_height or pad_width:
            image_tensor = torch.nn.functional.pad(
                image_tensor, (0, pad_width, 0, pad_height), mode="reflect"
            )
            mask_tensor = torch.nn.functional.pad(
                mask_tensor, (0, pad_width, 0, pad_height), mode="constant", value=0
            )
        image_tensor = image_tensor.to(self._device)
        mask_tensor = mask_tensor.to(self._device)
        with torch.inference_mode():
            predicted = self._model(image_tensor, mask_tensor)
        predicted = predicted[0].permute(1, 2, 0)[:height, :width]
        result = (predicted.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        result[~binary] = rgb[~binary]
        return result
