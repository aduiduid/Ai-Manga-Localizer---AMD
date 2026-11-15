from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .model_manager import ModelDependencyError

# A whole page goes through the network in one pass now, so its size decides
# the activation footprint.  2048 keeps a doubled-up manga spread affordable
# and is above every page in the sample books, so the guard only engages on
# genuinely oversized scans instead of quietly softening ordinary pages.
ERASE_MAX_SIDE = 2048


def _fit_for_inference(
    rgb: np.ndarray, binary: np.ndarray, max_side: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Downscale a page and its mask until the longer side fits ``max_side``."""
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width)) if max_side else 1.0
    if scale >= 1.0:
        return rgb, binary, 1.0
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = np.asarray(Image.fromarray(rgb).resize(size, Image.LANCZOS))
    mask = np.asarray(
        Image.fromarray(binary.astype(np.uint8) * 255).resize(size, Image.LANCZOS)
    )
    return image, mask >= 127, scale


def _restore_size(rgb: np.ndarray, size: tuple[int, int], scale: float) -> np.ndarray:
    """Grow an inference result back to the page it came from."""
    if scale >= 1.0:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize(size, Image.LANCZOS))


class LaMaInpainter:
    """Lazy, native-resolution wrapper around the pinned Big-LaMa TorchScript model.

    Only masked pixels are accepted from the network.  Unmasked source pixels
    are copied back byte-for-byte after inference, which makes the edit boundary
    an enforceable contract instead of a best-effort promise.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        graph_path: Path | None = None,
        max_side: int = ERASE_MAX_SIDE,
    ):
        self.model_path = Path(model_path)
        # Only a raw checkpoint needs a graph to run in; a scripted `.pt` brings
        # its own, so this stays None for the pinned Big-LaMa file.
        self.graph_path = Path(graph_path) if graph_path is not None else None
        self.requested_device = device
        self.max_side = max_side
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
        fitted, fitted_mask, scale = _fit_for_inference(rgb, binary, self.max_side)
        pad_height = (-fitted.shape[0]) % 8
        pad_width = (-fitted.shape[1]) % 8
        image_tensor = torch.from_numpy(fitted.astype(np.float32) / 255.0).permute(
            2, 0, 1
        )[None]
        mask_tensor = torch.from_numpy(fitted_mask.astype(np.float32))[None, None]
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
        predicted = predicted[0].permute(1, 2, 0)[: fitted.shape[0], : fitted.shape[1]]
        painted = (
            (predicted.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        )
        painted = _restore_size(painted, (width, height), scale)
        result = rgb.copy()
        result[binary] = painted[binary]
        return result


# ---------------------------------------------------------------------------
# AOT inpainting (manga-image-translator)
#
# There is no scripted graph for this checkpoint, only a bare state_dict, so
# the network is rebuilt tensor-for-tensor from the upstream definition.  The
# magic constants below are not ours: they are what the released weights were
# trained against, and dropping any one of them turns the output into a flat
# grey field.
# ---------------------------------------------------------------------------

AOT_RELU_GAIN = 1.7139588594436646
AOT_GATE_GAIN = 1.8


def _aot_relu(x):
    import torch.nn.functional as functional

    return functional.relu(x) * AOT_RELU_GAIN


def _aot_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    return 5 * (2 * (feat - mean) / std - 1)


def _build_aot_module(torch):
    """Declare the AOT generator against the installed torch version."""
    nn = torch.nn

    class Lambda(nn.Module):
        def __init__(self, function):
            super().__init__()
            self.function = function

        def forward(self, value):
            return self.function(value)

    class ScaledWSConv2d(nn.Conv2d):
        """Conv2d with weight standardisation and a learned per-channel gain."""

        def __init__(self, *args, transpose: bool = False, **kwargs):
            kwargs.pop("gain", None)
            super().__init__(*args, **kwargs)
            channels = self.in_channels if transpose else self.out_channels
            self.gain = nn.Parameter(torch.ones(channels, 1, 1, 1))
            self.eps = 1e-4

        def get_weight(self):
            import numpy as np

            fan_in = int(np.prod(self.weight.shape[1:]))
            var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdim=True)
            floor = torch.tensor(self.eps, dtype=var.dtype, device=var.device)
            scale = torch.rsqrt(torch.max(var * fan_in, floor)) * self.gain
            return self.weight * scale - mean * scale

        def forward(self, value):
            import torch.nn.functional as functional

            return functional.conv2d(
                value,
                self.get_weight(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    class ScaledWSTransposeConv2d(nn.ConvTranspose2d):
        def __init__(self, *args, **kwargs):
            kwargs.pop("gain", None)
            kwargs.pop("transpose", None)
            super().__init__(*args, **kwargs)
            self.gain = nn.Parameter(torch.ones(self.in_channels, 1, 1, 1))
            self.eps = 1e-4

        def get_weight(self):
            import numpy as np

            fan_in = int(np.prod(self.weight.shape[1:]))
            var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdim=True)
            floor = torch.tensor(self.eps, dtype=var.dtype, device=var.device)
            scale = torch.rsqrt(torch.max(var * fan_in, floor)) * self.gain
            return self.weight * scale - mean * scale

        def forward(self, value, output_size=None):
            import torch.nn.functional as functional

            return functional.conv_transpose2d(
                value,
                self.get_weight(),
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )

    class GatedWSConvPadded(nn.Module):
        def __init__(self, in_ch, out_ch, ks, stride=1, dilation=1):
            super().__init__()
            self.padding = nn.ReflectionPad2d(((ks - 1) * dilation) // 2)
            self.conv = ScaledWSConv2d(
                in_ch, out_ch, kernel_size=ks, stride=stride, dilation=dilation
            )
            self.conv_gate = ScaledWSConv2d(
                in_ch, out_ch, kernel_size=ks, stride=stride, dilation=dilation
            )

        def forward(self, value):
            value = self.padding(value)
            return self.conv(value) * torch.sigmoid(self.conv_gate(value)) * AOT_GATE_GAIN

    class GatedWSTransposeConvPadded(nn.Module):
        def __init__(self, in_ch, out_ch, ks, stride=1):
            super().__init__()
            padding = (ks - 1) // 2
            self.conv = ScaledWSTransposeConv2d(
                in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, transpose=True
            )
            self.conv_gate = ScaledWSTransposeConv2d(
                in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, transpose=True
            )

        def forward(self, value):
            return self.conv(value) * torch.sigmoid(self.conv_gate(value)) * AOT_GATE_GAIN

    class AOTBlock(nn.Module):
        def __init__(self, dim, rates=(2, 4, 8, 16)):
            super().__init__()
            for index, rate in enumerate(rates):
                setattr(
                    self,
                    f"block{index:02d}",
                    nn.Sequential(
                        nn.ReflectionPad2d(rate),
                        nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                        nn.ReLU(True),
                    ),
                )
            self.fuse = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0)
            )
            self.gate = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0)
            )
            self.rates = rates

        def forward(self, value):
            branches = [
                getattr(self, f"block{index:02d}")(value)
                for index in range(len(self.rates))
            ]
            fused = self.fuse(torch.cat(branches, 1))
            mask = torch.sigmoid(_aot_layer_norm(self.gate(value)))
            return value * (1 - mask) + fused * mask

    class AOTGenerator(nn.Module):
        def __init__(self, in_ch=4, out_ch=3, ch=32, blocks=10):
            super().__init__()
            self.head = nn.Sequential(
                GatedWSConvPadded(in_ch, ch, 3, stride=1),
                Lambda(_aot_relu),
                GatedWSConvPadded(ch, ch * 2, 4, stride=2),
                Lambda(_aot_relu),
                GatedWSConvPadded(ch * 2, ch * 4, 4, stride=2),
            )
            self.body_conv = nn.Sequential(*[AOTBlock(ch * 4) for _ in range(blocks)])
            self.tail = nn.Sequential(
                GatedWSConvPadded(ch * 4, ch * 4, 3, 1),
                Lambda(_aot_relu),
                GatedWSConvPadded(ch * 4, ch * 4, 3, 1),
                Lambda(_aot_relu),
                GatedWSTransposeConvPadded(ch * 4, ch * 2, 4, 2),
                Lambda(_aot_relu),
                GatedWSTransposeConvPadded(ch * 2, ch, 4, 2),
                Lambda(_aot_relu),
                GatedWSConvPadded(ch, out_ch, 3, stride=1),
            )

        def forward(self, img, mask):
            value = torch.cat([mask, img], dim=1)
            value = self.head(value)
            value = self.body_conv(value)
            value = self.tail(value)
            return value if self.training else torch.clip(value, -1, 1)

    return AOTGenerator


class AOTMangaInpainter:
    """Lazy wrapper around the manga-image-translator AOT generator.

    Whole-page inference only.  The checkpoint is 5.7M parameters against
    LaMa's 51M, which makes one pass over the entire page cheaper than the
    dozens of per-balloon crops the LaMa path used to run.  It also rules out
    per-crop inference: the network fills small isolated crops with flat white
    because it has no page around them to continue.

    Unmasked pixels are copied back byte-for-byte, exactly like LaMaInpainter.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        max_side: int = ERASE_MAX_SIDE,
    ):
        self.model_path = Path(model_path)
        self.requested_device = device
        self.max_side = max_side
        self._torch = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"未找到 AOT 权重：{self.model_path}。把 AOT 的 model.safetensors 放进该目录即可启用。"
            )
        try:
            import torch
        except ImportError as exc:
            raise ModelDependencyError(
                "AOT requires PyTorch. Install manga-localizer-studio[ml]."
            ) from exc
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ModelDependencyError(
                "AOT requires the safetensors package. Install "
                "manga-localizer-studio[ml]."
            ) from exc
        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.requested_device.startswith("gpu") or self.requested_device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"
        self._torch = torch
        self._device = torch.device(device)
        generator = _build_aot_module(torch)
        model = generator()
        model.load_state_dict(load_file(str(self.model_path)), strict=True)
        model.eval()
        model.to(self._device)
        self._model = model

    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("AOT input must be an HxWx3 RGB image")
        if mask.shape != rgb.shape[:2]:
            raise ValueError("AOT mask must match the image dimensions")
        binary = mask > 0
        if not binary.any():
            return rgb.copy()
        self._load()
        torch = self._torch
        height, width = rgb.shape[:2]
        fitted, fitted_mask, scale = _fit_for_inference(rgb, binary, self.max_side)
        pad_height = (-fitted.shape[0]) % 8
        pad_width = (-fitted.shape[1]) % 8
        image_tensor = torch.from_numpy(
            fitted.astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1)[None]
        mask_tensor = torch.from_numpy(fitted_mask.astype(np.float32))[None, None]
        if pad_height or pad_width:
            image_tensor = torch.nn.functional.pad(
                image_tensor, (0, pad_width, 0, pad_height), mode="replicate"
            )
            mask_tensor = torch.nn.functional.pad(
                mask_tensor, (0, pad_width, 0, pad_height), mode="replicate"
            )
        image_tensor = image_tensor.to(self._device)
        mask_tensor = mask_tensor.to(self._device)
        with torch.inference_mode():
            # The mask hole has to be blanked out of the input.  Leaving the
            # original glyphs in place makes the generator treat them as
            # content and paint them back.
            predicted = self._model(image_tensor * (1 - mask_tensor), mask_tensor)
        predicted = predicted[0].permute(1, 2, 0)[: fitted.shape[0], : fitted.shape[1]]
        painted = np.clip((predicted.cpu().numpy() + 1.0) * 127.5, 0, 255)
        painted = painted.round().astype(np.uint8)
        painted = _restore_size(painted, (width, height), scale)
        result = rgb.copy()
        result[binary] = painted[binary]
        return result
