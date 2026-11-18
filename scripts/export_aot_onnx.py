"""Export the AOT manga inpainter to ONNX for the DirectML provider.

torch is a CPU-only build here and torch-directml pins torch below 2.5, so
ONNX Runtime is the only way this network reaches the GPU.  AOT exports
cleanly; Big-LaMa does not (its FFC blocks need ``aten::fft_rfftn``, which no
opset supports), which is why this script only covers the AOT checkpoint.

Run once after placing ``model.safetensors``:

    .venv\\Scripts\\python.exe scripts\\export_aot_onnx.py

It writes ``aot-manga.onnx`` next to the checkpoint.  The graph is exported
with dynamic height/width, because every page has its own size, and it folds
the "blank the masked hole" step into the graph so the wrapper only has to
normalise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manga_localizer.config import AppPaths  # noqa: E402
from manga_localizer.inpainting import _build_aot_module  # noqa: E402
from manga_localizer.model_manager import ModelManager  # noqa: E402

OPSET = 17


class Exportable(torch.nn.Module):
    """The generator with the masking step baked in."""

    def __init__(self, generator: torch.nn.Module) -> None:
        super().__init__()
        self.generator = generator

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.generator(image * (1 - mask), mask)


def main() -> int:
    paths = AppPaths.from_env().ensure()
    manager = ModelManager(paths)
    weights = manager.aot_weights_path()
    target = manager.aot_onnx_path()
    if not weights.exists():
        print(f"AOT checkpoint not found: {weights}")
        return 1

    from safetensors.torch import load_file

    generator = _build_aot_module(torch)()
    generator.load_state_dict(load_file(str(weights)), strict=True)
    generator.eval()
    model = Exportable(generator).eval()

    image = torch.zeros(1, 3, 800, 1120)
    mask = torch.zeros(1, 1, 800, 1120)
    mask[:, :, 200:400, 200:400] = 1.0

    torch.onnx.export(
        model,
        (image, mask),
        str(target),
        input_names=["image", "mask"],
        output_names=["painted"],
        opset_version=OPSET,
        dynamo=False,
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "mask": {0: "batch", 2: "height", 3: "width"},
            "painted": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"wrote {target} ({target.stat().st_size // 1024} KiB)")

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return 0

    session = ort.InferenceSession(
        str(target), providers=ort.get_available_providers()[:1]
    )
    feed = {"image": image.numpy(), "mask": mask.numpy()}
    with torch.inference_mode():
        expected = model(image, mask).numpy()
    got = session.run(None, feed)[0]
    delta = float(np.abs(got - expected).max())
    print(f"verified against torch on {session.get_providers()[0]}: max|diff| {delta:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
