import numpy as np
import pytest

from manga_localizer.config import AppPaths
from manga_localizer.inpainting import LaMaInpainter
from manga_localizer.model_manager import ModelManager


def test_lama_wrapper_preserves_every_unmasked_pixel(tmp_path):
    torch = pytest.importorskip("torch")

    class WhiteFill(torch.nn.Module):
        def forward(self, image, mask):
            return torch.ones_like(image)

    model_path = tmp_path / "tiny-lama.pt"
    traced = torch.jit.trace(
        WhiteFill(),
        (torch.zeros((1, 3, 16, 16)), torch.zeros((1, 1, 16, 16))),
    )
    traced.save(str(model_path))
    source = np.arange(13 * 19 * 3, dtype=np.uint8).reshape((13, 19, 3))
    mask = np.zeros((13, 19), dtype=np.uint8)
    mask[3:9, 5:12] = 255
    output = LaMaInpainter(model_path, "cpu")(source, mask)
    assert output.shape == source.shape
    assert np.array_equal(output[mask == 0], source[mask == 0])
    assert np.all(output[mask > 0] == 255)


def test_a_state_dict_checkpoint_refuses_to_run_without_its_graph(tmp_path):
    """The manga checkpoint has no graph of its own, so it must not fall back."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    weights_path = tmp_path / "lama-manga.safetensors"
    weights_path.write_bytes(b"")
    checker = LaMaInpainter(weights_path, "cpu")
    with pytest.raises(FileNotFoundError, match="LaMa graph"):
        checker._load()


def _paths(tmp_path):
    return AppPaths(
        tmp_path,
        tmp_path / "models",
        tmp_path / "cache",
        tmp_path / "jobs",
        tmp_path / "settings.json",
    )


def test_the_erase_model_is_resolved_before_a_page_is_touched(tmp_path):
    manager = ModelManager(_paths(tmp_path))
    graph = manager.lama_weights_path()
    with pytest.raises(FileNotFoundError, match="LaMa"):
        manager.erase_model_paths("lama")
    graph.parent.mkdir(parents=True, exist_ok=True)
    graph.write_bytes(b"")
    assert manager.erase_model_paths("lama") == (graph, None)
    with pytest.raises(FileNotFoundError, match="lama-manga"):
        manager.erase_model_paths("lama-manga")
    manager.lama_manga_weights_path().parent.mkdir(parents=True, exist_ok=True)
    manager.lama_manga_weights_path().write_bytes(b"")
    assert manager.erase_model_paths("lama-manga") == (
        manager.lama_manga_weights_path(),
        graph,
    )
    with pytest.raises(ValueError, match="擦除模型"):
        manager.erase_model_paths("nope")
