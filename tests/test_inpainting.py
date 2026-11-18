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


def test_the_erase_model_falls_back_when_the_manga_weights_are_absent(tmp_path):
    """A hand-placed checkpoint must not fail a book; the general one takes over."""
    manager = ModelManager(_paths(tmp_path))
    graph = manager.lama_weights_path()
    with pytest.raises(FileNotFoundError, match="LaMa"):
        manager.erase_model_paths("lama")
    graph.parent.mkdir(parents=True, exist_ok=True)
    graph.write_bytes(b"")
    assert manager.erase_model_paths("lama") == (graph, None)
    effective, reason = manager.resolve_erase_model("lama-manga")
    assert effective == "lama"
    assert "lama-manga.safetensors" in reason
    assert manager.erase_model_paths("lama-manga") == (graph, None)
    manager.lama_manga_weights_path().parent.mkdir(parents=True, exist_ok=True)
    manager.lama_manga_weights_path().write_bytes(b"")
    assert manager.resolve_erase_model("lama-manga") == ("lama-manga", "")
    assert manager.erase_model_paths("lama-manga") == (
        manager.lama_manga_weights_path(),
        graph,
    )
    with pytest.raises(ValueError, match="擦除模型"):
        manager.erase_model_paths("nope")


def test_the_aot_checkpoint_is_reachable_without_the_lama_graph(tmp_path):
    """AOT rebuilds its own network, so it never needs the scripted LaMa file."""
    manager = ModelManager(_paths(tmp_path))
    effective, reason = manager.resolve_erase_model("aot-manga")
    assert effective == "lama"
    assert "model.safetensors" in reason
    manager.aot_weights_path().parent.mkdir(parents=True, exist_ok=True)
    manager.aot_weights_path().write_bytes(b"")
    assert manager.resolve_erase_model("aot-manga") == ("aot-manga", "")
    assert manager.erase_model_paths("aot-manga") == (
        manager.aot_weights_path(),
        None,
    )


def test_the_aot_wrapper_passes_a_whole_page_through_untouched(tmp_path):
    """No mask means no inference: the page must come back byte-identical."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from manga_localizer.inpainting import AOTMangaInpainter

    source = np.arange(24 * 17 * 3, dtype=np.uint8).reshape((24, 17, 3))
    mask = np.zeros((24, 17), dtype=np.uint8)
    output = AOTMangaInpainter(tmp_path / "missing.safetensors", "cpu")(source, mask)
    assert np.array_equal(output, source)


def test_the_aot_wrapper_reports_a_missing_checkpoint(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from manga_localizer.inpainting import AOTMangaInpainter

    source = np.zeros((24, 17, 3), dtype=np.uint8)
    mask = np.zeros((24, 17), dtype=np.uint8)
    mask[2:6, 3:9] = 255
    wrapper = AOTMangaInpainter(tmp_path / "missing.safetensors", "cpu")
    with pytest.raises(FileNotFoundError, match="AOT"):
        wrapper(source, mask)


def _white_hole_graph(path):
    """A stand-in for the exported generator: everything inside the mask is white."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "h", "w"])
    mask = helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, "h", "w"])
    painted = helper.make_tensor_value_info("painted", TensorProto.FLOAT, [1, 3, "h", "w"])
    # ones * (1 - mask) is 0 inside the hole and 1 outside; the wrapper only
    # reads the hole back, so the value there is what the test asserts on.
    nodes = [
        helper.make_node("Sub", ["ones", "mask"], ["outside"]),
        helper.make_node("Mul", ["image", "outside"], ["kept"]),
        helper.make_node("Add", ["kept", "ones"], ["painted"]),
    ]
    graph = helper.make_graph(
        nodes,
        "white-hole",
        [image, mask],
        [painted],
        initializer=[helper.make_tensor("ones", TensorProto.FLOAT, [1, 3, 1, 1], [1.0] * 3)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))


def test_the_aot_wrapper_runs_the_exported_graph_and_keeps_the_boundary(tmp_path):
    """The ONNX path is preferred when its file exists, and honours the contract."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    pytest.importorskip("onnxruntime")
    from manga_localizer.inpainting import AOTMangaInpainter

    onnx_path = tmp_path / "aot-manga.onnx"
    _white_hole_graph(onnx_path)
    source = np.full((24, 32, 3), 40, dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:10, 6:14] = 255

    wrapper = AOTMangaInpainter(tmp_path / "unused.safetensors", "cpu", onnx_path=onnx_path)
    output = wrapper(source, mask)
    assert wrapper._session is not None, "the exported graph should have been opened"
    assert wrapper._provider in {"DmlExecutionProvider", "CPUExecutionProvider"}
    assert np.array_equal(output[mask == 0], source[mask == 0])
    assert output[mask > 0].min() > source[mask > 0].max()


def test_a_missing_onnx_export_leaves_the_torch_path_in_charge(tmp_path):
    """No .onnx file is not an error: the state_dict still drives the network."""
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from manga_localizer.inpainting import AOTMangaInpainter

    wrapper = AOTMangaInpainter(tmp_path / "model.safetensors", "cpu", onnx_path=None)
    assert wrapper._load_onnx() is False
    assert wrapper._session is None
