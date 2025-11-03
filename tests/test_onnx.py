import os
import tempfile

import pytest

from src.model import ModelLoader
from src.onnx_exporter import ONNXExporter


class TestONNXExporter:
    @pytest.fixture
    def model_loader(self):
        return ModelLoader(device="cpu")

    @pytest.fixture
    def temp_onnx_path(self):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            yield tmp.name
        os.unlink(tmp.name)

    def test_export_model(self, model_loader, temp_onnx_path):
        exporter = ONNXExporter(model_loader.model, "cpu", temp_onnx_path)
        exporter.export_model()
        assert os.path.exists(temp_onnx_path)
        assert os.path.getsize(temp_onnx_path) > 0

    def test_export_creates_models_dir(self, model_loader):
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "models", "test.onnx")
            exporter = ONNXExporter(model_loader.model, "cpu", onnx_path)
            exporter.export_model()
            assert os.path.exists(onnx_path)
            assert os.path.getsize(onnx_path) > 0
