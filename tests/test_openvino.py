import os
import tempfile

import pytest

from src.model import ModelLoader
from src.onnx_exporter import ONNXExporter
from src.ov_exporter import OVExporter


class TestOVExporter:
    @pytest.fixture
    def model_loader(self):
        return ModelLoader(device="cpu")

    @pytest.fixture
    def temp_onnx_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            model_loader = ModelLoader(device="cpu")
            exporter = ONNXExporter(model_loader.model, "cpu", onnx_path)
            exporter.export_model()
            yield onnx_path

    def test_export_model(self, temp_onnx_path):
        exporter = OVExporter(temp_onnx_path)
        ov_model = exporter.export_model()
        assert ov_model is not None

    def test_invalid_onnx_path(self):
        exporter = OVExporter("nonexistent.onnx")
        with pytest.raises(ValueError, match="ONNX model not found"):
            exporter.export_model()

    def test_exporter_init(self, temp_onnx_path):
        exporter = OVExporter(temp_onnx_path)
        assert exporter.onnx_path == temp_onnx_path
        assert exporter.core is not None
