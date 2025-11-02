import os

import openvino as ov


class OVExporter:
    def __init__(self, onnx_model_path: str):
        self.onnx_path = onnx_model_path
        self.core = ov.Core()

    def export_model(self) -> ov.Model:
        if not os.path.isfile(self.onnx_path):
            raise ValueError(f"ONNX model not found: {self.onnx_path}")

        return self.core.read_model(self.onnx_path)
