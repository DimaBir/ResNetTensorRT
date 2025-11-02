import os
from typing import Optional

import numpy as np
import openvino as ov
import torch

from src.inference_base import InferenceBase
from src.onnx_exporter import ONNXExporter
from src.ov_exporter import OVExporter


class OVInference(InferenceBase):
    def __init__(self, model_loader, model_path: str, debug_mode: bool = False):
        super().__init__(model_loader, ov_path=model_path, debug_mode=debug_mode)
        self.core = ov.Core()
        self.ov_model = self.load_model()
        self.compiled_model = self.core.compile_model(self.ov_model, "AUTO")

    def load_model(self) -> ov.Model:
        self.onnx_path = self.ov_path.replace(".ov", ".onnx")

        if not os.path.exists(self.onnx_path):
            onnx_exporter = ONNXExporter(
                self.model_loader.model, self.model_loader.device, self.onnx_path
            )
            onnx_exporter.export_model()

        ov_exporter = OVExporter(self.onnx_path)
        return ov_exporter.export_model()

    def predict(self, input_data: torch.Tensor, is_benchmark: bool = False) -> Optional[np.ndarray]:
        super().predict(input_data, is_benchmark=is_benchmark)

        input_name = next(iter(self.compiled_model.inputs))
        outputs = self.compiled_model(inputs={input_name: input_data.cpu().numpy()})

        prob_key = next(iter(outputs))
        prob = outputs[prob_key]
        prob = np.exp(prob[0]) / np.sum(np.exp(prob[0]))

        return self.get_top_predictions(prob, is_benchmark)
