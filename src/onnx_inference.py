import os

import numpy as np
import onnxruntime as ort
import torch

from src.inference_base import InferenceBase
from src.onnx_exporter import ONNXExporter


class ONNXInference(InferenceBase):
    def __init__(self, model_loader, model_path: str, debug_mode: bool = False):
        super().__init__(model_loader, onnx_path=model_path, debug_mode=debug_mode)

    def load_model(self) -> ort.InferenceSession:
        if not os.path.exists(self.onnx_path):
            onnx_exporter = ONNXExporter(
                self.model_loader.model, self.model_loader.device, self.onnx_path
            )
            onnx_exporter.export_model()
        return ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])

    def predict(self, input_data: torch.Tensor, is_benchmark: bool = False) -> np.ndarray | None:
        super().predict(input_data, is_benchmark)

        input_name = self.model.get_inputs()[0].name
        ort_inputs = {input_name: input_data.cpu().numpy()}
        ort_outs = self.model.run(None, ort_inputs)

        prob = np.zeros(1000)
        if len(ort_outs) > 0:
            prob = ort_outs[0]
            if prob.ndim > 1:
                prob = prob[0]
            prob = np.exp(prob) / np.sum(np.exp(prob))

        return self.get_top_predictions(prob, is_benchmark)
