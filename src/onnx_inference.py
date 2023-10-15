import os
import logging
from src.inference_base import InferenceBase
import onnxruntime as ort
import numpy as np

from src.onnx_exporter import ONNXExporter


class ONNXInference(InferenceBase):
    def __init__(self, model_loader, model_path, image_processor):
        super().__init__(model_loader, model_path, image_processor)

    def load_model(self):
        if not os.path.exists(self.model_path):
            exporter = ONNXExporter()
            exporter.export_and_save(self.model_path)
        return ort.InferenceSession(self.model_path)

    def predict(self, input_data, topk: int):
        logging.info(f"Running prediction for ONNX model")
        input_name = self.model.get_inputs()[0].name
        ort_inputs = {input_name: input_data}
        ort_outs = self.model.run(None, ort_inputs)
        if len(ort_outs) > 0:
            prob = ort_outs[0]
            if prob.ndim > 1:
                prob = prob[0]
            prob = np.exp(prob) / np.sum(np.exp(prob))
        return self.get_top_predictions(prob, topk)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        super().benchmark(input_data, num_runs, warmup_runs)
