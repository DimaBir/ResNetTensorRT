import os
import logging
import numpy as np
import openvino as ov
from src.inference_base import InferenceBase
from src.onnx_exporter import ONNXExporter
from src.ov_exporter import OVExporter


class OVInference(InferenceBase):
    def __init__(self, model_loader, model_path):
        super().__init__(model_loader, ov_path=model_path)

        self.core = ov.Core()
        self.ov_model = self.load_model()
        self.compiled_model = self.core.compile_model(self.ov_model, "AUTO")

    def load_model(self):
        # Determine the path for the ONNX model
        self.onnx_path = self.ov_path.replace(".ov", ".onnx")

        # Check if ONNX model exists
        if not os.path.exists(self.onnx_path):
            onnx_exporter = ONNXExporter(self.model_loader.model, self.model_loader.device, self.onnx_path)
            onnx_exporter.export_model()

        ov_exporter = OVExporter(self.onnx_path)
        return ov_exporter.export_model()

    def predict(self, input_data):
        logging.info(f"Running prediction for OV model")

        input_name = next(iter(self.compiled_model.inputs))
        outputs = self.compiled_model.infer(inputs={input_name: input_data.cpu().numpy()})

        # Assuming the model returns a dictionary with one key for class probabilities
        prob_key = next(iter(outputs))
        prob = outputs[prob_key]

        # Apply Softmax to get probabilities
        prob = np.exp(prob[0]) / np.sum(np.exp(prob[0]))

        return self.get_top_predictions(prob)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        super().benchmark(input_data, num_runs, warmup_runs)
