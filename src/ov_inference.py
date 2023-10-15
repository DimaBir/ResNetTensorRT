import os
import logging
import numpy as np
import openvino as ov
from src.inference_base import InferenceBase
from onnx_exporter import ONNXExporter
from ov_exporter import OVExporter


class OVInference(InferenceBase):
    def __init__(self, model_path, image_processor, categories):
        super().__init__(model_path, image_processor, categories)

    def load_model(self):
        # Determine the path for the ONNX model
        onnx_model_path = self.model_path.replace(".ov", ".onnx")

        # Check if ONNX model exists
        if not os.path.exists(onnx_model_path):
            onnx_exporter = ONNXExporter()
            onnx_exporter.export_and_save(onnx_model_path)

        # Check if OV model exists
        if not os.path.exists(self.model_path):
            ov_exporter = OVExporter()
            ov_exporter.export_and_save(self.model_path)

        # Load the OV model using OpenVINO's API
        ie = ov.IECore()
        model = ie.read_network(model=self.model_path)
        exec_net = ie.load_network(network=model, device_name="CPU")
        return exec_net

    def predict(self, input_data, topk: int):
        # Run the OV model inference
        logging.info(f"Running prediction for OV model")
        input_name = next(iter(self.model.inputs))
        outputs = self.model.infer({input_name: input_data})

        # Extract probabilities and log top-k predictions
        prob_key = next(iter(outputs))
        prob = outputs[prob_key]
        prob = np.exp(prob[0]) / np.sum(np.exp(prob[0]))

        return self.get_top_predictions(prob, topk)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        super().benchmark(input_data, num_runs, warmup_runs)
