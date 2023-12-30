import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import openvino as ov

from common.utils import OV_PRECISION_FP32, OV_PRECISION_FP16
from src.inference_base import InferenceBase
from src.onnx_exporter import ONNXExporter
from src.ov_exporter import OVExporter


class OVInference(InferenceBase):
    def __init__(
        self,
        model_loader,
        model_path,
        precision=OV_PRECISION_FP32,
        execution_mode="PERFORMANCE",
        debug_mode=False,
    ):
        """
        Initialize the OVInference object.

        :param model_loader: Object responsible for loading the model and categories.
        :param model_path: Path to the OpenVINO model.
        :param precision: Precision type for the model ('FP32', 'FP16').
        :param execution_mode: Execution mode for inference ('ACCURACY' or 'PERFORMANCE').
        :param debug_mode: If True, print additional debug information.
        """
        super().__init__(model_loader, ov_path=model_path, debug_mode=debug_mode)
        self.core = ov.Core()

        # Set execution mode
        if execution_mode == "ACCURACY":
            self.core.set_property(
                "CPU",
                {
                    ov.properties.hint.execution_mode(): ov.properties.hint.ExecutionMode.ACCURACY
                },
            )
        else:
            self.core.set_property(
                "CPU",
                {
                    ov.properties.hint.execution_mode(): ov.properties.hint.ExecutionMode.PERFORMANCE
                },
            )

        self.precision = precision
        self.ov_model = self.load_model()
        self.compiled_model = self.compile_model()

    def load_model(self):
        """
        Load the OpenVINO model. If the ONNX model does not exist, export it.

        :return: Loaded OpenVINO model.
        """
        self.onnx_path = self.ov_path.replace(".ov", ".onnx")

        if not os.path.exists(self.onnx_path):
            onnx_exporter = ONNXExporter(
                self.model_loader.model, self.model_loader.device, self.onnx_path
            )
            onnx_exporter.export_model()
        logging.info("Loaded model")

        ov_exporter = OVExporter(self.onnx_path)
        logging.info("Exported model")
        return ov_exporter.export_model()

    def compile_model(self):
        """
        Compile the OpenVINO model with the specified precision.

        :return: Compiled OpenVINO model.
        """
        try:
            # Set inference precision
            if self.precision == OV_PRECISION_FP16:
                self.core.set_property(
                    "CPU",
                    {
                        ov.properties.hints.inference_precision: ov.properties.hints.Precision.FP16
                    },
                )
            elif self.precision == OV_PRECISION_FP32:
                self.core.set_property(
                    "CPU",
                    {
                        ov.properties.hints.inference_precision: ov.properties.hints.Precision.FP32
                    },
                )

            return self.core.compile_model(self.ov_model, "AUTO")
        except Exception as e:
            logging.error(f"Error during model compilation: {e}")
            raise

    def predict(self, input_data, is_benchmark=False):
        """
        Run prediction on the input data using the OpenVINO model.

        :param input_data: Data to run the prediction on.
        :param is_benchmark: If True, the prediction is part of a benchmark run.
        :return: Top predictions based on the probabilities.
        """
        logging.info(f"Entered predict")
        super().predict(input_data, is_benchmark=is_benchmark)

        input_name = next(iter(self.compiled_model.inputs))
        logging.info(f"Compiled inputs ")
        outputs = self.compiled_model(inputs={input_name: input_data.cpu().numpy()})
        logging.info(f"Compiled model ")

        # Extract probabilities from the output
        prob_key = next(iter(outputs))
        prob = outputs[prob_key]
        logging.info(f"Extract probabilities")

        # Convert to FP32 if the model precision is FP16
        if self.precision == OV_PRECISION_FP16:
            prob = prob.astype(np.float32)

        # Apply softmax to the probabilities
        prob = F.softmax(torch.from_numpy(prob[0]), dim=0).numpy()

        return self.get_top_predictions(prob, is_benchmark)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        """
        Benchmark the prediction performance using the OpenVINO model.

        :param input_data: Data to run the benchmark on.
        :param num_runs: Number of runs for the benchmark.
        :param warmup_runs: Number of warmup runs before the benchmark.
        :return: Average inference time in milliseconds.
        """
        return super().benchmark(input_data, num_runs, warmup_runs)

    def get_top_predictions(self, prob: np.ndarray, is_benchmark=False):
        """
        Get the top predictions based on the probabilities.
        """
        if is_benchmark:
            return None

        # Get the top indices and probabilities
        top_indices = prob.argsort()[-self.topk :][::-1]
        top_probs = prob[top_indices]

        # Prepare the list of predictions
        predictions = []
        for i in range(self.topk):
            probability = top_probs[i]
            class_label = self.categories[0][int(top_indices[i])]
            predictions.append({"label": class_label, "confidence": float(probability)})

            # Log the top predictions
            logging.info(f"#{i + 1}: {probability * 100:.2f}% {class_label}")

        return predictions
