from src.onnx_inference import ONNXInference
from src.onnx_exporter import ONNXExporter
import onnxruntime as ort
import os

class ONNXInferenceCUDA(ONNXInference):
    def __init__(self, model_loader, model_path, debug_mode=False):
        """
        Initialize the ONNXInference object.

        :param model_loader: Object responsible for loading the model and categories.
        :param model_path: Path to the ONNX model.
        :param debug_mode: If True, print additional debug information.
        """
        super().__init__(model_loader, model_path, debug_mode=debug_mode)

    def load_model(self):
        """
        Load the ONNX model. If the model does not exist, export it.

        :return: Loaded ONNX model.
        """
        if not os.path.exists(self.onnx_path):
            onnx_exporter = ONNXExporter(
                self.model_loader.model, self.model_loader.device, self.onnx_path
            )
            onnx_exporter.export_model()
        return ort.InferenceSession(self.onnx_path, providers=["CUDAExecutionProvider"])
