import argparse
import openvino as ov
import torch
from model import ModelLoader
from onnx_exporter import ONNXExporter
from ov_exporter import OVExporter
import onnxruntime as ort


# Model Initialization Functions
def init_onnx_model(onnx_path: str, model_loader: ModelLoader, device: torch.device) -> ort.InferenceSession:
    onnx_exporter = ONNXExporter(model_loader.model, device, onnx_path)
    onnx_exporter.export_model()
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def init_ov_model(onnx_path: str) -> ov.CompiledModel:
    ov_exporter = OVExporter(onnx_path)
    return ov_exporter.export_model()


def init_cuda_model(model_loader: ModelLoader, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    cuda_model = model_loader.model.to(device)
    if device == "cuda":
        cuda_model = torch.jit.trace(cuda_model, [torch.randn((1, 3, 224, 224)).to(device)])
    return cuda_model


def parse_arguments():
    # Initialize ArgumentParser with description
    parser = argparse.ArgumentParser(description="PyTorch Inference")
    parser.add_argument(
        "--image_path",
        type=str,
        default="./inference/cat3.jpg",
        help="Path to the image to predict",
    )
    parser.add_argument(
        "--topk", type=int, default=5, help="Number of top predictions to show"
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./inference/model.onnx",
        help="Path where model in ONNX format will be exported",
    )
    parser.add_argument(
        "--mode",
        choices=["onnx", "ov", "cuda", "all"],
        required=True,
        help="Mode for exporting and running the model. Choices are: onnx, ov, cuda or all.",
    )

    return parser.parse_args()
