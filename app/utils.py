import torch

from src.model import ModelLoader
from src.image_processor import ImageProcessor
from src.onnx_inference import ONNXInference
from src.ov_inference import OVInference
from src.pytorch_inference import PyTorchInference
from src.tensorrt_inference import TensorRTInference, CUDA_AVAILABLE


def init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loader = ModelLoader(device=device)
    model = model_loader.model
    inference_classes = init_inference_classes(model_loader)
    return {
        'model': model,
        'inference_classes': inference_classes
    }


def init_inference_classes(model_loader):
    # Extracting the inference initialization from the main.py
    inference_classes = {
        "onnx": ONNXInference(model_loader),
        "ov": OVInference(model_loader),
        "cpu": PyTorchInference(model_loader, device="cpu"),
    }
    if CUDA_AVAILABLE:
        inference_classes["cuda"] = PyTorchInference(model_loader, device="cuda")
        inference_classes["tensorrt"] = TensorRTInference(model_loader, device="cuda")
    return inference_classes


def run_benchmark(initialization_data):
    # Use the initialized data
    model = initialization_data['model']
    inference_classes = initialization_data['inference_classes']

    benchmark_results = {}
    for key, inference_class in inference_classes.items():
        benchmark_results[key] = inference_class.benchmark(img_batch)

    return benchmark_results


def run_prediction(image_path, initialization_data):
    # Use the initialized data
    model = initialization_data['model']
    inference_classes = initialization_data['inference_classes']

    img_processor = ImageProcessor(image_path)
    img_batch = img_processor.process_image()
    results = {}
    for key, inference_class in inference_classes.items():
        results[key] = inference_class.predict(img_batch)
    return results
