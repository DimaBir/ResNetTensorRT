from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
from config import SSL_CERT_PATH, SSL_KEY_PATH

import sys
sys.path.append('/usr/src/app')
from common.utils import cuda_is_available

# Importing model and inference classes
from src.image_processor import ImageProcessor
from src.model import ModelLoader
from src.onnx_inference import ONNXInference
from src.ov_inference import OVInference
from src.pytorch_inference import PyTorchInference
from src.tensorrt_inference import TensorRTInference

app = Flask(__name__)


def process_image(image_file):
    image = Image.open(BytesIO(image_file.read()))
    img_processor = ImageProcessor(device="cpu")
    return img_processor.process_image(image)


def get_inference_class(model_type, model_loader):
    if model_type == "pytorch":
        return PyTorchInference(model_loader, device="cpu")
    elif model_type == "onnx":
        return ONNXInference(model_loader, "./models/model.onnx")
    elif model_type == "ov":
        return OVInference(model_loader, "./models/model.ov")
    elif model_type == "tensorrt":
        return TensorRTInference(model_loader, device="cpu")
    elif model_type == "all":
        return None  # Placeholder for 'all' models


def run_all_benchmarks(img_batch):
    model_loader = ModelLoader(device="cpu")
    benchmark_results = {}

    # PyTorch CPU Benchmark
    pytorch_cpu_inference = PyTorchInference(model_loader, device="cpu")
    benchmark_results["PyTorch (CPU)"] = pytorch_cpu_inference.benchmark(img_batch)

    # PyTorch GPU Benchmark
    if cuda_is_available():
        pytorch_gpu_inference = PyTorchInference(model_loader, device="cuda")
        benchmark_results["PyTorch (GPU)"] = pytorch_gpu_inference.benchmark(img_batch)

    # ONNX CPU Benchmark
    onnx_inference = ONNXInference(model_loader, "path_to_onnx_model")
    benchmark_results["ONNX (CPU)"] = onnx_inference.benchmark(img_batch)

    # OpenVINO CPU Benchmark
    ov_inference = OVInference(model_loader, "path_to_ov_model")
    benchmark_results["OpenVINO (CPU)"] = ov_inference.benchmark(img_batch)

    # TensorRT CPU Benchmark
    if cuda_is_available():
        tensorrt_inference = TensorRTInference(model_loader, device="cpu")
        benchmark_results["TensorRT (CPU)"] = tensorrt_inference.benchmark(img_batch)

    return benchmark_results


@app.route("/demo")
def index():
    return render_template("demo.html")


@app.route("/process", methods=["POST"])
def process_request():
    image_file = request.files.get("image")
    model_type = request.form.get("model")
    mode = request.form.get("mode")

    img_batch = process_image(image_file)
    model_loader = ModelLoader(device="cpu")

    if mode == "benchmark" and model_type == "all":
        results = run_all_benchmarks(img_batch)
        return jsonify({"benchmark": results})

    inference_class = get_inference_class(model_type, model_loader)
    if inference_class is None:
        return jsonify({"error": "Invalid model type selected"}), 400

    if mode == "predict":
        results = inference_class.predict(img_batch)
        return jsonify({"predictions": results})
    elif mode == "benchmark":
        results = inference_class.benchmark(img_batch)
        return jsonify({"benchmark": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, ssl_context=(SSL_CERT_PATH, SSL_KEY_PATH), debug=True)
