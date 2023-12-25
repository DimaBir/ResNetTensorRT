import time
import logging
import hashlib

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from PIL import Image
from io import BytesIO
from config import SSL_CERT_PATH, SSL_KEY_PATH
from werkzeug.exceptions import RequestEntityTooLarge
from flask_limiter.util import get_remote_address
from flask_limiter import Limiter, RateLimitExceeded
import os
import uuid

import sys

sys.path.append("/usr/src/app")
from common.utils import cuda_is_available

# Importing model and inference classes
from src.image_processor import ImageProcessor
from src.model import ModelLoader
from src.onnx_inference import ONNXInference
from src.ov_inference import OVInference
from src.pytorch_inference import PyTorchInference
from src.tensorrt_inference import TensorRTInference

app = Flask(__name__)


UPLOAD_FOLDER = "static/user_files"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
MAX_FILES_IN_UPLOAD_FOLDER = 10

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# Configure rate limiting
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=["5 per minute"])


@app.errorhandler(RateLimitExceeded)
def handle_rate_limit_error(e):
    response = jsonify({"error": "Rate limit exceeded"})
    response.status_code = 429  # Too Many Requests
    return response


# Function to check if the file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to process the uploaded image
def process_image(file_path):
    # Open the image file
    with Image.open(file_path) as image:
        img_processor = ImageProcessor(device="cpu")
        return img_processor.process_image(image)


# Function to manage file limit in the upload folder
def manage_file_limit(upload_folder):
    files_in_directory = os.listdir(upload_folder)
    number_of_files = len(files_in_directory)

    if number_of_files >= MAX_FILES_IN_UPLOAD_FOLDER:
        oldest_file = min(
            files_in_directory,
            key=lambda x: os.path.getctime(os.path.join(upload_folder, x)),
        )
        os.remove(os.path.join(upload_folder, oldest_file))


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


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return "File is too large", 413


@app.route("/demo")
def index():
    return render_template("demo.html")


@app.route("/process", methods=["POST"])
def process_request():
    image_file = request.files.get("image")
    model_type = request.form.get("inferenceMode")
    mode = request.form.get("mode")
    cnn_model = request.form.get("cnnModel")  # Retrieve the selected CNN model

    # Add logging statements
    logging.info(
        "Received request with model_type: %s, mode: %s, image_file: %s",
        model_type,
        mode,
        image_file.filename,
    )
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    if image_file is None or image_file.filename == "":
        logging.error("No file part or no selected file")
        return jsonify({"error": "No file part or no selected file"}), 400

    if not allowed_file(image_file.filename):
        logging.error("Invalid file type: %s", image_file.filename)
        return jsonify({"error": "Invalid file format. Allowed formats are png, jpg, jpeg, gif."}), 400

    # Generate a unique filename using a hash
    file_content = image_file.read()
    file_hash = hashlib.sha256(file_content).hexdigest()
    ext = image_file.filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{file_hash}.{ext}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

    if not os.path.exists(file_path):
        # Save the uploaded file with the unique name
        with open(file_path, 'wb') as f:
            f.write(file_content)

    # Save the uploaded file with the unique name
    image_file.seek(0)
    image_file.save(file_path)

    logging.info("Saved file: %s", file_path)

    # Process the uploaded image using ImageProcessor
    device = "cuda" if cuda_is_available() else "cpu"
    img_processor = ImageProcessor(img_path=file_path, device=device)
    img_batch = img_processor.process_image()
    # img_batch = img_processor.process_image_official()

    if img_batch is None:
        return jsonify({"error": "Invalid file type"}), 400

    logging.info("Loading pre-trained model, for %s", cnn_model)
    model_loader = ModelLoader(model_type=cnn_model, device=device)

    if mode == "benchmark":

        # Benchmark mode logic
        logging.info("Running all benchmarks")
        results = run_all_benchmarks(img_batch)
        return jsonify({"benchmark": results})

    elif mode == "predict":

        # Predict mode logic
        logging.info("Running prediction for model type: %s", model_type)
        inference_class = get_inference_class(model_type, model_loader)
        if inference_class is None:
            logging.error("Invalid model type selected: %s", model_type)
            return jsonify({"error": "Invalid model type selected"}), 400

        start_time = time.time()
        predictions = inference_class.predict(img_batch)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000

        return jsonify({"predictions": predictions, "inference_time": inference_time})
    else:
        # Handle unexpected mode
        logging.error("Invalid mode selected: %s", mode)
        return jsonify({"error": "Invalid mode selected"}), 400


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    app.run(
        host="0.0.0.0", port=5000, ssl_context=(SSL_CERT_PATH, SSL_KEY_PATH), debug=True
    )
