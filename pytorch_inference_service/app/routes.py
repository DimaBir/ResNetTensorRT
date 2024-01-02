from flask import Blueprint, request, jsonify
from redis import Redis
from .pytorch_inference import PyTorchInference
import os

bp = Blueprint('pytorch', __name__)
model_cache = {}

# Initialize Redis client
redis_client = Redis(host='redis', port=6379, db=0)


@bp.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_key = data.get('image_key')  # Key to retrieve the image path from Redis

    image_path = redis_client.get(image_key)
    if not image_path:
        return jsonify({"error": "Image path not found in cache"}), 404

    model_type = data.get('model_type', 'resnet50')

    if model_type not in model_cache:
        model_cache[model_type] = PyTorchInference(model_type=model_type).model

    model = model_cache[model_type]
    pytorch_inference = PyTorchInference(model_type=model.model_type)
    prediction = pytorch_inference.predict(os.path.abspath(image_path.decode('utf-8')))

    return jsonify({"prediction": prediction})


@bp.route('/benchmark', methods=['POST'])
def benchmark():
    data = request.json
    image_key = data.get('image_key')  # Key to retrieve the image path from Redis

    image_path = redis_client.get(image_key)
    if not image_path:
        return jsonify({"error": "Image path not found in cache"}), 404

    model_type = data.get('model_type', 'resnet50')
    num_runs = data.get('num_runs', 100)

    pytorch_inference = PyTorchInference(model_type=model_type)
    benchmark_result = pytorch_inference.benchmark(os.path.abspath(image_path.decode('utf-8')), num_runs=num_runs)

    return jsonify({"benchmark_result": benchmark_result})
