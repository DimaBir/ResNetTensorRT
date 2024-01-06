from flask import Blueprint, request, jsonify
from .inference import OVInference
import torch
import redis

bp = Blueprint('ov_inference', __name__)

@bp.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_key = data.get('model_key')
    image_key = data.get('image_key')

    # Retrieve model and image from Redis
    redis_client = redis.Redis(host='redis', port=6379, db=0)
    model_path = redis_client.get(model_key)
    image_path = redis_client.get(image_key)

    # Load the model and run inference (modify as needed)
    model_loader = ModelLoader(device="cpu")  # Update as per your model loading logic
    ov_inference = OVInference(model_loader, model_path)
    prediction = ov_inference.predict(torch.load(image_path))

    return jsonify({"prediction": prediction})

@bp.route('/benchmark', methods=['POST'])
def benchmark():
    # Similar implementation for benchmarking
    pass
