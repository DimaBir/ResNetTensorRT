import torch
from flask import Blueprint, request, jsonify, current_app as app
from redis import Redis
from torchvision.utils import save_image

from .utils.model import ModelLoader
from .utils.image_processor import ImageProcessor
import os
import uuid
import io

bp = Blueprint('core_logic', __name__)

# In-memory model cache
model_cache = {}


def get_redis_client():
    return Redis(host=app.config['REDIS_HOST'], port=app.config['REDIS_PORT'], db=app.config['REDIS_DB'])


@bp.route('/load_model', methods=['POST'])
def handle_load_model():
    data = request.json
    model_type = data.get("model_type", "resnet50")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if model is already loaded
    if model_type not in model_cache:
        model_loader = ModelLoader(model_type=model_type, device=device)
        model_cache[model_type] = model_loader.model

    return jsonify({"message": f"Model {model_type} loaded successfully on {device}"})


@bp.route('/process_image', methods=['POST'])
def handle_process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    image_stream = io.BytesIO(image_file.read())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_processor = ImageProcessor(img_path=image_stream, device=device)
    processed_image = image_processor.process_image()

    tmp_dir = app.config['TMP_DIR']
    os.makedirs(tmp_dir, exist_ok=True)
    image_name = f"{uuid.uuid4()}.png"
    image_path = os.path.join(tmp_dir, image_name)
    save_image(processed_image, image_path)

    redis_client = get_redis_client()
    redis_client.set(image_name, image_path)

    return jsonify({"message": "Image processed successfully", "image_path": image_path})
