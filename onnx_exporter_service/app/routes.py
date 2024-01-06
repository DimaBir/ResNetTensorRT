from flask import Blueprint, request, jsonify
from .onnx_exporter import ONNXExporter
import torch

bp = Blueprint('onnx_exporter', __name__)


@bp.route('/export', methods=['POST'])
def export_model():
    data = request.json
    model_path = data.get('model_path')
    onnx_path = data.get('onnx_path')
    device = data.get('device', 'cpu')

    # Load the model (this part depends on how you're storing/loading models)
    model = torch.load(model_path)

    exporter = ONNXExporter(model, device, onnx_path)
    exporter.export_model()

    return jsonify({"message": f"Model exported to {onnx_path}"})