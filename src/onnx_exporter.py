import torch
from torch.onnx import export
from torchvision import models

class ONNXExporter:
    def __init__(self, model, onnx_path: str):
        self.model = model
        self.onnx_path = onnx_path

    def export_model(self):
        self.mode.eval()

        # Define dummy input tensor
        x = torch.randn(1, 3, 224, 224).to(self.model.device)

        # Export model as ONNX
        export(self.model, x, self.onnx_path, verbose=True, input_names=['input'], output_names=['output'])
        print(f"Model exported to {self.onnx_path}")