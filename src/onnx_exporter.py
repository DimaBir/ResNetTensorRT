import os
from typing import Union

import torch
from torch.onnx import export, TrainingMode

DUMMY_INPUT_SHAPE = (1, 3, 224, 224)


class ONNXExporter:
    def __init__(self, model: torch.nn.Module, device: Union[str, torch.device], onnx_path: str):
        self.model = model
        self.onnx_path = onnx_path
        self.device = device if isinstance(device, torch.device) else torch.device(device)

    def export_model(self):
        self.model.eval()
        dummy_input = torch.randn(*DUMMY_INPUT_SHAPE).to(self.device)

        model_dir = os.path.dirname(self.onnx_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        export(
            self.model,
            dummy_input,
            self.onnx_path,
            training=TrainingMode.EVAL,
            verbose=True,
            input_names=["input"],
            output_names=["output"],
        )
        print(f"Model exported to {self.onnx_path}")
