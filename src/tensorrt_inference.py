import logging

import numpy as np
import torch

from src.inference_base import InferenceBase

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    try:
        import torch_tensorrt as trt
    except ImportError:
        logging.warning("torch-tensorrt not installed. Running in CPU mode only.")
        CUDA_AVAILABLE = False

DUMMY_INPUT_SHAPE = (1, 3, 224, 224)


class TensorRTInference(InferenceBase):
    def __init__(
        self,
        model_loader,
        device: str | torch.device,
        precision: torch.dtype = torch.float32,
        debug_mode: bool = False,
    ):
        self.precision = precision
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        super().__init__(model_loader, debug_mode=debug_mode)
        if CUDA_AVAILABLE:
            self.load_model()

    def load_model(self):
        self.model = self.model_loader.model.to(self.device).eval()

        scripted_model = torch.jit.trace(
            self.model, torch.randn(*DUMMY_INPUT_SHAPE).to(self.device)
        )

        if CUDA_AVAILABLE:
            self.model = trt.compile(
                scripted_model,
                inputs=[trt.Input(DUMMY_INPUT_SHAPE, dtype=self.precision)],
                enabled_precisions={self.precision},
            )

    def predict(self, input_data: torch.Tensor, is_benchmark: bool = False) -> np.ndarray | None:
        super().predict(input_data, is_benchmark=is_benchmark)

        with torch.no_grad():
            outputs = self.model(input_data.to(self.device).to(dtype=self.precision))

        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

        return self.get_top_predictions(prob, is_benchmark)
