import torch
import torch_tensorrt as trt
import logging


class TensorRTInference:
    def __init__(self, model_loader, precision=torch.float32):
        self.model_loader = model_loader
        self.model = None
        self.precision = precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        # Load the PyTorch model
        self.model = self.model_loader.model.to(self.device).eval()

        # Convert the PyTorch model to TensorRT
        self.model = trt.ts.compile(
            self.model, inputs=[trt.Input((1, 3, 224, 224), dtype=self.precision)]
        )

    def predict(self, img_batch, topk: int):
        logging.info(
            f"Running prediction for TensorRT (CUDA) model with {self.precision} precision"
        )
        with torch.no_grad():
            outputs = self.model(img_batch.to(self.device))
        # Post-processing and printing results
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

        return self.log_top_predictions(prob, topk)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        super().benchmark(input_data, num_runs, warmup_runs)
