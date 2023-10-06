from benchmark.benchmark_utils import run_benchmark
from benchmark import PyTorchBenchmark, ONNXBenchmark, OVBenchmark
import openvino as ov
import torch
import onnxruntime as ort


def benchmark_onnx_model(ort_session: ort.InferenceSession):
    run_benchmark(None, None, None, ort_session, onnx=True)


def benchmark_ov_model(ov_model: ov.CompiledModel):
    ov_benchmark = OVBenchmark(ov_model, input_shape=(1, 3, 224, 224))
    ov_benchmark.run()


def benchmark_cuda_model(cuda_model: torch.nn.Module, device: str, dtype: torch.dtype):
    run_benchmark(cuda_model, device, dtype)
