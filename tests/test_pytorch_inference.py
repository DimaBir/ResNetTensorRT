import pytest
import torch

from src.model import ModelLoader
from src.pytorch_inference import PyTorchInference


class TestPyTorchInference:
    @pytest.fixture
    def model_loader(self):
        return ModelLoader(device="cpu")

    @pytest.fixture
    def inference(self, model_loader):
        return PyTorchInference(model_loader, device="cpu", debug_mode=False)

    @pytest.fixture
    def input_data(self):
        return torch.randn(1, 3, 224, 224)

    def test_init(self, inference):
        assert inference.device.type == "cpu"
        assert inference.model is not None

    def test_load_model(self, inference):
        model = inference.load_model()
        assert model is not None
        assert next(model.parameters()).device.type == "cpu"

    def test_predict_shape(self, inference, input_data):
        result = inference.predict(input_data, is_benchmark=False)
        assert result is not None

    def test_predict_benchmark_mode(self, inference, input_data):
        result = inference.predict(input_data, is_benchmark=True)
        assert result is None

    def test_model_in_eval_mode(self, inference, input_data):
        inference.predict(input_data)
        assert not inference.model.training

    def test_no_grad_during_inference(self, inference, input_data):
        with torch.no_grad():
            result = inference.predict(input_data, is_benchmark=True)
        assert result is None

    def test_benchmark_execution(self, inference, input_data):
        avg_time, throughput = inference.benchmark(input_data, num_runs=2, warmup_runs=1)
        assert avg_time > 0
        assert throughput > 0
