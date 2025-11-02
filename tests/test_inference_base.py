import numpy as np
import pytest
import torch

from src.inference_base import InferenceBase
from src.model import ModelLoader


class MockInference(InferenceBase):
    def load_model(self):
        return None

    def predict(self, input_data, is_benchmark=False):
        super().predict(input_data, is_benchmark)
        return np.random.rand(1000)


class TestInferenceBase:
    @pytest.fixture
    def model_loader(self):
        return ModelLoader(device="cpu")

    @pytest.fixture
    def inference(self, model_loader):
        return MockInference(model_loader, debug_mode=False)

    @pytest.fixture
    def input_data(self):
        return torch.randn(1, 3, 224, 224)

    def test_init(self, inference):
        assert inference.topk == 5
        assert inference.batch_size == 8
        assert inference.debug_mode is False

    def test_custom_topk(self, model_loader):
        inference = MockInference(model_loader, topk=3)
        assert inference.topk == 3

    def test_custom_batch_size(self, model_loader):
        inference = MockInference(model_loader, batch_size=16)
        assert inference.batch_size == 16

    def test_prepare_batch(self, inference, input_data):
        batch = inference._prepare_batch(input_data)
        assert batch.shape[0] == inference.batch_size
        assert batch.shape[1:] == (3, 224, 224)

    def test_get_top_predictions(self, inference):
        prob = np.random.rand(1000)
        result = inference.get_top_predictions(prob, is_benchmark=False)
        assert result is not None

    def test_get_top_predictions_benchmark(self, inference):
        prob = np.random.rand(1000)
        result = inference.get_top_predictions(prob, is_benchmark=True)
        assert result is None

    def test_benchmark_returns_tuple(self, inference, input_data):
        result = inference.benchmark(input_data, num_runs=2, warmup_runs=1)
        assert isinstance(result, tuple)
        assert len(result) == 2
        avg_time, throughput = result
        assert avg_time > 0
        assert throughput > 0

    def test_predict_calls_parent(self, inference, input_data):
        result = inference.predict(input_data, is_benchmark=False)
        assert result is not None
