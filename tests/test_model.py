import pytest
import torch
from src.model import ModelLoader


class TestModelLoader:
    @pytest.fixture
    def device(self):
        return "cpu"

    def test_init_with_default_model(self, device):
        loader = ModelLoader(model_type="resnet50", device=device)
        assert loader.model is not None
        assert loader.categories is not None
        assert len(loader.categories) == 1000

    def test_init_with_efficientnet(self, device):
        loader = ModelLoader(model_type="efficientnet", device=device)
        assert loader.model is not None

    def test_init_with_mobilenet(self, device):
        loader = ModelLoader(model_type="mobilenet_v2", device=device)
        assert loader.model is not None

    def test_unsupported_model_type(self, device):
        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelLoader(model_type="invalid_model", device=device)

    def test_device_handling(self):
        loader = ModelLoader(device="cpu")
        assert isinstance(loader.device, torch.device)
        assert loader.device.type == "cpu"

    def test_model_on_correct_device(self, device):
        loader = ModelLoader(device=device)
        model_device = next(loader.model.parameters()).device
        assert model_device.type == device
