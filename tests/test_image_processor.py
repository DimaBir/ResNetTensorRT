import os
import tempfile

import pytest
import torch
from PIL import Image

from src.image_processor import ImageProcessor


class TestImageProcessor:
    @pytest.fixture
    def temp_image_path(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new("RGB", (256, 256), color="red")
            img.save(tmp.name)
            yield tmp.name
        os.unlink(tmp.name)

    @pytest.fixture
    def device(self):
        return "cpu"

    def test_init(self, temp_image_path, device):
        processor = ImageProcessor(temp_image_path, device)
        assert processor.img_path == temp_image_path
        assert isinstance(processor.device, torch.device)
        assert processor.device.type == device

    def test_process_image_shape(self, temp_image_path, device):
        processor = ImageProcessor(temp_image_path, device)
        result = processor.process_image()

        assert result.shape == (1, 3, 224, 224)
        assert result.device.type == device

    def test_process_image_normalization(self, temp_image_path, device):
        processor = ImageProcessor(temp_image_path, device)
        result = processor.process_image()

        assert result.dtype == torch.float32
        assert result.min() >= -3.0
        assert result.max() <= 3.0

    def test_invalid_image_path(self, device):
        processor = ImageProcessor("nonexistent.jpg", device)
        with pytest.raises(FileNotFoundError):
            processor.process_image()

    def test_transform_pipeline(self, temp_image_path, device):
        processor = ImageProcessor(temp_image_path, device)
        assert processor.transform is not None
