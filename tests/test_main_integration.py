import os
import tempfile
from unittest.mock import patch

import pytest
from PIL import Image


class TestMainIntegration:
    @pytest.fixture
    def temp_image(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new("RGB", (256, 256), color="blue")
            img.save(tmp.name)
            yield tmp.name
        os.unlink(tmp.name)

    @pytest.mark.slow
    @patch("main.torch.cuda.is_available")
    @patch("main.plot_benchmark_results")
    def test_main_cpu_mode(self, mock_plot, mock_cuda, temp_image):
        mock_cuda.return_value = False

        with patch(
            "sys.argv", ["main.py", "--image_path", temp_image, "--mode", "cpu", "--topk", "3"]
        ):
            from main import main

            main()

    @pytest.mark.slow
    @patch("main.torch.cuda.is_available")
    def test_main_onnx_mode(self, mock_cuda, temp_image):
        mock_cuda.return_value = False

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test.onnx")
            with patch(
                "sys.argv",
                ["main.py", "--image_path", temp_image, "--mode", "onnx", "--onnx_path", onnx_path],
            ):
                from main import main

                main()
                assert os.path.exists(onnx_path)

    def test_cuda_availability_check(self):
        from main import CUDA_AVAILABLE

        assert isinstance(CUDA_AVAILABLE, bool)

    @pytest.mark.slow
    @patch("main.torch.cuda.is_available")
    def test_main_with_debug_mode(self, mock_cuda, temp_image):
        mock_cuda.return_value = False

        with patch("sys.argv", ["main.py", "--image_path", temp_image, "--mode", "cpu", "-D"]):
            from main import main

            main()
