from unittest.mock import patch

from common.utils import parse_arguments, INFERENCE_MODES, DEFAULT_TOPK


class TestParseArguments:
    def test_default_values(self):
        with patch("sys.argv", ["prog"]):
            args = parse_arguments()
            assert args.topk == DEFAULT_TOPK
            assert args.mode == "all"
            assert args.DEBUG is False

    def test_custom_image_path(self):
        with patch("sys.argv", ["prog", "--image_path", "/path/to/image.jpg"]):
            args = parse_arguments()
            assert args.image_path == "/path/to/image.jpg"

    def test_custom_topk(self):
        with patch("sys.argv", ["prog", "--topk", "10"]):
            args = parse_arguments()
            assert args.topk == 10

    def test_mode_selection(self):
        for mode in INFERENCE_MODES:
            with patch("sys.argv", ["prog", "--mode", mode]):
                args = parse_arguments()
                assert args.mode == mode

    def test_debug_flag(self):
        with patch("sys.argv", ["prog", "-D"]):
            args = parse_arguments()
            assert args.DEBUG is True

        with patch("sys.argv", ["prog", "--DEBUG"]):
            args = parse_arguments()
            assert args.DEBUG is True

    def test_custom_onnx_path(self):
        with patch("sys.argv", ["prog", "--onnx_path", "/custom/path.onnx"]):
            args = parse_arguments()
            assert args.onnx_path == "/custom/path.onnx"

    def test_custom_ov_path(self):
        with patch("sys.argv", ["prog", "--ov_path", "/custom/path.ov"]):
            args = parse_arguments()
            assert args.ov_path == "/custom/path.ov"
