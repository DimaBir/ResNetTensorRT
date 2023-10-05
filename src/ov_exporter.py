import os
import openvino as ov
from openvino import utils
import openvino.inference_engine as ie


class OVExporter:
    """
    OVExporter handles the conversion of an ONNX model to OpenVINO's internal representation.
    """

    def __init__(self, onnx_model_path: str):
        """
        Initialize the OVExporter with the path to the ONNX model.

        :param onnx_model_path: str
            Path to the ONNX model file.
        """
        self.onnx_model_path = onnx_model_path

    def export_model(self) -> ie.IENetwork:
        """
        Convert the ONNX model to OpenVINO's internal representation.

        :return: ov.ie.IENetwork
            The converted OpenVINO model.
        """
        if not os.path.isfile(self.onnx_path):
            raise ValueError(f"ONNX model wasn't found in path: {self.onnx_path}")

        # Convert the ONNX model to OpenVINO's internal representation
        ov_model = utils.convert_model(self.onnx_model_path)
        return ov_model
