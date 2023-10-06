import onnxruntime as ort
import openvino as ov
import numpy as np
import torch
from typing import List
from prediction.prediction_utils import make_prediction


# Prediction Functions
def predict_onnx_model(
    ort_session: ort.InferenceSession,
    img_batch: np.ndarray,
    topk: int,
    categories: List[str],
):
    make_prediction(ort_session, img_batch.cpu().numpy(), topk, categories)


def predict_ov_model(
    ov_model: ov.CompiledModel, img_batch: np.ndarray, topk: int, categories: List[str]
):
    make_prediction(ov_model, img_batch.cpu().numpy(), topk, categories)


def predict_cuda_model(
    cuda_model: torch.nn.Module,
    img_batch: torch.Tensor,
    topk: int,
    categories: List[str],
    precision: torch.dtype,
):
    make_prediction(cuda_model, img_batch, topk, categories, precision)
