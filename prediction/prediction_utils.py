import logging
from typing import List, Tuple, Union, Dict, Any
import openvino as ov
import torch
import onnxruntime as ort
import numpy as np


def make_prediction(
    model: Union[torch.nn.Module, ort.InferenceSession, ov.CompiledModel],
    img_batch: Union[torch.Tensor, np.ndarray],
    topk: int,
    categories: List[str],
    precision: torch.dtype = None,
) -> None:
    """
    Make and print predictions for the given model, img_batch, topk, and categories.

    :param model: The model (or ONNX Runtime InferenceSession) to make predictions with.
    :param img_batch: The batch of images to make predictions on.
    :param topk: The number of top predictions to show.
    :param categories: The list of categories to label the predictions.
    :param precision: The data type to be used for the predictions (typically torch.float32 or torch.float16) for PyTorch models.
    """
    is_onnx_model = isinstance(model, ort.InferenceSession)
    is_ov_model = isinstance(model, ov.CompiledModel)

    if is_onnx_model:
        # Get the input name for the ONNX model.
        input_name = model.get_inputs()[0].name

        # Run the model with the properly named input.
        ort_inputs = {input_name: img_batch}
        ort_outs = model.run(None, ort_inputs)

        # Assuming the model returns a list with one array of class probabilities.
        if len(ort_outs) > 0:
            prob = ort_outs[0]

            # Checking if prob has more than one dimension and selecting the right one.
            if prob.ndim > 1:
                prob = prob[0]

            # Apply Softmax to get probabilities
            prob = np.exp(prob) / np.sum(np.exp(prob))
    elif is_ov_model:
        # For OV, the input name is usually the first input
        input_name = next(iter(model.inputs))
        outputs = model(inputs={input_name: img_batch})

        # Assuming the model returns a dictionary with one key for class probabilities
        prob_key = next(iter(outputs))
        prob = outputs[prob_key]

        # Apply Softmax to get probabilities
        prob = np.exp(prob[0]) / np.sum(np.exp(prob[0]))

    else:  # PyTorch Model
        if isinstance(img_batch, np.ndarray):
            img_batch = torch.tensor(img_batch)
        else:
            img_batch = img_batch.clone().to(precision)
        model.eval()
        with torch.no_grad():
            outputs = model(img_batch.to(precision))
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

    top_indices = prob.argsort()[-topk:][::-1]
    top_probs = prob[top_indices]

    for i in range(topk):
        probability = top_probs[i]
        if is_onnx_model:
            # Accessing the DataFrame by row number using .iloc[]
            class_label = categories.iloc[top_indices[i]].item()
        else:
            class_label = categories[0][int(top_indices[i])]
        logging.info(f"#{i + 1}: {int(probability * 100)}% {class_label}")