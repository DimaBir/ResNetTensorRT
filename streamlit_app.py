"""
Streamlit interface for ResNet TensorRT benchmark application.

This app provides a user-friendly web interface to:
- Upload or select images for inference
- Configure benchmark parameters
- Run inference across different backends (PyTorch, ONNX, OpenVINO, TensorRT)
- Display predictions and benchmark results
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from common.utils import (
    DEFAULT_IMAGE_PATH,
    DEFAULT_ONNX_PATH,
    DEFAULT_OV_PATH,
    DEFAULT_TOPK,
    INFERENCE_MODES,
)
from src.image_processor import ImageProcessor
from src.model import ModelLoader
from src.onnx_inference import ONNXInference
from src.ov_inference import OVInference
from src.pytorch_inference import PyTorchInference

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    try:
        import torch_tensorrt  # noqa: F401
        from src.tensorrt_inference import TensorRTInference
    except ImportError:
        CUDA_AVAILABLE = False
        st.warning("torch-tensorrt not installed. TensorRT and CUDA modes will be unavailable.")


def display_image(image_path: str):
    """Display the input image."""
    img = Image.open(image_path)
    st.image(img, caption="Input Image", use_container_width=True)


def run_inference(
    image_path: str,
    mode: str,
    topk: int,
    onnx_path: str,
    ov_path: str,
    debug_mode: bool = False,
) -> dict[str, tuple[float, float]]:
    """
    Run inference based on selected mode.

    Args:
        image_path: Path to input image
        mode: Inference mode (onnx, ov, cpu, cuda, tensorrt, all)
        topk: Number of top predictions to show
        onnx_path: Path to ONNX model
        ov_path: Path to OpenVINO model
        debug_mode: Enable debug logging

    Returns:
        Dictionary of benchmark results {model_name: (avg_time_ms, throughput)}
    """
    benchmark_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and process image
    model_loader = ModelLoader(device=device)
    img_processor = ImageProcessor(img_path=image_path, device=device)
    img_batch = img_processor.process_image()

    # Create a placeholder for predictions
    predictions_placeholder = st.empty()

    # ONNX inference
    if mode in ["onnx", "all"]:
        with st.spinner("Running ONNX inference..."):
            onnx_inference = ONNXInference(model_loader, onnx_path, debug_mode=debug_mode)
            benchmark_result = onnx_inference.benchmark(img_batch)
            predictions = onnx_inference.predict(img_batch)
            benchmark_results["ONNX (CPU)"] = benchmark_result

            if predictions is not None:
                display_predictions(predictions, model_loader.categories, topk, "ONNX (CPU)")

    # OpenVINO inference
    if mode in ["ov", "all"]:
        with st.spinner("Running OpenVINO inference..."):
            ov_inference = OVInference(model_loader, ov_path, debug_mode=debug_mode)
            benchmark_result = ov_inference.benchmark(img_batch)
            predictions = ov_inference.predict(img_batch)
            benchmark_results["OpenVINO (CPU)"] = benchmark_result

            if predictions is not None:
                display_predictions(predictions, model_loader.categories, topk, "OpenVINO (CPU)")

    # PyTorch CPU inference
    if mode in ["cpu", "all"]:
        with st.spinner("Running PyTorch CPU inference..."):
            pytorch_cpu_inference = PyTorchInference(
                model_loader, device="cpu", debug_mode=debug_mode
            )
            benchmark_result = pytorch_cpu_inference.benchmark(img_batch)
            predictions = pytorch_cpu_inference.predict(img_batch)
            benchmark_results["PyTorch (CPU)"] = benchmark_result

            if predictions is not None:
                display_predictions(predictions, model_loader.categories, topk, "PyTorch (CPU)")

    # CUDA and TensorRT inference (only if CUDA available)
    if CUDA_AVAILABLE:
        # PyTorch CUDA inference
        if mode in ["cuda", "all"]:
            with st.spinner("Running PyTorch CUDA inference..."):
                pytorch_cuda_inference = PyTorchInference(
                    model_loader, device=device, debug_mode=debug_mode
                )
                benchmark_result = pytorch_cuda_inference.benchmark(img_batch)
                predictions = pytorch_cuda_inference.predict(img_batch)
                benchmark_results["PyTorch (CUDA)"] = benchmark_result

                if predictions is not None:
                    display_predictions(
                        predictions, model_loader.categories, topk, "PyTorch (CUDA)"
                    )

        # TensorRT inference
        if mode in ["tensorrt", "all"]:
            precisions = [torch.float16, torch.float32]
            for precision in precisions:
                precision_name = "FP16" if precision == torch.float16 else "FP32"
                with st.spinner(f"Running TensorRT {precision_name} inference..."):
                    tensorrt_inference = TensorRTInference(
                        model_loader, device=device, precision=precision, debug_mode=debug_mode
                    )
                    benchmark_result = tensorrt_inference.benchmark(img_batch)
                    predictions = tensorrt_inference.predict(img_batch)
                    benchmark_results[f"TRT_{precision}"] = benchmark_result

                    if predictions is not None:
                        display_predictions(
                            predictions, model_loader.categories, topk, f"TensorRT {precision_name}"
                        )

    return benchmark_results


def display_predictions(prob, categories, topk: int, model_name: str):
    """Display top-k predictions."""
    top_indices = prob.argsort()[-topk:][::-1]
    top_probs = prob[top_indices]

    st.subheader(f"Predictions - {model_name}")
    for i in range(topk):
        probability = top_probs[i]
        class_label = categories[0][int(top_indices[i])]
        st.write(f"#{i + 1}: {int(probability * 100)}% {class_label}")


def display_benchmark_results(results: dict[str, tuple[float, float]]):
    """Display benchmark results in a table."""
    st.subheader("Benchmark Results")

    # Create DataFrame for display
    import pandas as pd

    data = {
        "Model": list(results.keys()),
        "Avg Time (ms)": [f"{results[model][0]:.2f}" for model in results.keys()],
        "Throughput (samples/sec)": [f"{results[model][1]:.2f}" for model in results.keys()],
    }
    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

    # Display metrics in columns
    cols = st.columns(len(results))
    for idx, (model, (avg_time, throughput)) in enumerate(results.items()):
        with cols[idx]:
            st.metric(label=model, value=f"{avg_time:.2f} ms", delta=f"{throughput:.1f} img/s")


def main():
    st.set_page_config(
        page_title="ResNet TensorRT Benchmark",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🚀 ResNet TensorRT Benchmark Interface")
    st.markdown(
        """
    This application provides a user-friendly interface to benchmark ResNet inference
    across different backends: PyTorch (CPU/CUDA), ONNX, OpenVINO, and TensorRT.
    """
    )

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Image selection/upload
    st.sidebar.subheader("Image Input")
    image_source = st.sidebar.radio("Select image source:", ["Sample Images", "Upload Image"])

    image_path = None
    if image_source == "Sample Images":
        sample_images = []
        inference_dir = Path("./inference")
        if inference_dir.exists():
            sample_images = [str(f) for f in inference_dir.glob("*.jpg")] + [
                str(f) for f in inference_dir.glob("*.png")
            ]

        if sample_images:
            selected_image = st.sidebar.selectbox("Choose a sample image:", sample_images)
            image_path = selected_image
        else:
            st.sidebar.warning("No sample images found in ./inference directory")
            image_path = DEFAULT_IMAGE_PATH
    else:
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                image_path = tmp_file.name

    # Inference mode selection
    st.sidebar.subheader("Inference Settings")

    # Filter available modes based on CUDA availability
    available_modes = INFERENCE_MODES.copy()
    if not CUDA_AVAILABLE:
        available_modes = [m for m in available_modes if m not in ["cuda", "tensorrt"]]
        st.sidebar.info("CUDA/TensorRT modes unavailable (GPU not detected)")

    mode = st.sidebar.selectbox(
        "Select inference mode:",
        available_modes,
        index=available_modes.index("all") if "all" in available_modes else 0,
        help="Choose which inference backend(s) to benchmark",
    )

    topk = st.sidebar.slider(
        "Top-K predictions:",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOPK,
        help="Number of top predictions to display",
    )

    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        onnx_path = st.text_input("ONNX model path:", value=DEFAULT_ONNX_PATH)
        ov_path = st.text_input("OpenVINO model path:", value=DEFAULT_OV_PATH)
        debug_mode = st.checkbox("Enable debug mode", value=False)

    # Main content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Input Image")
        if image_path and os.path.exists(image_path):
            display_image(image_path)
        else:
            st.warning("Please select or upload an image to proceed")

    with col2:
        st.subheader("Actions")

        # System info
        with st.expander("System Information"):
            st.write(f"**Device:** {'CUDA (GPU)' if CUDA_AVAILABLE else 'CPU'}")
            if CUDA_AVAILABLE:
                st.write(f"**GPU Name:** {torch.cuda.get_device_name(0)}")
            st.write(f"**PyTorch Version:** {torch.__version__}")

        # Run benchmark button
        if st.button("▶️ Run Benchmark", type="primary", use_container_width=True):
            if image_path and os.path.exists(image_path):
                try:
                    st.info(f"Running benchmark with mode: **{mode}**")

                    # Run inference and get results
                    results = run_inference(
                        image_path=image_path,
                        mode=mode,
                        topk=topk,
                        onnx_path=onnx_path,
                        ov_path=ov_path,
                        debug_mode=debug_mode,
                    )

                    # Display results
                    st.success("Benchmark completed!")
                    display_benchmark_results(results)

                    # Clean up temporary file if uploaded
                    if image_source == "Upload Image" and image_path.startswith("/tmp"):
                        try:
                            os.unlink(image_path)
                        except Exception:
                            pass

                except Exception as e:
                    st.error(f"Error during benchmark: {str(e)}")
                    st.exception(e)
            else:
                st.error("Please select or upload a valid image first!")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | 
        <a href='https://github.com/DimaBir/ResNetTensorRT'>GitHub Repository</a></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
