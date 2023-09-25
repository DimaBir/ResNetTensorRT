from model import ModelLoader
from image_processor import ImageProcessor
from benchmark import Benchmark
import torch
import torch_tensorrt
import logging
import argparse

# Set up logging
logging.basicConfig(filename='model.log', level=logging.INFO)

def main():
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument('--image_path', type=str, default='./inference/cat3.jpg', required=True, help='Path to the image to predict')
    parser.add_argument('--topk', type=int, default=1, help='Number of top predictions to show')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topk = args.topk

    model_loader = ModelLoader(device=device)
    img_processor = ImageProcessor(img_path=args.image_path, device=device)
    img_batch = img_processor.process_image()

    prob = model_loader.predict(img_batch)
    probs, classes = torch.topk(prob, topk)
    for i in range(topk):
        probability = probs[i].item()
        class_label = model_loader.categories[0][int(classes[i])]
        logging.info("My prediction: %{} {}".format(int(probability * 100), class_label))

    logging.info("Running Benchmark for CPU")
    benchmark_cpu = Benchmark(model_loader.model.to("cpu"), device="cpu", dtype=torch.float32)
    benchmark_cpu.run()

    logging.info("Running Benchmark for CUDA")
    benchmark_cuda = Benchmark(model_loader.model.to("cuda"), device="cuda", dtype=torch.float32)
    benchmark_cuda.run()

    print("Tracing CUDA model")
    traced_model = torch.jit.trace(model_loader.model, [torch.randn((1, 3, 224, 224)).to("cuda")])

    for precision in [torch.float32, torch.float16]:
        logging.info(f"Compiling and Running Inference Benchmark for TensorRT with precision: {precision}")
        trt_model = torch_tensorrt.compile(
            traced_model,
            inputs=[torch_tensorrt.Input((1, 3, 224, 224), dtype=precision)],
            enabled_precisions={precision}
        )
        benchmark_trt = Benchmark(trt_model, device="cuda", dtype=precision)
        benchmark_trt.run()

        print("Making prediction with TensorRT model")
        trt_model.eval()

        with torch.no_grad():
            outputs = trt_model(img_batch.to(precision))
        prob_trt = torch.nn.functional.softmax(outputs[0], dim=0)

        probs_trt, classes_trt = torch.topk(prob_trt, topk)
        for i in range(topk):
            probability = probs_trt[i].item()
            class_label = model_loader.categories[0][int(classes_trt[i])]
            print("%{} {}".format(int(probability * 100), class_label))


if __name__ == "__main__":
    main()
