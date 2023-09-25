import tensorrt as trt
import pycuda.driver as cuda

# Initialize CUDA Driver
cuda.init()


def convert_onnx_to_tensorrt(onnx_file_path, engine_file_path, precision_mode):
    # Create a TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create a builder
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network(
            1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Set the workspace size
        config.max_workspace_size = 1 << 30  # 1GB

        # Set the precision mode
        if precision_mode == "FP16":
            config.set_flag(trt.BuilderFlag.FP16)

        # Parse the ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return

        # Build and save the engine
        with builder.build_engine(network, config) as engine:
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())
