# https://github.com/Linaom1214/tensorrt-python
# https://github.com/ultralytics/yolov5

import tensorrt as trt

onnx = "weights/last.onnx"
f = "weights/last.trt"
half = True
workspace = 9
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
builder.create_network(explicit_batch)
profile = builder.create_optimization_profile()
profile.set_shape("input", (1, 3, 224, 224), (32, 3, 224, 224), (32, 3, 224, 224))
config = builder.create_builder_config()
config.max_workspace_size = workspace * 1 << 30
idx = config.add_optimization_profile(profile)
flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(flag)
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(str(onnx)):
    raise RuntimeError(f'failed to load ONNX file: {onnx}')

inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]

if builder.platform_has_fast_fp16 and half:
    config.set_flag(trt.BuilderFlag.FP16)
with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
    t.write(engine.serialize())

