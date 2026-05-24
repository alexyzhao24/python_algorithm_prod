import onnxruntime as ort
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # This will automatically manage CUDA context initialization
import cv2
import argparse

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='opencv_dnn_test.py')

parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to ONNX model file"
)

parser.add_argument(
    "--engine",
    type=str,
    required=True,
    help="Path to TensorRT engine file"
)


parser.add_argument(
    "--input_name",
    type=str,
    required=True,
    help="Input image name"
)

parser.add_argument(
    "--output_onnx",
    type=str,
    required=True,
    help="Output onnx name"
)

parser.add_argument(
    "--output_rt",
    type=str,
    required=True,
    help="Output rt name"
)


args = parser.parse_args()

print(f"Onnx Model file: {args.model}")
print(f"TensorRT Engine file: {args.engine}")
print(f"Input image name: {args.input_name}")
print(f"Output onnx name: {args.output_onnx}")
print(f"Output rt name: {args.output_rt}")


# Load BMP image using OpenCV (assumes grayscale or 3-channel RGB)
img = cv2.imread(args.input_name, cv2.IMREAD_GRAYSCALE)  # use IMREAD_COLOR for RGB

# Check image loaded correctly
if img is None:
    raise RuntimeError("Failed to load image")

# Let's linear scale the image by a scale of 3
img = cv2.resize(img, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

# save the resized image
cv2.imwrite("LinearlyScaled" + args.input_name , img);

# Normalize image to float32
img = img.astype(np.float32) / 255.0  # match preprocessing in C++


### Case I: using onnx model
# Reshape to match ONNX input shape: (1, 1, H, W) or (1, 3, H, W)
img_input = np.expand_dims(img, axis=0)  # shape becomes (1, H, W)
img_input = np.expand_dims(img_input, axis=0)  # shape becomes (1, 1, H, W)

# Load the ONNX model
session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Debug: Print shapes
print("Input shape expected:", session.get_inputs()[0].shape)
print("Feeding shape:", img_input.shape)

# Run inference
output_onnx = session.run([output_name], {input_name: img_input})[0]

# Inspect output
print("Output shape:", output_onnx.shape)
print("Sample values:", output_onnx.ravel()[:10])  # print first few values

# Save output image
if output_onnx.ndim == 4 and output_onnx.shape[1] == 1:
    output_img = output_onnx[0, 0]  # (1, 1, H, W) -> (H, W)
    cv2.imwrite(args.output_onnx, (output_img * 255).astype(np.uint8))


import tensorrt as trt
'''
tensort actually import tensor_bindings:
 Within ~/miniconda3/lib/python3.12/site-packages/tensorrt:
__init__.py: -->
from tensorrt_bindings import *
from tensorrt_bindings import __version__

Now under ~/miniconda3/lib/python3.12/site-packages/tensorrt_bindings:
drwxrwxr-x 3     4096 Dec 18 16:33 plugin
drwxrwxr-x 2     4096 Dec 18 16:33 __pycache__
-rwxrwxr-x 1  4860336 Dec 18 16:33 tensorrt.so
-rw-rw-r-- 1     5684 Dec 18 16:33 __init__.py


__init__.py: -->
from .tensorrt import *
__version__ = "10.7.0"

This is low-level bindings (likely through pybind11) from a Conda-based install.
* You're using the real TensorRT 10.7.0, but only a minimal Conda Python wrapper.
* This explains the unusual structure:
    > you don’t have get_binding_name, get_binding_index (older version) or get_tensor_index (newer version)
    > but you do have a set of new-style API functions like via print(dir(engine) later:
        get_tensor_name(i) / get_tensor_shape(i) / get_tensor_dtype(i) / num_io_tensors ✅
'''

### Case II: using tensorRT model
# 1) Load the .engine file.
# 2) Allocate device memory for input/output.
# 3) Bind the input/output.
# 4) Execute the engine.
# 5) Copy result from GPU to CPU

print("TensorRT Python file:", trt.__file__)
print("TensorRT Python version:", trt.__version__)  # --> version 10.7.0

import tensorrt_bindings.tensorrt as trt_bindings
# print(dir(trt_bindings))  # List available attributes

H, W = img.shape

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load engine from file (assumed to be already built)
with open(args.engine, "rb") as f, trt.Runtime(trt.Logger()) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Let's see what methods available for bindings and tensor name
# print(dir(engine))

# Create context
context = engine.create_execution_context()

# Engine info
print(f"TensorRT Engine Name: {engine.name}")
print(f"Number of I/O Tensors: {engine.num_io_tensors}")
print(f"Number of Layers: {engine.num_layers}")
print(f"Number of Optimization Profiles: {engine.num_optimization_profiles}")

print("\n--- Tensor Info ---")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    shape = engine.get_tensor_shape(name)  # ← FIXED: use name, not index
    dtype = engine.get_tensor_dtype(name)
    location = engine.get_tensor_location(name)
    mode = engine.get_tensor_mode(name)

    mode_str = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
    print(f"[{i}] Name: {name}")
    print(f"     Mode: {mode_str}")
    print(f"     Shape: {shape}")
    print(f"     DType: {dtype}")
    print(f"     Location: {location}")


# Set dynamic input shape
context.set_input_shape("input", img_input.shape)
assert context.all_binding_shapes_specified

# Get the input/output tensor names
input_name = "input"    # must match ONNX input name
output_name = "output"  # must match ONNX output name

# Get input/output tensor indices
input_idx = None
output_idx = None

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.INPUT and name == input_name:
        input_idx = i
    elif mode == trt.TensorIOMode.OUTPUT and name == output_name:
        output_idx = i

assert input_idx is not None, "Input tensor not found."
assert output_idx is not None, "Output tensor not found."

# Allocate memory for input and output
# critical for using pycuda with numpy: conver to int():
#   * The issue is that np.prod(output_shape) returns a NumPy int64,
#   * but pycuda.driver.mem_alloc() expects a Python built-in int (which maps to C++ unsigned long).
input_nbytes = int(img_input.nbytes)   # conversion to int() is critical
output_shape = context.get_tensor_shape(output_name)
output_nbytes = int(np.prod(output_shape) * np.dtype(np.float32).itemsize) # conversion to int() is critical

# Allocate device memory
d_input = cuda.mem_alloc(input_nbytes)
d_output = cuda.mem_alloc(output_nbytes)

# Set up bindings
bindings = [None] * engine.num_io_tensors
bindings[input_idx] = int(d_input)
bindings[output_idx] = int(d_output)

# Transfer input to device
cuda.memcpy_htod(d_input, img_input)

# Run inference
success = context.execute_v2(bindings)
if not success:
    raise RuntimeError("TensorRT inference failed.")

# Copy result back
output_rt = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output_rt, d_output)

# Inspect output
print("Output shape:", output_rt.shape)
print("Sample values:", output_rt.ravel()[:10])  # print first few values
# Save output image if output.ndim == 4 and output.shape[1] == 1:
if output_rt.ndim == 4 and output_rt.shape[1] == 1:
    output_img = output_rt[0, 0]  # (1, 1, H, W) -> (H, W)
    cv2.imwrite(args.output_rt, (output_img * 255).astype(np.uint8))


# Typical execution:
# python opencv_dnn_test.py  --model ../models/SRCNN_dynamic.onnx --input_name ../data/comic.bmp --output_onnx comic_onnx.bmp --output_rt comic_rt.bmp --engine ../models/SRCNN_dynamic.engine
