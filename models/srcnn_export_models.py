import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor
import torch.nn.functional as F
from pathlib import Path
import cv2
import os

import matplotlib.pyplot as plt

import scipy.io as sio
import numpy as np

'''
 To avoid border effects, there is NO padding!

 For training, the image patches, there is
    Input low-res patch size: 31x31
    Output high-res patch size: 21x21
'''
class SRCNN(nn.Module):
    def __init__(
        self,
        channels=1,
        conv1_out=64,
        conv2_out=32,
        dtype=torch.float64
        ):
        super().__init__()

        # determine dimensions
        self.channels = channels


        ## Layer Conv1: conv + relu --> Patch Extraction and Representation similar to PCA/DCT/Haar
        self.conv1 = nn.Sequential(
            ## batch size N=128 and input size (C_in=1, C_out=64, H_in=33, W_in=33): kerel_size=9, and stride=1, padding=0 (dilation=1)
            # output can be precisely described as:
            #        out(N_i, C_out_j) = bias(Cout_j) + sum_{k=0}^{C_in-1} weight(C_out_j, k) * input(N_i, k)
            ## Out is determined by input and network structure
            # H_out = (H_in + 2*padding[0] - dilation[0]x(kernel_size[0] - 1) - 1)/stride[0] + 1
            #       = (33 + 2x0 - 1x(9-1) - 1)/1 + 1 ==> 25
            # W_out = (W_in + 2*padding[1] - dialation[1]x(kernel_size[1] - 1) - 1)/stride[1] + 1
            #       = (33 + 2x0 - 1x(9-1) - 1)/1 + 1 ==> 25
            nn.Conv2d(in_channels=channels, out_channels=conv1_out, kernel_size=9, padding=0, stride=1),
            nn.ReLU(),
        )

        ## Layer Conv2: conv + relu --> Non-Linear Mapping from low-res patch to high-res patch
        self.conv2 = nn.Sequential(
            ## batch size N=128 and input size (C_in=64, C_out=32, C_in=1, H_in=25, W_in=25): kerel_size=1, and stride=1, padding=0 (dilation=1)
            ## Out is determined by input and network structure
            # H_out = (H_in + 2*padding[0] - dilation[0]x(kernel_size[0] - 1) - 1)/stride[0] + 1
            #       = (25 + 2x0 - 1x(1-1) - 1)/1 + 1 ==> 25
            # W_out = (W_in + 2*padding[1] - dialation[1]x(kernel_size[1] - 1) - 1)/stride[1] + 1
            #       = (25 + 2x0 - 1x(1-1) - 1)/1 + 1 ==> 25
            nn.Conv2d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=1, padding=0, stride=1),
            nn.ReLU(),
        )

        ## Layer Conv3: conv + relu  --> Reconstruction via averaging of overalpping high-res patches
        self.conv3 = nn.Sequential(
            ## batch size N=128 and input size (C_in=32, C_out=1, H_in=25, W_in=25): kerel_size=5, and stride=1, padding=0 (dilation=1)
            ## Out is determined by input and network structure
            # H_out = (H_in + 2*padding[0] - dilation[0]x(kernel_size[0] - 1) - 1)/stride[0] + 1
            #       = (25 + 2x0 - 1x(5-1) - 1)/1 + 1 ==> 21
            # W_out = (W_in + 2*padding[1] - dialation[1]x(kernel_size[1] - 1) - 1)/stride[1] + 1
            #       = 25 + 2x0 - 1x(5-1) - 1)/1 + 1 ==> 21
            nn.Conv2d(in_channels=conv2_out, out_channels=channels, kernel_size=5, padding=0, stride=1),
        )

    ## All the components of the SRCNN defined here!
    def forward(self, x):
        # 1st layer of 1x1 convolution
        # input patch size: 33x33
        # output patch size:
        x1 = self.conv1(x)

        # 2nd layer
        x2 = self.conv2(x1)

        # 3rd layer
        return self.conv3(x2)


# https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html
modelG = SRCNN(channels=1,
            conv1_out=64,
            conv2_out=32,
            dtype=torch.float64)

modelC = SRCNN(channels=1,
            conv1_out=64,
            conv2_out=32,
            dtype=torch.float64)

## Model will generate shift due to no padding
padding = 33 - 21
shift = padding // 2  ## integer result

## device important: so we save two versions: one based on CPU and one based on GPU
deviceG = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Name of the device = {torch.cuda.get_device_name(0)}')
deviceC = "cpu"

### Load in trained model wights saved by torch.save(model.state_dict(), PATH)
modelG.load_state_dict(torch.load("SRCNN_net.pth", weights_only=True))
modelG.eval()
modelG.to(deviceG)

modelC.load_state_dict(torch.load("SRCNN_net.pth", weights_only=True))
modelC.eval()
modelC.to(deviceC)

# Save the traced.scrippted model that can be easily loaded into C++
torch.save(modelC, "SRCNN_modelC.pth")
torch.save(modelG, "SRCNN_model.pth")

### Create a dummy CUDA tensor of the same shape as the actual input
# Actual input shape is (1, 1, 1083, 750) and is on CUDA (GPU)
dummy_inputG = torch.randn(1, 1, 1083, 750, device='cuda:0')

# Trace the model using the dummy input
dummy_outputG = modelG(dummy_inputG)
traced_modelG = torch.jit.trace(modelG, dummy_inputG)

# Save the traced.scrippted model that can be easily loaded into C++
traced_modelG.save("SRCNN_traced.pth")

# Convert to a ONNX model and save it
model_path = os.path.join(os.getcwd(), 'SRCNN.onnx')
torch.onnx.export(modelG, dummy_inputG, model_path, export_params=True, opset_version=14,
                  do_constant_folding=True, input_names=['input'], output_names=['output'])
# Print success message
print(f'Model saved to SRCNN.onnx with ONNX format.')


# Convert to a ONNX model and save it: let's make model dynamic to accept any input image size
onnx_model_path = os.path.join(os.getcwd(), 'SRCNN_dynamic.onnx')
torch.onnx.export(modelG, dummy_inputG, onnx_model_path, export_params=True, opset_version=14,
                  do_constant_folding=True, input_names=['input'], output_names=['output'],
                  dynamic_axes={
                    'input': {2: 'height', 3: 'width'},   # allow dynamic height & width
                    'output': {2: 'height', 3: 'width'}   # optional, if output shape depends on input
                })
# Print success message
print(f'Model saved to SRCNN_dynmic.onnx with ONNX format.')

## Finally convert to a TensorRT model from ONNX model: needs to handle dynamic input size
# ONNX files are high-level graph descriptions, while .engine file are mucu larger (6 times larger in this case)
# 1) are serialized binaries with precompiled kernels, optimized for a specific GPU.
# 2) If you enable dynamic input shapes, TensorRT saves multiple optimization profiles which increase file size.
# 3) Some temporary buffers may be embedded during engine serialization depending on config settings.
import tensorrt as trt

print("TensorRT Python file:", trt.__file__)
print("TensorRT Python version:", trt.__version__)  # --> version 10.7.0

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

# Load and parse ONNX model
with open("SRCNN_dynamic.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("Failed to parse the ONNX file")

# Create config and profile
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30); # 1GB workspace limit

profile = builder.create_optimization_profile()

input_name = network.get_input(0).name
# (min, opt, max) shapes
profile.set_shape(input_name,
                  min=(1, 1, 64, 64),
                  opt=(1, 1, 640, 480),
                  max=(1, 1, 1920, 1080))

config.add_optimization_profile(profile)

# Build serialized engine
serialized_engine = builder.build_serialized_network(network, config)
with open("SRCNN_dynamic.engine", 'wb') as f:
    f.write(serialized_engine)
print(f"Saved engine to: SRCNN_dynamic.engine")


### Create a dummy CPU tensor of the same shape as the actual input
dummy_inputC = torch.randn(1, 1, 1083, 750, device='cpu')
# Trace the model using the dummy input
dummy_outputC = modelC(dummy_inputC)
traced_modelC = torch.jit.trace(modelC, dummy_inputC)
# Save the traced.scrippted model that can be easily loaded into C++
traced_modelC.save("SRCNN_tracedC.pth")
