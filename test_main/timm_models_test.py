'''
Pytorch Image Models (timm)

`timm` is a deep-learning library created by Ross Wightman and is a collection of SOTA
computer vision models, layers, utilities, optimizers, schedulers, data-loaders, augmentations
and also training/validating scripts with ability to reproduce ImageNet training results.
'''
import timm
import torch
import os

### First list all the models
# List all available models
all_models = timm.list_models()
print(len(all_models), all_models[:5])  # Shows total and first five

# List only models with pretrained weights
pretrained_models = timm.list_models(pretrained=True)
print(len(pretrained_models), pretrained_models[:5])

# List model with search pattern: ViT
vit_models = timm.list_models('vit*')
print(len(vit_models), vit_models[:5])


### Second download selected models and conversion
# Download a specific model with pretrained weights
model_name = 'vit_tiny_patch16_224.augreg_in21k'
model = timm.create_model(model_name, pretrained=True)
model.eval()  # Set the model to evaluation mode
# Print model architecture
print(model)

# Example input tensor
dummy_input = torch.randn(1, 3, 224, 224)

# Trace the model
traced_model = torch.jit.trace(model, dummy_input)
# Save the model to a traced.script file
model_path = os.path.join(os.getcwd(), f'{model_name}.pth')
# Save the traced.scrippted model that can be easily loaded into C++
traced_model.save(model_path)
print(f'Model saved to {model_path} with traced script format.')

# Convert to a ONNX model and save it
model_path = os.path.join(os.getcwd(), f'{model_name}.onnx')
torch.onnx.export(model, dummy_input, model_path, export_params=True, opset_version=14,
                  do_constant_folding=True, input_names=['input'], output_names=['output'])
# Print success message
print(f'Model saved to {model_path} with ONNX format.')
