import torch
import torch.nn as nn
import onnx
from onnx2torch import convert

dataset_path = 'images/dataset_test/'
result_path = 'training/results/'
checkpoints_path = 'training/checkpoints/'
model_path = 'models/u2net.onnx'

# Model conversion and loading
onnx_model = onnx.load(model_path)
rembg_model = convert(onnx_model)
rembg_model.load_state_dict(torch.load(f'{checkpoints_path}/model_check_1.pth'))
rembg_model.eval()

dynamic_axes_dict = {'actual_input': [0, 2, 3], 'Output': [0]} 

tensor_sample = torch.randn(1, 3, 320, 320, requires_grad = True)

# Exporting the model to onnx
torch.onnx.export(rembg_model, tensor_sample, f'{result_path}/model.onnx', export_params=True, opset_version=11, do_constant_folding=True, input_names = ['input'], output_names = ['output'], dynamic_axes = dynamic_axes_dict)
