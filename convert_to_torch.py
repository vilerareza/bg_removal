import torch
import onnx
from onnx2torch import convert
import torchinfo
from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple


def normalize(img, mean: Tuple[float, float, float], std: Tuple[float, float, float], size: Tuple[int, int],) -> Dict[str, np.ndarray]:
    # PIL
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    # ndarray
    #img = img[:,:,::-1]
    #img = cv.resize(img, size)
    img = img / np.max(img)

    tmpImg = np.zeros((img.shape[0], img.shape[1], 3))
    print (tmpImg.shape)
    tmpImg[:, :, 0] = (img[:, :, 0] - mean[0]) / std[0]
    tmpImg[:, :, 1] = (img[:, :, 1] - mean[1]) / std[1]
    tmpImg[:, :, 2] = (img[:, :, 2] - mean[2]) / std[2]
    tmpImg = tmpImg.transpose((2, 0, 1))
    print (tmpImg.shape)
    return np.expand_dims(tmpImg, 0).astype(np.float32)


# Model conversion
model_path = 'models/u2net.onnx'
onnx_model = onnx.load(model_path)
pytorch_model = convert(onnx_model)
pytorch_model.eval()
#torchinfo.summary(pytorch_model, input_size = (1, 3, 320, 320), col_names=['input_size', 'output_size', 'kernel_size'])

# Input image
#data = torch.randn(1, 3, 320, 320)
img_path = 'images/1_2.jpg'
img_size = (320, 320)
img_mean = (0.485, 0.456, 0.406)
img_std = (0.229, 0.224, 0.225)
img = Image.open(img_path) 
#img = cv.imread(img_path) 
img = normalize(img, img_mean, img_std, img_size) 
in_data = torch.from_numpy(img)
#in_data = torch.transpose(in_data, 2, 3)

print (in_data.size())

# Prediction
output = pytorch_model(in_data)
#outputarray = np.array(output)
#print (outputarray.shape)
mask = output[0][0]
print (mask.size())
mask = torch.transpose(mask, 0, 2)

# Getting result
mask = mask.detach().cpu().numpy()
#print (mask.shape)

#cv.imwrite('out/test_new.png', img_real)

# Plotting
fig,ax = plt.subplots(2) 
ax[0].imshow(np.squeeze(img).transpose(1,2,0))
ax[1].imshow(mask.transpose(1,0,2), cmap = 'gray', vmin = 0, vmax = 1)
#ax[1].imshow(mask, cmap = 'gray', vmin = 0, vmax = 1)
#ax[0,0].imshow(img_ori)
#ax[0,1].imshow(drawing, cmap = 'gray', vmin = 0, vmax = 255)
#ax[1,0].imshow(mask, cmap = 'gray', vmin = 0, vmax = 255)
#ax[1,1].imshow(img_real)

plt.show()