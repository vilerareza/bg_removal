'''
Function to transform the tensors for training
'''
import cv2 as cv
import numpy as np
from PIL import Image

import torch

def transform(img: Image, mask, mean, std, size):
    '''Input image'''
    img = img.convert("RGB").resize(size, Image.LANCZOS)
    img = img / np.max(img)
    new_img = np.zeros((img.shape[0], img.shape[1], 3))
    new_img[:, :, 0] = (img[:, :, 0] - mean[0]) / std[0]
    new_img[:, :, 1] = (img[:, :, 1] - mean[1]) / std[1]
    new_img[:, :, 2] = (img[:, :, 2] - mean[2]) / std[2]
    new_img = new_img.transpose((2, 0, 1)).astype(np.float32)
    #new_img = np.expand_dims(new_img, 0).astype(np.float32)
    x = torch.from_numpy(new_img)
    '''Mask image'''
    mask = cv.resize(mask, size)
    mask = cv.normalize(mask, None, 0,1,cv.NORM_MINMAX, cv.CV_32F)
    mask = np.expand_dims(mask, 0).astype(np.float32)
    y = torch.from_numpy(mask)
    #y = y.type(torch.LongTensor)

    return x, y