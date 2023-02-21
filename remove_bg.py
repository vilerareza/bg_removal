import torch
import os
import onnx
from onnx2torch import convert
import torchinfo
from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple


def normalize(img_PIL, mean, std, size):
    # PIL
    img_PIL = img_PIL.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    # np array
    #img = np.array(img_PIL)
    img = img_PIL / np.max(img_PIL)
    tmpImg = np.zeros((img.shape[0], img.shape[1], 3))
    #print (tmpImg.shape)
    tmpImg[:, :, 0] = (img[:, :, 0] - mean[0]) / std[0]
    tmpImg[:, :, 1] = (img[:, :, 1] - mean[1]) / std[1]
    tmpImg[:, :, 2] = (img[:, :, 2] - mean[2]) / std[2]
    tmpImg = tmpImg.transpose((2, 0, 1))
    #print (tmpImg.shape)
    return np.expand_dims(tmpImg, 0).astype(np.float32)

def post_process(img, mask, mask_thres=30):

    # Resize mask w and h to match to match the img
    mask = cv.resize(mask, (img.shape[1], img.shape[0]))
    # Mask thresholding for weak pixels
    mask[mask<mask_thres] = 0
    # Light blurring for edge smoothing
    #mask = cv.GaussianBlur(mask, (3,3), 0)
    # Expanding dimension
    mask = np.expand_dims(mask, axis=2)
    # Blending alpha to image
    img = np.concatenate([img, mask], 2)

    '''Colored background'''
    r = np.full(img.shape[:2], 255, dtype = np.uint8)
    r = np.expand_dims(r, axis=2)
    g = np.full(img.shape[:2], 0, dtype = np.uint8)
    g = np.expand_dims(g, axis=2)
    b = np.full(img.shape[:2], 0, dtype = np.uint8)
    b = np.expand_dims(b, axis=2)
    bg = np.concatenate([r, g, b], 2)

    background = Image.fromarray(bg)
    foreground = Image.fromarray(img)

    background.paste(foreground, (0, 0), foreground)

    #print (background.shape)
    new_img = np.array(background).astype(np.uint8)
    print (new_img.shape)
    return new_img, mask

    return img, mask

    '''Deprecated'''
    # foreground_threshold = 250
    # background_threshold = 1
    # erode_structure_size = 10

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # # Resize mask w and h to match to match the img
    # mask = cv.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv.INTER_LANCZOS4)
    # mask = cv.resize(mask, (img.shape[1], img.shape[0]))
    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    # mask = cv.GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=cv.BORDER_DEFAULT)
    # #mask = np.where(mask < 127, 0, 255).astype(np.uint8)  # convert again to binary

    # is_foreground = mask > foreground_threshold
    # #is_foreground = mask == 255
    # is_background = mask < background_threshold
    # #is_background = mask == 0 background_threshold

    # structure = None
    # if erode_structure_size > 0:
    #     structure = np.ones((erode_structure_size, erode_structure_size), dtype=np.uint8)

    # is_foreground = binary_erosion(is_foreground, structure=structure)
    # is_background = binary_erosion(is_background, structure=structure, border_value=1)

    # trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    # trimap[is_foreground] = 255
    # trimap[is_background] = 0

    # img_normalized = img / 255.0
    # trimap_normalized = trimap / 255.0

    # alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    # foreground = estimate_foreground_ml(img_normalized, alpha)
    # cutout = stack_images(foreground, alpha)

    # cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    # #cutout = Image.fromarray(cutout)

    # return cutout, alpha*255


    
    # mask = np.expand_dims(mask, axis=2)

    # return np.concatenate([img, mask], 2), mask

    # Create alpha channel
    # alpha = (np.ones((img_array.shape[0:2]))*255).astype(np.uint8)
    # Mask thresholding
    # mask[mask<mask_thres] = 0
    #erosion_size = 5
    #Options = [cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE]
    #element = cv.getStructuringElement(Options[1],(erosion_size, erosion_size))
    #mask = cv.erode(mask, element)
    #ret, mask = cv.threshold(mask, mask_thres, 255, cv.THRESH_BINARY)
    # mask = cv.GaussianBlur(mask, (3,3), 0)
    #mask[mask<20] = 0
    #mask[mask!=0] = 255
    #mask = cv.medianBlur(mask, 3)
    #ret, mask = cv.threshold(mask, 50, 255, cv.THRESH_BINARY)
    #mask = cv.distanceTransform(mask, cv.DIST_L2, 3)
    #mask[mask>25] = 255
    # Expanding dimension
    # mask = np.expand_dims(mask, axis=2)
    #return np.concatenate([img, mask], 2), mask


model_path = 'models/u2net.onnx'
checkpoints_path = 'training/checkpoints/'
test_images_path = 'images/test_set/'
result_path = 'images/test_result/'

# Model conversion
onnx_model = onnx.load(model_path)
rembg_model = convert(onnx_model)
rembg_model.load_state_dict(torch.load(f'{checkpoints_path}/model_check_210223_0.pth'))
rembg_model.eval()

# Input images
files = os.listdir(test_images_path)
for file in files:
    img_path = os.path.join(test_images_path, file)
    img_size = (320, 320)
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    img = Image.open(img_path) 
    img = normalize(img, img_mean, img_std, img_size) 
    in_data = torch.from_numpy(img)

    # Prediction
    output = rembg_model(in_data)
    mask = output[0][0]
    mask = torch.transpose(mask, 0, 2)

    # Getting result
    mask = mask.detach().cpu().numpy()
    
    ma = np.max(mask)
    mi = np.min(mask)
    mask = (mask - mi) / (ma - mi)

    mask = mask.transpose(1,0,2)
    mask = (mask*255).astype(np.uint8)

    img_array = cv.imread(img_path, 1)

    img_array, mask = post_process(img_array, mask)
    cv.imwrite(f'{result_path}/{os.path.splitext(file)[0]}.png', img_array)
    cv.imwrite(f'{result_path}/{os.path.splitext(file)[0]}_mask.png', mask)


'''Plotting'''
# fig,ax = plt.subplots(2) 
# ax[0].imshow(np.squeeze(img).transpose(1,2,0))
# ax[1].imshow(mask.transpose(1,0,2), cmap = 'gray', vmin = 0, vmax = 1)
#ax[1].imshow(mask, cmap = 'gray', vmin = 0, vmax = 1)
#ax[0,0].imshow(img_ori)
#ax[0,1].imshow(drawing, cmap = 'gray', vmin = 0, vmax = 255)
#ax[1,0].imshow(mask, cmap = 'gray', vmin = 0, vmax = 255)
#ax[1,1].imshow(img_real)
# plt.show()