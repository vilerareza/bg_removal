'''
Create binary mask based on original image with removed background
Input: Original image with removed background (RGBA)
Output: Mask binary image
'''

import os
import cv2 as cv
import numpy as np
from skimage import morphology as morph
from matplotlib import pyplot as plt

# Directory containing original image with removed bg
dir_car_no_bg = 'images/studio/nobg/'
# Path to directory for resulted car mask image 
dir_car_mask = 'images/studio/mask_car/'
# Path to directory for resulted car mask (without window) image 
dir_car_mask_no_win = 'images/studio/mask_car_no_win/'

files = os.listdir(dir_car_no_bg)

def get_large_contours(cts, size):
    # Return contours with size larger than size. Ranked from smallest to largest
    area = []
    large_contours = []
    for ct in cts:
        a = cv.contourArea(ct)
        if a > size:
            area.append(a)
            large_contours.append(ct)
    return area, large_contours

for file in files:

    file_path = f'{dir_car_no_bg}/{file}'
    img = cv.imread(file_path, -1)
    
    # mask car
    mask_car = np.zeros(img.shape[0:2]).astype(np.uint8)
    mask_car[img[:,:,3]==255] = 255
    # edges = morph.area_opening(edges,300)
    cv.imwrite(f'{dir_car_mask}/{file}', mask_car)

    # mask car np window
    mask_car_no_win = np.zeros(img.shape[0:2]).astype(np.uint8)
    cts, _ = cv.findContours(mask_car, mode = cv.RETR_EXTERNAL, method = cv.CHAIN_APPROX_NONE)
    areas, lcts = get_large_contours(cts, 2000)
    mask_car_no_win = cv.fillPoly(mask_car_no_win, [lcts[0]], 255)
    cv.imwrite(f'{dir_car_mask_no_win}/{file}', mask_car_no_win)


'''Single file proof check'''
# file_path = 'images/dataset_test/nobg/64.png'
# img = cv.imread(file_path, -1)
# print (f'Shape: {img.shape}')

# # mask car
# mask_car = np.zeros(img.shape[0:2]).astype(np.uint8)
# mask_car[img[:,:,3]==255] = 255
# # mask car and window
# mask_car_window = np.zeros(img.shape[0:2])
# cts, _ = cv.findContours(mask_car, mode = cv.RETR_EXTERNAL, method = cv.CHAIN_APPROX_NONE)
# mask_car_window = cv.fillPoly(mask_car_window, [cts[0]], 255)

# fig,ax = plt.subplots(2) 
# ax[0].imshow(mask_car, cmap='gray', vmin=0, vmax=255)
# ax[1].imshow(mask_car_window, cmap='gray', vmin=0, vmax=255)
# #plt.imshow(mask_car, cmap='gray', vmin=0, vmax=255)
# plt.show()