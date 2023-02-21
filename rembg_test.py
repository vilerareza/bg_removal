import os
import rembg
import cv2 as cv

test_images_path = 'images/test_set/'
result_path = 'images/test_result/'

file_names = os.listdir(test_images_path)

'''Remove set of images'''
for file in file_names:
    img = cv.imread(os.path.join(test_images_path, file))
    img_r = rembg.remove (img)
    cv.imwrite(f'{result_path}/{file}_original_model.png', img_r)