from torch.utils.data import Dataset, DataLoader
import os
import cv2 as cv
from PIL import Image
import random
import numpy as np

class CarDataset(Dataset):

    img_size = (320, 320)
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    
    def __init__(self, dataset_path, transform_func, train=False):
        self.dataset_path = dataset_path
        self.img_names = os.listdir(os.path.join(dataset_path,'cars'))
        self.transform = transform_func
        self.train = train
        if self.train:
            self.img_names = self.img_names[:int(len(self.img_names) * 0.9)]
            #self.img_names = self.img_names[:20]
        else:
            self.img_names = self.img_names[int(len(self.img_names) * 0.9):]
            #self.img_names = self.img_names[10:15]

    def __len__(self):
        """Returns the dataset size"""
        return len(self.img_names)

    def __getitem__(self, index):
        '''Fetch the data'''
        image_name = self.img_names[index]
        img = Image.open(os.path.join(self.dataset_path,'cars', image_name))
        mask = cv.imread(os.path.join(self.dataset_path,'mask_car_no_win', f'{os.path.splitext(image_name)[0]}.png'), 0)
        x, y = self.transform(img, mask, self.img_mean, self.img_std, self.img_size)

        return x, y
