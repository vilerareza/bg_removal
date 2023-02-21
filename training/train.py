'''
Retrain the existing u2net.onnx
'''

import time
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torcheval.metrics import BinaryAccuracy
import torchinfo
import onnx
from onnx2torch import convert

from transform import transform
from dataset import CarDataset

# Number of training epochs
n_epochs = 1
# Training batch size
batch_size = 5
# Validation batch size
batch_size_val = 1
# Path to dataset folder
dataset_path = '../images/dataset_test/'
# Path to test image for evaluating model performance on every epoch
test_img = 'path/to/test/image.jpg'
# Path to folder for saving the inference test result using test iage
result_path = 'result/'
# Path to folder for saving the model chackpoints during training
checkpoints_path = 'checkpoints/'
# Path to origin u2net onnx model
model_path = '../models/u2net.onnx'
# Loss function
bce_loss = nn.BCELoss(size_average=True)


def train_loop(epoch, n_epochs, data_loader, model, loss_func, optimizer):
    '''Training loop to perform on every epoch'''
    # n of images in dataset
    size = len(data_loader.dataset)

    for batch, (x, y) in enumerate(data_loader):
        
        # Batch start time
        t_start = time.time()

        # Tensor preparation
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        if torch.cuda.is_available():
            x_var, y_var = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
        else:
            x_var, y_var = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
        
        # Prediction and loss
        d0, d1, d2, d3, d4, d5, d6= model(x_var)
        loss2, loss = loss_func(d0, d1, d2, d3, d4, d5, d6, y_var)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Batch end time
        t_end = time.time()

        loss = loss.item()
        current = (batch+1)*len(x)
        print (f'Epoch: {epoch+1}/{n_epochs}, Batch: {batch+1}/{int(size/len(x))}, Loss: {loss}, images: {current}/{size}, time elapsed: {int(t_end-t_start)}s')

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
    
    return model


def test_loop(data_loader, model, loss_func, accuracy_metric):

    size = len(data_loader.dataset)
    n_batches = len(data_loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            # wrap them in Variable
            if torch.cuda.is_available():
                x_var, y_var = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
            else:
                x_var, y_var = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
        
            # Prediction and loss
            d0, d1, d2, d3, d4, d5, d6= model(x_var)
            loss2, loss = loss_func(d0, d1, d2, d3, d4, d5, d6, y_var)

            accuracy_metric.update(torch.flatten(d0), torch.flatten(y_var))
            acc = accuracy_metric.compute()

            test_loss += loss

    test_loss /= n_batches
    correct /= size
    print(f'Accuracy: {acc}%, Avg loss: {test_loss:>8f}')


def test_inference(img_path, model, result_path):

    def __normalize(img, mean, std, size):
        # PIL
        img = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        img = img / np.max(img)
        tmpImg = np.zeros((img.shape[0], img.shape[1], 3))
        tmpImg[:, :, 0] = (img[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (img[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (img[:, :, 2] - mean[2]) / std[2]
        tmpImg = tmpImg.transpose((2, 0, 1))
        return np.expand_dims(tmpImg, 0).astype(np.float32)

    img_path = img_path
    img_size = (320, 320)
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    img = Image.open(img_path) 
    img = __normalize(img, img_mean, img_std, img_size) 
    in_data = torch.from_numpy(img)

    # Prediction
    model.eval()
    output = model(in_data)
    mask = output[0][0]
    mask = torch.transpose(mask, 0, 2)

    # Getting result
    mask = (mask.detach().cpu().numpy())*255
    mask = mask.transpose(1,0,2)
    cv.imwrite(result_path, mask)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    #print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))
    
    return loss0, loss

def start_train():

    # Model conversion and loading
    onnx_model = onnx.load(model_path)
    rembg_model = convert(onnx_model)
    rembg_model.load_state_dict(torch.load('checkpoints/u2net_state_160223.pth'))
    # Determine the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    rembg_model = rembg_model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(rembg_model.parameters(), lr=1e-5)
    
    # Metric
    criterion = muti_bce_loss_fusion
    accuracyMetric = BinaryAccuracy()

    data = CarDataset(dataset_path, transform, train=True)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    valid_data = CarDataset(dataset_path, transform, train=False)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    '''Data loader verification'''
    #imgs, masks = next(iter(train_loader))
    # print(f"Img batch shape: {imgs.size()}")
    # print(f"Mask batch shape: {masks.size()}")
    # img = imgs[0].squeeze()
    # #img = torch.transpose(img, 0, 2)
    # img = img.detach().cpu().numpy()
    # img = img.transpose(1,2,0)
    # label = masks[0].squeeze()
    # label = label.detach().cpu().numpy()
    # print (img.shape)
    # fig,ax = plt.subplots(2) 
    # ax[0].imshow(img)
    # ax[1].imshow(label, cmap='gray', vmin=0, vmax=1)
    # #plt.imshow(img)
    # plt.show()

    rembg_model.train()

    for epoch in range(n_epochs):
        # Training
        rembg_model = train_loop(epoch, n_epochs, train_loader, rembg_model, criterion, optimizer)
        # Validation
        test_loop(valid_loader, rembg_model, criterion, accuracyMetric)
        # Save the checkpoint
        torch.save(rembg_model.state_dict(), f'{checkpoints_path}/model_check_210223_{epoch}.pth')
        rembg_model.train()  # resume train
        # Inference test
        # test_inference(test_img, rembg_model, f'results/test_{epoch}.jpg')

if __name__ == "__main__":
    start_train()