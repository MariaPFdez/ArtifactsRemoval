'''
@author: MARÍA PEÑA FERNÁNDEZ
'''

import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pathlib
from pathlib import Path
import shutil


def save_deartif_image(image_deartif, name_path):
    '''
    Args:

    - name_path (string): the path where we want to save our images.
    
    - image_deartif (tensor): the image we want to save.
    
    Return: none
    
    '''
    image_deartif = image_deartif.view(image_deartif.size(0), 1, 224, 224)
    save_image(image_deartif, name_path)
    
    
    

def create_folders(type_artifact, filters_code):
    '''
    Args:

    - type_artifact (string): it can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
    
    - filters_code (string): it represents the structure of the filters of the network. For example '915' or '515'.
    
    Return: none
    
    '''
    image_dir = f'Outputs'
    os.makedirs(image_dir, exist_ok=True)
    image_dir = f'training_info'
    os.makedirs(image_dir, exist_ok=True)
    image_dir = f'trained_models'
    os.makedirs(image_dir, exist_ok=True)
    image_dir2 = f'Outputs/images20k_{type_artifact}_{filters_code}_val'
    os.makedirs(image_dir2, exist_ok=True)
    
    
    
    
def obtain_data(type_artifact):
    '''
    Args:

    - type_artifact (string): it can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
     
    Return:
    
    - train_all (list of strings): list of the internal paths of the images that will be used to train (only includes the study and the particular image path)
    
    - val_all (list of strings): list of the internal paths of the image that will be used to validate (only includes the study and the particular image path)
    
    '''
    
    train_all=[]
    val_all=[]

    artifact_train = os.listdir(f'{type_artifact}/Training')
    artifact_train.sort()
    artifact_val = os.listdir(f'{type_artifact}/Validation')
    artifact_val.sort()

    for i in range(len(artifact_train)):
        x_train_files=os.listdir(f'{type_artifact}/Training/{artifact_train[i]}')
        for j,x_train_file in enumerate(x_train_files):
            x_train_file=str(artifact_train[i])+'/'+x_train_file
            x_train_files[j]=x_train_file
        train_all+=x_train_files

    for i in range(len(artifact_val)):
        x_val_files=os.listdir(f'{type_artifact}/Validation/{artifact_val[i]}')
        for j,x_val_file in enumerate(x_val_files):
            x_val_file=str(artifact_val[i])+'/'+x_val_file
            x_val_files[j]=x_val_file
        val_all+=x_val_files 
    
        
    return train_all,val_all


    
class ArtifactDataset(Dataset):
    '''
    Atts:

    - type_artifact (string): it can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
    
    - data_type (string): it can be 'Training' and 'Validation'
    
    - artif_paths (string): the path of the artifacts.
    
    - sharp_paths (string): the path of the corresponding sharp images.
    
    - transforms (object of method Compose (of transforms module)): the transformations that should be made to the images.
    
    '''
    def __init__(self, type_artifact, data_type, artif_paths, sharp_paths=None, transforms=None):      
        self.artif = artif_paths
        self.sharp = sharp_paths
        self.transforms = transforms
        self.ty = data_type
        self.art_ty = type_artifact
         
    def __len__(self):
        return (len(self.artif))
    
    def __getitem__(self, i):
        artif_image = cv2.imread(f"{self.art_ty}/{self.ty}/{self.artif[i]}",cv2.IMREAD_GRAYSCALE)
        
        if self.transforms:
            artif_image = self.transforms(artif_image)
            
        if self.sharp is not None:
            sharp_image = cv2.imread(f"fastMRI_brain_DICOM/{self.ty}/{self.sharp[i]}",cv2.IMREAD_GRAYSCALE)
            sharp_image = self.transforms(sharp_image)
            return (artif_image, sharp_image)
        else:
            return artif_image
    

def prepare_data(type_artifact,train_all,val_all,batch_size,transform):
    '''
    Args:
    
    - type_artifact (string): it can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
    
    - train_all (list of strings): list of the internal paths of the images that will be used to train (only includes the study and the particular image path)
    
    - val_all (list of strings): list of the internal paths of the image that will be used to validate (only includes the study and the particular image path)
    
    - batch_size (int): the batch size that we want for our model. 
    
    - transform (object of method Compose (of transforms module)): the transformations that should be made to the images.
    
    Return:
    
    - train_data (object of the class ArtifactDataset): it contains the images of both the images with artifacts and the sharp ones that we will use to train
    
    - val_data (object of the class ArtifactDataset): it contains the images of both the images with artifacts and the sharp ones that we will use to validate
    
    - trainloader (object of class DataLoader from torch.utils.data): it has the information of the training set that we introduce into the training method
    
    - valloader (object of class DataLoader from torch.utils.data): it has the information of the validation set that we introduce into the validate method
    
    '''
    train_data = ArtifactDataset(type_artifact,'Training',train_all, train_all, transform)
    val_data = ArtifactDataset(type_artifact,'Validation',val_all, val_all, transform)
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle = False)
    return train_data, val_data, trainloader, valloader
    

class ArtifactCNN(nn.Module):
    '''
    It has the structure of the net that we want to train.
    '''
    def __init__(self):
        super(ArtifactCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9,padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5,padding=2)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    

def optim_sche(model,learning_rate):
    '''
    Args:
    
    - model (object of class ArtifactCNN): it is the network that we want to train.
    
    - learning_rate (float): the learning rate hyperparameter of our model.
    
    
    Return:
    
    - criterion (object of class MSELoss from module numpy): it will compute the loss in the training process
    
    - optimizer (object of class Adam from module torch.optim): it holds the current state and updates the parameters based on the computed gradients.
    
    - scheduler (object of class ReduceLROnPlateau from module torch.optim.lr_scheduler): it reduces the learning rate when a metric has stopped improving. 
    
    '''    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=5, factor=0.5, verbose=True)    
    return criterion, optimizer, scheduler
  
    
    
def fit(model, dataloader, epoch, criterion, optimizer, device, batch_size):
    '''
    Args:
    
    - model (object of class ArtifactCNN): it is the network that we want to train.
    
    - dataloader (object of class DataLoader from torch.utils.data): it has the information of the dataset that we introduce into the fit function
    
    - epoch (int): the epoch the process is in
    
    - criterion (object of class MSELoss from module numpy): it will compute the loss in the training process
    
    - optimizer (object of class Adam from module torch.optim): it holds the current state and updates the parameters based on the computed gradients.
    
    - scheduler (object of class ReduceLROnPlateau from module torch.optim.lr_scheduler): it reduces the learning rate when a metric has stopped improving.
    
    - device (string): it is 'cpu' or 'cuda:0', depending on the processor being used
    
    - batch_size (int): the batch size that we want for our model.
    
    Return: 
    
    - train_loss (float): value of the mean of the (training) images loss in a certain epoch
    
    - train_ssim (float): value of the mean of the (training) images ssim in a certain epoch
    
    - train_psnr (float): value of the mean of the (training) images psnr in a certain epoch   
    
    - running_loss (list of floats): values of the all the (training) images loss in a certain epoch
    
    - running_ssim (list of floats): values of the all the (training) images ssim in a certain epoch
    
    - running_psnr (list of floats): values of the all the (training) images psnr in a certain epoch
    
    '''
    
    model.train()
    running_loss = []
    running_ssim = []
    running_psnr = []
    for i, data in enumerate(dataloader):
        blur_image = data[0]
        sharp_image = data[1]
        blur_image = blur_image.to(device)
        sharp_image = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)
        outputs=outputs.cpu().detach().numpy()
        sharp_image=sharp_image.cpu().detach().numpy()
        acc = ssim(outputs[0,0,:,:], sharp_image[0,0,:,:])
        acc2 = psnr(outputs[0,0,:,:], sharp_image[0,0,:,:])
        running_ssim.append(acc)
        running_psnr.append(acc2)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        running_loss.append(loss.item())
    train_loss = np.sum(running_loss)/len(dataloader)
    train_ssim = np.sum(running_ssim)/(len(dataloader))
    train_psnr = np.sum(running_psnr)/(len(dataloader))   
    return train_loss,train_ssim,train_psnr,running_loss,running_ssim,running_psnr



def validate(model, dataloader, epoch, type_artifact,criterion, optimizer, val_data, filters_code, device, batch_size):
    ''' 
    Args: all of them have been previously described
    
    Return:
    
    - val_loss (list of floats): values of the (validation) images loss during the 40 epochs
    
    - val_ssim (list of floats): values of the (validation) images ssim during the 40 epochs
    
    - val_psnr (list of floats): values of the (validation) images psnr during the 40 epochs
    
    - running_loss (list of floats): values of the all the (validation) images loss in a certain epoch
    
    - running_ssim (list of floats): values of the all the (validation) images ssim in a certain epoch
    
    - running_psnr (list of floats): values of the all the (validation) images psnr in a certain epoch
    
    '''
    
    model.eval()
    running_loss = []
    running_ssim = []
    running_psnr = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            blur_image = data[0]
            sharp_image = data[1]
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            outputs2=outputs.cpu().detach().numpy()
            out=np.zeros((224,224))
            sharp_image2=sharp_image.cpu().detach().numpy()
            inp=np.zeros((224,224))
            acc = ssim(outputs2[0,0,:,:], sharp_image2[0,0,:,:])
            acc2 = psnr(outputs2[0,0,:,:], sharp_image2[0,0,:,:])
            running_ssim.append(acc)
            running_psnr.append(acc2)  
            running_loss.append(loss.item())
            if epoch == 0 and i == int((len(val_data)/dataloader.batch_size)-1):
                save_deartif_image(sharp_image.cpu().data, f"Outputs/images20k_{type_artifact}_{filters_code}_val/sharp{epoch}.jpg")
                save_deartif_image(blur_image.cpu().data, f"Outputs/images20k_{type_artifact}_{filters_code}_val/artif{epoch}.jpg")
            if i == int((len(val_data)/dataloader.batch_size)-1):
                save_deartif_image(outputs.cpu().data, f"Outputs/images20k_{type_artifact}_{filters_code}_val/val_deartif{epoch}.jpg")
        val_loss = np.sum(running_loss)/len(dataloader)
        val_ssim = np.sum(running_ssim)/(len(dataloader))
        val_psnr = np.sum(running_psnr)/(len(dataloader))      
        return val_loss, val_ssim, val_psnr,running_loss,running_ssim,running_psnr
    

def training(model, trainloader, valloader, num_epochs, val_data, criterion, optimizer, scheduler, type_artifact, filters_code, device,batch_size):
    
    '''
    Args: all of them have been previously described
    
    Return: 
    
    model (object of class ArtifactCNN): it is the model already trained.
    
    train_loss (list of floats): values of the (training) images loss during the 40 epochs
    
    train_ssim (list of floats): values of the (training) images ssim during the 40 epochs 
    
    train_psnr (list of floats): values of the (training) images psnr during the 40 epochs
    
    val_loss (list of floats): values of the (validation) images loss during the 40 epochs
    
    val_ssim (list of floats): values of the (validation) images ssim during the 40 epochs
    
    val_psnr (list of floats): values of the (validation) images psnr during the 40 epochs
    
    '''
    train_loss  = []
    val_loss = []
    train_ssim = []
    val_ssim = []
    train_psnr = []
    val_psnr = []

    train_total_loss  = []
    val_total_loss = []
    train_total_ssim = []
    val_total_ssim = []
    train_total_psnr = []
    val_total_psnr = []

    start = time.time()
    for epoch in tqdm(range(num_epochs), total = num_epochs):
        train_epoch_loss,train_epoch_ssim,train_epoch_psnr,train_epoch_total_loss,train_epoch_total_ssim,train_epoch_total_psnr = fit(model, trainloader, epoch, criterion, optimizer, device,batch_size)
        val_epoch_loss,val_epoch_ssim,val_epoch_psnr,val_epoch_total_loss,val_epoch_total_ssim,val_epoch_total_psnr = validate(model, valloader, epoch, type_artifact,criterion, optimizer, val_data, filters_code, device,batch_size)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        train_ssim.append(train_epoch_ssim)
        val_ssim.append(val_epoch_ssim)
        train_psnr.append(train_epoch_psnr)
        val_psnr.append(val_epoch_psnr)
        train_total_loss += train_epoch_total_loss
        val_total_loss += val_epoch_total_loss
        train_total_ssim += train_epoch_total_ssim
        val_total_ssim += val_epoch_total_ssim
        train_total_psnr += train_epoch_total_psnr
        val_total_psnr += val_epoch_total_psnr  
        scheduler.step(val_epoch_loss)
    end = time.time()
    print(f"Took {((end-start)/60):.3f} minutes to train")   
    return model,train_loss,train_ssim,train_psnr,val_loss, val_ssim, val_psnr

    
def save_model(model, train_loss, train_ssim, train_psnr, val_loss, val_ssim, val_psnr, type_artifact, filters_code):
    
    '''
    Args: all of them have been previously described
    
    Return: none
    
    '''
    torch.save([train_loss, train_ssim, train_psnr,val_loss, val_ssim, val_psnr], f'training_info/tasas20k_{type_artifact}_{filters_code}.pth')
    torch.save(model, f'trained_models/modelo20k_{type_artifact}_{filters_code}.pth')
    
    
def erase_checkpoints(type_artifact):
    '''
    Args:

    - type_artifact (string): it can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
    
    Return: none
    
    '''
    folder=Path(f'{type_artifact}')
    for file in folder.glob('**/**/*'):
        if str(file)[-11:]=='checkpoints':
            shutil.rmtree(file,ignore_errors=True) 
    
    
def train_model(type_artifact,batch_size,learning_rate,num_epochs, filters_code):
    '''
    Args:

    - type_artifact (string): it can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
    
    - batch_size (int): the batch size that we want for our model.
    
    - learning_rate (float): the learning rate hyperparameter of our model.
    
    - num_epochs (int): the number of epoch that the training process will have
    
    - filters_code (string): it represents the structure of the filters of the network. For example '915' or '515'.
    
    Return:
    
    - model (object of class ArtifactCNN): it is the model already trained.
    
    '''
    os.chdir(os.path.expanduser('~/'))
    erase_checkpoints(type_artifact)
    create_folders(type_artifact, filters_code)
    print('Folders created')
    train_all, val_all = obtain_data(type_artifact)
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor()])
    train_data, val_data, trainloader, valloader = prepare_data(type_artifact,train_all,val_all,batch_size,transform)
    print('Data prepared')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'     
    model = ArtifactCNN().to(device)
    criterion, optimizer, scheduler = optim_sche(model,learning_rate)
    print('Training process is going to start')
    model,train_loss,train_ssim,train_psnr,val_loss, val_ssim, val_psnr = training(model, trainloader, valloader, num_epochs, val_data, criterion, optimizer, scheduler,type_artifact, filters_code, device,batch_size)
    print('Training process has finished')
    print('Saving...')
    save_model(model, train_loss, train_ssim, train_psnr,val_loss, val_ssim, val_psnr, type_artifact, filters_code)
    print('DONE')
    return model
    
    
    
    
    
    
    
    
