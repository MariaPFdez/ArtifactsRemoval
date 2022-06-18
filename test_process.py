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
import torchmetrics
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure


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
    
    - type_artifact (string): the artifact of the corresponding model
    
    - filters_code (string): it represents the structure of the filters of the network. For example '915' or '515'.
    
    Return: none
    
    '''
    image_dir = f'Outputs'
    os.makedirs(image_dir, exist_ok=True)
    image_dir2 = f'Outputs/images20k_{type_artifact}_{filters_code}_test'
    os.makedirs(image_dir2, exist_ok=True)
    
    
def obtain_data(type_artifact):
    '''
    Args:
    
    - type_artifact (string): the artifact of the corresponding model
    
    Return:
    
    - test_all (list of strings): list of the internal paths of the images that will be used to test (only includes the study and the particular image path)
    
    '''
    test_all=[]

    artifact_test = os.listdir(f'{type_artifact}/Test')
    artifact_test.sort()

    for i in range(len(artifact_test)):
        x_test_files=os.listdir(f'{type_artifact}/Test/{artifact_test[i]}')
        for j,x_test_file in enumerate(x_test_files):
            x_test_file=str(artifact_test[i])+'/'+x_test_file
            x_test_files[j]=x_test_file
        test_all+=x_test_files
              
    return test_all


    
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
    

def prepare_data(type_artifact,test_all,transform):
    '''
    Args:
    
    - type_artifact (string): the artifact of the corresponding model
    
    - test_all (list of strings): list of the internal paths of the images that will be used to test (only includes the study and the particular image path)
    
    - transform (object of method Compose (of transforms module)): the transformations that should be made to the images.
    
    Return:
    
    - test_data (object of the class ArtifactDataset): it contains the images of both the images with artifacts and the sharp ones that we will use to test
    
     - testloader (object of class DataLoader from torch.utils.data): it has the information of the test dataset that we introduce into the test function
    
    ''' 
    
    test_data = ArtifactDataset(type_artifact,'Test',test_all, test_all, transform)
    testloader = DataLoader(test_data, batch_size=1, shuffle = False)
    return test_data, testloader
    


def images_to_save(type_artifact):
    '''
    Args:
    
    - type_artifact (string): the artifact of the corresponding model
    
    Return:
    
    - num_images (list of integers): it includes indexes of the images that will be saved
    
    '''
    if type_artifact == 'Noise':
        num_images = (506, 568, 83, 61, 631, 398)
    elif type_artifact == 'Blur':
        num_images = (523, 668, 291, 657, 4, 27)
    elif type_artifact == 'BiasField':
        num_images = (563, 17, 6, 221, 611, 584)
    elif type_artifact == 'Motion':
        num_images = (346, 54, 485, 164, 540, 366)      
    elif type_artifact == 'Spike':
        num_images = (643, 7, 67, 1, 38, 29)        
    elif type_artifact == 'Ghosting':
        num_images = (13, 33, 354, 10, 679, 2)
    return num_images
        
        
        
        
def test(model, test_data, dataloader, num_images, device, type_artifact, filters_code):
    
    '''
    Args:
    
    - model (object of the class ArtifactCNN): the model we want to test
    
    - test_data (object of the class ArtifactDataset): it contains the images of both the images with artifacts and the sharp ones that we will use to test
    
    - dataloader (object of class DataLoader from torch.utils.data): it has the information of the test dataset that we introduce into the function
    
    - num_images (list of integers): it includes indexes of the images that will be saved
    
    - device (string): it is 'cpu' or 'cuda:0', depending on the processor being used
    
    - type_artifact (string): it can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'
    
    - filters_code (string): it represents the structure of the filters of the network. For example '915' or '515'.
    
    Return:
    
    - test_loss (float): value of the mean of the (test) images loss
    
    - test_ssim (float): value of the mean of the (test) images ssim
    
    - test_psnr (float): value of the mean of the (test) images psnr
    
    - test_ms_ssim (float): value of the mean of the (test) images ms-ssim
          
    '''
    
    running_loss = []
    running_ssim = []
    running_psnr = []
    running_ms_ssim = []
    criterion = nn.MSELoss()
    ms_ssim_obj = MultiScaleStructuralSimilarityIndexMeasure()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(test_data))):
            blur_image = data[0]
            sharp_image = data[1]
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            ms_ssim = ms_ssim_obj(outputs,sharp_image).item()
            loss=criterion(sharp_image,outputs)
            outputs2=outputs.cpu().detach().numpy()
            sharp_image2=sharp_image.cpu().detach().numpy()
            acc = ssim(sharp_image2[0,0,:,:], outputs2[0,0,:,:])
            acc2 = psnr(sharp_image2[0,0,:,:], outputs2[0,0,:,:])
            running_loss.append(loss.item())
            running_ssim.append(acc)
            running_psnr.append(acc2)
            j=0
            if str(ms_ssim) != 'nan':
                j += 1
                running_ms_ssim.append(ms_ssim)
            if i in num_images:
                save_deartif_image(sharp_image.cpu().data, f"Outputs/images20k_{type_artifact}_{filters_code}_test/sharp{i}.jpg")
                save_deartif_image(blur_image.cpu().data, f"Outputs/images20k_{type_artifact}_{filters_code}_test/artif{i}.jpg")
                save_deartif_image(outputs.cpu().data, f"Outputs/images20k_{type_artifact}_{filters_code}_test/test_deartif{i}.jpg")
        test_loss = np.sum(running_loss)/len(dataloader)
        test_ssim = np.sum(running_ssim)/len(dataloader)
        test_psnr = np.sum(running_psnr)/len(dataloader)
        test_ms_ssim = np.sum(running_ms_ssim)/(len(dataloader)-j)
    return test_loss,test_ssim,test_psnr,test_ms_ssim
    
    
def erase_checkpoints(type_artifact):
    
    '''
    Args:
    
    - type_artifact (string): the artifact of the corresponding model
    
    Return: none
    
    '''
    folder=Path(f'{type_artifact}')
    for file in folder.glob('**/**/*'):
        if str(file)[-11:]=='checkpoints':
            shutil.rmtree(file,ignore_errors=True) 
    
    
def test_model(type_artifact, model, filters_code):
    
    '''
    Args:

    - model (object of the class ArtifactCNN): the model we want to get the metrics of
    
    - type_artifact (string): the artifact of the corresponding model
    
    - filters_code (string): it indicates the name of the corresponding model    
    
    Return:
    
    - test_loss (float): value of the mean of the (test) images loss
    
    - test_ssim (float): value of the mean of the (test) images ssim
    
    - test_psnr (float): value of the mean of the (test) images psnr
    
    - test_ms_ssim (float): value of the mean of the (test) images ms-ssim
    
    '''
    os.chdir(os.path.expanduser('~/'))
    erase_checkpoints(type_artifact)
    create_folders(type_artifact, filters_code)
    print('Folders created')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    test_all = obtain_data(type_artifact)
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor()])
    test_data, testloader = prepare_data(type_artifact,test_all,transform)
    print('Data is prepared')  
    num_images = images_to_save(type_artifact)
    print('Test process is going to start')
    test_loss,test_ssim,test_psnr,test_ms_ssim = test(model, test_data, testloader, num_images, device, type_artifact, filters_code)
    print('Test process has finished')
    return test_loss,test_ssim,test_psnr,test_ms_ssim
    
    
    
    
    
    
    
