'''
@author: MARÍA PEÑA FERNÁNDEZ
'''

import pathlib
import os
from pathlib import Path
import tqdm
from tqdm.notebook import tqdm_notebook
import pydicom
import torch
import random
from PIL import Image
import torchio as tio
import numpy as np
import artifacts 
import matplotlib.pyplot as plt
import torchvision
import multiprocessing
from functools import partial

def Counter(path,BodyPart):
    '''
    Args:

    - file (string): the original DICOM folder of the file.

    - BodyPart (string): it can be either 'brain' or 'knee'.
    
    '''
    
    folder=Path(path)
    count=0
    if BodyPart=='knee':
        for file in folder.glob('*/*/*/'):
            count2=len(os.listdir(file))
            count+=count2
    elif BodyPart=='brain':
        for file in folder.glob('*/'):
            count2=len(os.listdir(file))
            count+=count2
    return(count)

def FirstPreprocessing(file):
    '''
    Args:

    - file (string): the original DICOM folder of the file.
    
    '''
    file2=pydicom.read_file(file) #the file is read
    a=file2.pixel_array.shape #needed to create the tensor
    tensor2 = torch.ones(1,1,a[0],a[1]) #a tensor is created
    x = np.array(file2.pixel_array)#the data of the file is transformed into an array
    scaled_image = (np.maximum(x, 0) / x.max()) * 255.0  #the values are rescaled (now they go from 0 to 255)
    scaled_image = np.uint8(np.round(scaled_image)) #they are transformed to integers
    y = scaled_image.astype(np. float) #data is transformed into float in order to transform it into a tensor
    slice = torch.tensor(y) #data is now in a 2D tensor
    tensor2[0,0,:,:]=slice[:,:] #data is in a 4D tensor
    return tensor2,scaled_image

def ShowImages(ScaledImage,transScalImage):
    '''
    Args:

    - ScaledImage (tensor): 2D tensor that includes the information of the original image.

    - transScalImage (tensor): 2D tensor that includes the information of the transformed image.
    
    '''
    fig = plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow(ScaledImage, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(transScalImage, cmap='gray')

def SaveImages(file,ScaledImage,transScalImage,TypeArtifact):      
    '''
    Args:

    - file (string): the original DICOM folder of the file.

    - ScaledImage (tensor): 2D tensor that includes the information of the original image.

    - transScalImage (tensor): 2D tensor that includes the information of the transformed image.

    - TypeArtifact (string): it can be 'Motion', 'Ghosting', 'BiasField' and 'Spike'.
    
    '''
    image1 = Image.fromarray(ScaledImage) #the initial image is created
    new_file_name =(f"{file.stem}.jpg")
    image1.save(str(file.parent)+'/'+str(new_file_name)) #the initial image is saved
    image2 = Image.fromarray(transScalImage) #the transformed image is created
    #we will create a folder if it doesnt exist to save the new transformed images with the same structure as the original ones
    if os.path.isdir(TypeArtifact+'/'+str(file.parent)): 
        image2.save(TypeArtifact+'/'+str(file.parent)+'/'+str(new_file_name))
    else:
        os.makedirs(TypeArtifact+'/'+str(file.parent))
        image2.save(TypeArtifact+'/'+str(file.parent)+'/'+str(new_file_name))
     
    
def MultiProc(file,BodyPart,TypeArtifact,showImage,saveImage):
    '''
    Args:

    - file (string): the original DICOM folder of the file.

    - BodyPart (string): it can be either 'brain' or 'knee'.
    
    - TypeArtifact (string): it can be 'Motion', 'Ghosting', 'BiasField' and 'Spike'.

    - showImage (boolean): True if you want images to be shown, False if not.

    - saveImage (boolean): True if you want images to be saved, False if not.
    
    '''
    TensorImage,ScaledImage=FirstPreprocessing(file)
    transScalImage=artifacts.addArtifact(TypeArtifact,TensorImage)
    if showImage:
        ShowImages(ScaledImage,transScalImage)
    if saveImage:
        SaveImages(file,ScaledImage,transScalImage,TypeArtifact)
    
    
def Preprocessing(path,BodyPart,TypeArtifact,showImage,saveImage,multiprocess=False,num_process=multiprocessing.cpu_count()):
    '''
    Args:

    - path (string): the original DICOM folder.

    - BodyPart (string): it can be either 'brain' or 'knee'.

    - TypeArtifact (string): it can be 'Motion', 'Ghosting', 'BiasField' and 'Spike'.

    - showImage (boolean): True if you want images to be shown, False if not.

    - saveImage (boolean): True if you want images to be saved, False if not.

    - multiprocess (boolean): True if you want to use multiprocessing, False if not. Default value: False.

    - num_process (int): number of processors you want to use. Default value: multiprocessing.cpu_count()
    
    '''
    folder=Path(path)#the structure of folders for knee and brain is different
    if BodyPart=='brain':
        glob=folder.glob('*/*.dcm')
    elif BodyPart=='knee':
        glob=folder.glob('*/*/*/*.dcm')
    if multiprocess: #we will do it with more than one processors
        pool=multiprocessing.Pool(num_process)
        Multi=partial(MultiProc,BodyPart=BodyPart,TypeArtifact=TypeArtifact,showImage=showImage,saveImage=saveImage)
        pool.map(Multi,tqdm_notebook(glob,total=Counter(path,BodyPart),desc='Process bar'))
    else: #the process will be done with just one processor
        for file in tqdm_notebook(glob,total=Counter(path,BodyPart),desc='Process bar'):
            TensorImage,ScaledImage=FirstPreprocessing(file)
            transScalImage=artifacts.addArtifact(TypeArtifact,TensorImage)
            if showImage:
                ShowImages(ScaledImage,transScalImage)
            if saveImage:
                SaveImages(file,ScaledImage,transScalImage,TypeArtifact)






    
    
    
    