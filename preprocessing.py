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
    
    folder = Path(path)
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
    data_read=pydicom.read_file(file) #the file is read
    shapeImage=data_read.pixel_array.shape #needed to create the tensor
    data_tensor = torch.ones(1,1,shapeImage[0],shapeImage[1]) #a tensor is created
    data_array = np.array(data_read.pixel_array)#the data of the file is transformed into an array
    scaled_image = (np.maximum(data_array, 0) / data_array.max()) * 255.0  #the values are rescaled (now they go from 0 to 255)
    scaled_image = np.uint8(np.round(scaled_image)) #they are transformed to integers
    scaled_float_image = scaled_image.astype(np. float) #data is transformed into float in order to transform it into a tensor
    slice = torch.tensor(scaled_float_image) #data is now in a 2D tensor
    data_tensor[0,0,:,:]=slice[:,:] #data is in a 4D tensor
    return data_tensor,scaled_image

def ShowImages(ScaledImage,transScalImage):
    '''
    Args:

    - ScaledImage (tensor): 2D tensor that includes the information of the original image.

    - transScalImage (list of tensors): list of 2D tensors that includes the information of all the transformed images.
    
    '''
    fig = plt.figure(figsize=(15,15))
    length=len(transScalImage)
    plt.subplot(1, length+1, 1)
    plt.imshow(ScaledImage, cmap='gray')    
    for i in range(length):
        plt.subplot(1, length+1, i+2)
        plt.imshow(transScalImage[i], cmap='gray')

def SaveSharpImages(file,ScaledImage):      
    '''
    Args:

    - file (string): the original DICOM folder of the file.

    - ScaledImage (tensor): 2D tensor that includes the information of the original image.
    
    '''
    scaledImage = Image.fromarray(ScaledImage) #the initial image is created
    new_file_name =(f"{file.stem}.jpg")
    scaledImage.save(str(file.parent)+'/'+str(new_file_name)) #the initial image is saved
    
def SaveArtifImages(file,transScalImage,TypeArtifact):       
    '''
    Args:

    - file (string): the original DICOM folder of the file.

    - transScalImage (tensor): 2D tensor that includes the information of the transformed image.

    - TypeArtifact (string): it can be 'Motion', 'Ghosting', 'BiasField' and 'Spike'.
    
    '''
    transScaledImage = Image.fromarray(transScalImage) #the transformed image is created
    new_file_name =(f"{file.stem}.jpg")
    #we will create a folder if it doesnt exist to save the new transformed images with the same structure as the original ones
    if os.path.isdir(TypeArtifact+'/'+str(file.parent)): 
        transScaledImage.save(TypeArtifact+'/'+str(file.parent)+'/'+str(new_file_name))
    else:
        os.makedirs(TypeArtifact+'/'+str(file.parent))
        transScaledImage.save(TypeArtifact+'/'+str(file.parent)+'/'+str(new_file_name))
     
    
def MultiProc(file,BodyPart,TypeArtifactArray,showImage,saveImage):
    '''
    Args:

    - file (string): the original DICOM folder of the file.

    - BodyPart (string): it can be either 'brain' or 'knee'.
    
    - TypeArtifactArray (list of strings): the elements can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.

    - showImage (boolean): True if you want images to be shown, False if not.

    - saveImage (boolean): True if you want images to be saved, False if not.
    
    '''
    TensorImage,ScaledImage=FirstPreprocessing(file)
    artifScalImage=[]
    for i in range(len(TypeArtifactArray)):
        artifScalImage.append(artifacts.addArtifact(TypeArtifactArray[i],TensorImage))
    if showImage:
        ShowImages(ScaledImage,artifScalImage)
    if saveImage:
        SaveSharpImages(file,ScaledImage)
        for i in range(len(TypeArtifactArray)):
            SaveArtifImages(file,artifScalImage[i],TypeArtifactArray[i])
    
    
def Preprocessing(path,BodyPart,TypeArtifactArray,showImage,saveImage,multiprocess=False,num_process=multiprocessing.cpu_count()):
    '''
    Args:

    - path (string): the original DICOM folder.

    - BodyPart (string): it can be either 'brain' or 'knee'.

    - TypeArtifactArray (list of strings): the elements can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.

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
        Multi=partial(MultiProc,BodyPart=BodyPart,TypeArtifactArray=TypeArtifactArray,showImage=showImage,saveImage=saveImage)
        pool.map(Multi,tqdm_notebook(glob,total=Counter(path,BodyPart),desc='Process bar'))
    else: #the process will be done with just one processor
        for file in tqdm_notebook(glob,total=Counter(path,BodyPart),desc='Process bar'):
            TensorImage,ScaledImage=FirstPreprocessing(file)
            artifScalImage=[]
            for i in range(len(TypeArtifactArray)):
                artifScalImage.append(artifacts.addArtifact(TypeArtifactArray[i],TensorImage))
            if showImage:
                ShowImages(ScaledImage,artifScalImage)
            if saveImage:
                SaveSharpImages(file,ScaledImage)
                for i in range(len(TypeArtifactArray)):
                    SaveArtifImages(file,artifScalImage[i],TypeArtifactArray[i])






    
    
    
    