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
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import time

def addArtifact(TypeArtifact,FileAsTensor):
    '''
    Args:

    - FileAsTensor(tensor): 4D tensor that includes the information of the original image.

    - TypeArtifact (string): it can be 'Motion', 'Ghosting', 'BiasField', 'Noise', 'Blur' and 'Spike'.
    
    '''
        
    if TypeArtifact=='Motion':
        rdm=random.randint(1,10) #random variable for the intensity of the artifact
        add_artifact=tio.RandomMotion(num_transforms=rdm, image_interpolation='linear') #creation of the mask of the artifact
    elif TypeArtifact=='Ghosting':
        add_artifact=tio.RandomGhosting(num_ghosts=(3,10),intensity=1.5,axes=(1,2))
    elif TypeArtifact=='BiasField':
        add_artifact=tio.RandomBiasField(coefficients=1)
    elif TypeArtifact=='Spike':
        add_artifact=tio.RandomSpike()
    elif TypeArtifact=='Noise':
        add_artifact=tio.RandomNoise(mean=(1,6),std=20)
    elif TypeArtifact=='Blur':
        add_artifact=tio.RandomBlur(std=(0,7,7))
    transformedImage=add_artifact(FileAsTensor)
    transArrayImage=np.array(transformedImage[0,0,:,:]) #simulated data is transformed into an array 
    transScalImage = (np.maximum(transArrayImage, 0) / transArrayImage.max()) * 255.0 #simulated data is rescaled
    transScalImage = np.uint8(np.round(transScalImage)) #simulated data is transformed into integers
    return transScalImage