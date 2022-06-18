'''
@author: MARÍA PEÑA FERNÁNDEZ
'''
import PIL
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision
from torchvision.transforms import transforms

def filters(model, artifact, filters_code, save_fig = True):
    
    '''
    Args:
    
    - model (object of the class ArtifactCNN): the model we want to get the filters of
    
    - artifact (string): the artifact of the corresponding model
    
    - filters_code (string): it indicates the name of the corresponding model
    
    - save_fig (boolean): it indicates whether you want to save the image or not. The default value is True
    
    Return: none
    
    '''
            
    model_weights =[]
    conv_layers = []
    model_children = list(model.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    
    number_filters = 4
    if filters_code == '16_8':
        number_filters = 1
    elif filters_code == '32_16':
        number_filters = 2
    elif filters_code == '128_64':
        number_filters = 8 

    model_weights = model_weights[0].squeeze(0)
    fig = plt.figure(figsize=(80, 20))
    for i in range(len(model_weights)):
        a = fig.add_subplot(number_filters,16, i+1)
        imgplot = plt.imshow(model_weights.cpu().detach().numpy()[i,0,:,:],'gray')
        a.axis("off")        
        
    if save_fig:
        plt.savefig(f'{filters_code}_filters_{artifact}.jpg', bbox_inches='tight')


        
        
        
        
def feature_maps(image_path, model, artifact, filters_code, save_fig = True):
    
    '''
    Args:
    
    - image_path (string): it is the path of the image we want
    
    - model (object of the class ArtifactCNN): the model we want to get the feature maps of
    
    - artifact (string): the artifact of the corresponding model
    
    - filters_code (string): it indicates the name of the corresponding model
    
    - save_fig (boolean): it indicates whether you want to save the image or not. The default value is True
    
    Return: none
    
    '''

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    model_weights =[]
    conv_layers = []
    model_children = list(model.children())
    counter = 0
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    
    number_filters = 8
    if filters_code == '16_8':
        number_filters = 2
    elif filters_code == '32_16':
        number_filters = 4
    elif filters_code == '128_64':
        number_filters = 16   
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    outputs = []
    names = []
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        for i in range(feature_map.shape[0]):
            processed.append(feature_map[i,:,:].data.cpu().numpy())
    fig = plt.figure(figsize=(20, 20))
    for i in range(len(processed[0:8*number_filters])):
        a = fig.add_subplot(number_filters,8, i+1)
        imgplot = plt.imshow(processed[i],'gray')
        plt.axis("off")
    
    if save_fig:
        plt.savefig(f'{filters_code}_maps_{artifact}.jpg', bbox_inches='tight')
