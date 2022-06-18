import torch
import matplotlib
import matplotlib.pyplot as plt




def plots(list_artifacts,list_models,metric,show_fig=True,save_fig=True):
    '''
    Args:
    
    type_artifact (list of strings): the elements can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'. They can be repeated if different models are considered
    
    list_models (list of arrays): the elements can be '915', '935', '515'..., i.e. the name of the models that have been trained
    
    metric (string): it can be 'MSE', 'SSIM' and 'PSNR'
    
    show_fig (boolean): it indicates whether you want to show the image or not. The default value is True
    
    save_fig (boolean): it indicates whether you want to save the image or not. The default value is True
    
    '''
    
    if metric == 'MSE':
        index = 0
        init_values = [0.0118, 0.008, 0.0029, 0.0055, 0.0098, 0.0020]
    elif metric == 'SSIM':
        index = 1
        init_values = [0.8703, 0.4750, 0.9157, 0.7940, 0.8652, 0.6377]
    elif metric == 'PSNR':
        index = 2
        init_values = [23.0799, 22.6673, 29.2714, 24.5657, 22.5171, 27.2354]    
        
    plt.figure(figsize=(15, 7))
    colors=['lawngreen','turquoise','mediumslateblue','grey','tomato','orange','red','blue','yellow','green']
    
    for i in range(len(list_artifacts)):
        if list_artifacts[i] == 'Blur':
            data0 = init_values[0]
        elif list_artifacts[i] == 'Spike':
            data0 = init_values[1]
        elif list_artifacts[i] == 'Motion':
            data0 = init_values[2]
        elif list_artifacts[i] == 'Ghosting':
            data0 = init_values[3]
        elif list_artifacts[i] == 'BiasField':
            data0 = init_values[4]
        elif list_artifacts[i] == 'Noise':
            data0 = init_values[5]
        data = torch.load(f'training_info/tasas20k_{list_artifacts[i]}_{list_models[i]}.pth')
        plt.plot([data0] + data[index], color = colors[i], label = f'{list_artifacts[i]} {list_models[i]}')
        
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric}')
    plt.legend(frameon=False)

    if save_fig:
        plt.savefig(f'{metric}_{list_artifacts}_{list_models}.jpg')
        
    if show_fig:
        plt.show()
        






