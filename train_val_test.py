import shutil
from sklearn.model_selection import train_test_split
import os


def filesPath(path):
    '''
    Args:

    - path (string): the original DICOM folder.
    
    '''
    directory = os.listdir(path)
    directory.sort()
    data_paths = []
    for i in range(len(directory)):
        data_paths.append(directory[i])    
    return data_paths


def splitFiles(data_paths):
    '''
    Args:

    - data_paths (list of strings): it includes the identity number of the studies of MRI
    
    '''
    (train_total, test) = train_test_split(data_paths, test_size=0.2)
    (train,val) = train_test_split(train_total, test_size=0.25)
    return train,val,test


def completePaths(path,type_artifact_array,train,val,test):
    '''
    Args:

    - path (string): the original DICOM folder.

    - type_artifact_array (list of strings): the elements can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
    
    - train (list of strings): it includes the identity number of the studies that will be used to train
    
    - val (list of strings): it includes the identity number of the studies that will be used to validate
    
    - test (list of strings): it includes the identity number of the studies that will be used to test
    
    '''
    train_all = []
    val_all = []
    test_all = []

    for i in range(len(train)):
        train_files = os.listdir(f'{type_artifact_array[0]}/{path}/{train[i]}')
        for j,train_file in enumerate(train_files):
            train_file = str(train[i])+'/'+train_file
            train_files[j] = train_file
        train_all += train_files

    for i in range(len(val)):
        val_files = os.listdir(f'{type_artifact_array[0]}/{path}/{val[i]}')
        for j,val_file in enumerate(val_files):
            val_file = str(val[i])+'/'+val_file
            val_files[j] = val_file
        val_all += val_files


    for i in range(len(test)):
        test_files = os.listdir(f'{type_artifact_array[0]}/{path}/{test[i]}')
        for j,test_file in enumerate(test_files):
            test_file = str(test[i])+'/'+test_file
            test_files[j] = test_file
        test_all += test_files
    
    return train_all,val_all,test_all



def createFiles(path,type_artifact_array,train,val,test):
    '''
    Args:

    - path (string): the original DICOM folder.

    - type_artifact_array (list of strings): the elements can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
    
    - train (list of strings): it includes the identity number of the studies that will be used to train
    
    - val (list of strings): it includes the identity number of the studies that will be used to validate
    
    - test (list of strings): it includes the identity number of the studies that will be used to test
    
    '''
    os.makedirs(f'{path}/Training', exist_ok=True)
    os.makedirs(f'{path}/Test', exist_ok=True)
    os.makedirs(f'{path}/Validation', exist_ok=True)
    
    for i,artifact in enumerate(type_artifact_array):
        os.makedirs(f'{artifact}/Training', exist_ok=True)
        os.makedirs(f'{artifact}/Test', exist_ok=True)
        os.makedirs(f'{artifact}/Validation', exist_ok=True)
    
    for i,data in enumerate(train):
        os.makedirs(f'{path}/Training/{data}')
        for i,artifact in enumerate(type_artifact_array):
            os.makedirs(f'{artifact}/Training/{data}')
            
    for i,data in enumerate(val):
        os.makedirs(f'{path}/Validation/{data}')
        for i,artifact in enumerate(type_artifact_array):
            os.makedirs(f'{artifact}/Validation/{data}')     
            
    for i,data in enumerate(test):
        os.makedirs(f'{path}/Test/{data}')
        for i,artifact in enumerate(type_artifact_array):
            os.makedirs(f'{artifact}/Test/{data}')

    
def divideData(path,type_artifact_array,train_all,val_all,test_all):
    '''
    Args:

    - path (string): the original DICOM folder.

    - type_artifact_array (list of strings): the elements can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
    
    - train_all (list of strings): it contains the name of each training file, including the identity number of the study
    
    - val_all (list of strings): it contains the name of each validation file, including the identity number of the study
    
    - test_all (list of strings): it contains the name of each test file, including the identity number of the study
    
    '''
    for i,data in enumerate(train_all):
        source = f'{path}/{data}'
        destination = f'{path}/Training/{data}'
        shutil.move(source,destination)
        for i,artifact in enumerate(type_artifact_array):
            source_art = f'{artifact}/{path}/{data}'
            destination_art = f'{artifact}/Training/{data}'
            shutil.move(source_art,destination_art)

    for i,data in enumerate(test_all):
        source = f'{path}/{data}'
        destination = f'{path}/Test/{data}'
        shutil.move(source,destination)
        for i,artifact in enumerate(type_artifact_array):
            source_art = f'{artifact}/{path}/{data}'
            destination_art = f'{artifact}/Test/{data}'
            shutil.move(source_art,destination_art)
            

    for i,data in enumerate(val_all):
        source = f'{path}/{data}'
        destination = f'{path}/Validation/{data}'
        shutil.move(source,destination)
        for i,artifact in enumerate(type_artifact_array):
            source_art = f'{artifact}/{path}/{data}'
            destination_art = f'{artifact}/Validation/{data}'
            shutil.move(source_art,destination_art)
            
            
def eraseEmptyFolders(path,type_artifact_array):
    '''
    Args:

    - path (string): the original DICOM folder.

    - type_artifact_array (list of strings): the elements can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
    
    '''
    for i,artifact in enumerate(type_artifact_array):
        shutil.rmtree(f'{artifact}/{path}',ignore_errors=True)            
            
            
def splitTrainValTest(path,type_artifact_array):
    '''
    Args:

    - path (string): the original DICOM folder.

    - type_artifact_array (list of strings): the elements can be 'Motion', 'Ghosting', 'BiasField', 'Blur', 'Noise' and 'Spike'.
    
    '''
    data_paths = filesPath(path)
    train,val,test = splitFiles(data_paths)
    train_list,val_list,test_list = completePaths(path,type_artifact_array,train,val,test)
    createFiles(path,type_artifact_array,train,val,test)
    divideData(path,type_artifact_array,train_list,val_list,test_list)
    eraseEmptyFolders(path,type_artifact_array)