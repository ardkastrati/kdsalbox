import os
from torch.utils.data import Dataset
import torch
import numpy as np

from .utils import get_image_path_tuples, read_image, read_saliency
from .image_processing import process

############################################################
# Train Dataset Manager
############################################################
class TrainDataManager(Dataset):

    def __init__(self, input_images, input_saliencies, verbose, preprocess_parameter_map, N=None):

        self.verbose = verbose
        self.path_images = input_images #os.path.join(input_dir, 'Images', mode)
        self.path_saliency = input_saliencies
        self.preprocess_parameter_map = preprocess_parameter_map

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names if n != '.DS_Store'])
        self.list_names = list_names

        if N is not None:
            self.list_names = list_names[:N]
        
        if self.verbose:
            print("Init dataset")
            print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        # set path
        
        ima_name = self.list_names[index]+'.jpg'
        img_path = os.path.join(self.path_images, ima_name)

        ima_name = self.list_names[index]+'.jpg'
        sal_path = os.path.join(self.path_saliency, ima_name)

        # IMAGE
        img = read_image(img_path) # Needs to be able to take the shape and put it to saliency for generality (some models can be weird)

        if img is None:
            ima_name = self.list_names[index + 1]+'.jpg'
            img_path = os.path.join(self.path_images, ima_name)

            ima_name = self.list_names[index + 1]+'.jpg'
            sal_path = os.path.join(self.path_saliency, ima_name)
            img = read_image(img_path)

        img = np.transpose(img, (2, 0, 1)) / 255.0
        img = torch.FloatTensor(img)

        # SALIENCY
        sal_img = read_saliency(sal_path)
        sal_img = process(sal_img, self.preprocess_parameter_map) # Preprocessing training data on the fly!
        sal_img = torch.FloatTensor(sal_img)
        sal_img = torch.unsqueeze(sal_img, 0)

        return (img, sal_img)

############################################################
# Test Dataset Manager
############################################################
class TestDataManager(Dataset):

    def __init__(self, input_images, input_saliencies, verbose, preprocess_parameter_map, N=None):

        self.verbose = verbose
        self.path_images = input_images #os.path.join(input_dir, 'Images', mode)
        self.path_saliency = input_saliencies
        self.preprocess_parameter_map = preprocess_parameter_map

        # get list images
        list_names = os.listdir(self.path_images)
        list_names = np.array([n.split('.')[0] for n in list_names if n != '.DS_Store'])
        self.list_names = list_names

        if N is not None:
                self.list_names = list_names[:N]
        
        if self.verbose:
            print("Init dataset")
            print("\t total of {} images.".format(self.list_names.shape[0]))

    def __len__(self):
        return self.list_names.shape[0]

    def __getitem__(self, index):
        # set path
        ima_name = self.list_names[index]+'.jpg'
        img_path = os.path.join(self.path_images, ima_name)

        ima_name = self.list_names[index]+'.jpg'
        sal_path = os.path.join(self.path_saliency, ima_name)

        # IMAGE
        img = read_image(img_path) # Needs to be able to take the shape and put it to saliency for generality (some models can be weird)
        if img is None:
            ima_name = self.list_names[index + 1]+'.jpg'
            img_path = os.path.join(self.path_images, ima_name)

            ima_name = self.list_names[index + 1]+'.jpg'
            sal_path = os.path.join(self.path_saliency, ima_name)
            img = read_image(img_path)

        img = np.transpose(img, (2, 0, 1)) / 255.0
        img = torch.FloatTensor(img)

        # SALIENCY
        sal_img = read_saliency(sal_path)
        sal_img = process(sal_img, self.preprocess_parameter_map) # Preprocessing testing data on the fly!
        sal_img = torch.FloatTensor(sal_img)
        sal_img = torch.unsqueeze(sal_img, 0)

        return (img, sal_img, ima_name)


############################################################
# Run Dataset Manager
############################################################
class RunDataManager(Dataset):
    """
        Data manager for Run
    """
    def __init__(self, input_dir, output_dir, verbose=False, recursive=False, N=None):

        self.verbose = verbose
        self.recursive = recursive
        
        self.image_path_tuples = get_image_path_tuples(input_dir, output_dir, recursive=self.recursive)
        self.num_paths = len(self.image_path_tuples)

        if self.verbose:
            print("Init dataset in mode run")
            print("\t total of {} images.".format(self.num_paths))
    
    def __len__(self):
        return self.num_paths

    def __getitem__(self, index):
        input_path = self.image_path_tuples[index][0]
        output_path = self.image_path_tuples[index][1]

        # IMAGE
        img = read_image(input_path)
        if img is None:
            input_path = self.image_path_tuples[index+1][0]
            output_path = self.image_path_tuples[index+1][1]
            img = read_image(input_path)
        img = torch.FloatTensor(img).permute(2, 0, 1) / 255.0

        return img, input_path, output_path