import os
from torch.utils.data import Dataset
import torch
import numpy as np


from backend.utils import read_image, read_saliency
from backend.image_processing import process

class MultiDataManager(Dataset):
    """ 
    Wraps a torch dataset around a list of images grouped by folder (each folder corresponds to a label)
    
    folders_paths specifies tuples of (path to folder containing images, path to folder containing saliencies)
    If include_label = True then each item will be a tuple of (label, (image, saliency))

    """

    def __init__(self, folders_paths, verbose, preprocess_parameter_map, include_label=False):

        self.verbose = verbose
        self.preprocess_parameter_map = preprocess_parameter_map
        self.include_label = include_label

        # get the paths to all images and saliencies under the specified folders
        all_images_paths = []
        all_saliencies_paths = []
        labels_to_indices = {}
        indices_to_labels = []
        for i,(images_path,saliencies_path) in enumerate(folders_paths):
            # get the names of all images in the folder
            names = os.listdir(images_path)
            names = [n for n in names if n != '.DS_Store'] # filter out .DS_Store files
            images_paths = [os.path.join(images_path, n) for n in names]
            saliencies_paths = [os.path.join(saliencies_path, n) for n in names]

            # build a dictionary mapping labels to the indices of their samples
            start_index = len(all_images_paths)
            all_images_paths.extend(images_paths)
            all_saliencies_paths.extend(saliencies_paths)
            end_index = len(all_images_paths)
            
            labels_to_indices[i] = np.arange(start_index,end_index)
            indices_to_labels.extend(np.repeat([i], end_index - start_index))

        self.all_images_paths = np.array(all_images_paths)
        self.all_saliencies_paths = np.array(all_saliencies_paths)
        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = np.array(indices_to_labels)
        
        if self.verbose:
            print("Init dataset")
            print("\t total of {} images.".format(self.all_images_paths.shape[0]))
            print(f"\t Classes are: {labels_to_indices}")
        

    def __len__(self):
        return self.all_images_paths.shape[0]

    def __getitem__(self, index):
        # set path
        img_path = self.all_images_paths[index]
        sal_path = self.all_saliencies_paths[index]

        # IMAGE
        img = read_image(img_path) # Needs to be able to take the shape and put it to saliency for generality (some models can be weird)
        img = np.transpose(img, (2, 0, 1)) / 255.0
        img = torch.FloatTensor(img)

        # SALIENCY
        sal_img = read_saliency(sal_path)
        sal_img = sal_img / 255.0
        sal_img = process(sal_img, self.preprocess_parameter_map) # Preprocessing training data on the fly!
        sal_img = torch.FloatTensor(sal_img)
        sal_img = torch.unsqueeze(sal_img, 0)

        return (self.indices_to_labels[index], (img, sal_img)) if self.include_label else (img, sal_img)

    # returns a dictionary mapping labels to the indices of their samples
    def get_labels_to_indices(self):
        return self.labels_to_indices
