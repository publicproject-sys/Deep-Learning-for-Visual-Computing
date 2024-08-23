import pickle
from typing import Tuple
import numpy as np
import os

#from dlvc.datasets.dataset import  Subset, ClassificationDataset
#from dataset import  Subset, ClassificationDataset
from .dataset import Subset, ClassificationDataset

class CIFAR10Dataset(ClassificationDataset):
    '''
    Custom CIFAR-10 Dataset.
    '''

    def __init__(self, fdir: str, subset: Subset, transform=None):
        '''
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        '''

        self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.fdir = fdir
        self.subset = subset
        self.transform = transform
        
        ## TODO implement
        # See the CIFAR-10 website on how to load the data files
         # Check if the directory exists
        if not os.path.isdir(fdir):
            raise ValueError(f"{fdir} is not a valid directory.")
        
        # Determine which files to load based on the subset
        if subset.name == "TRAINING":
            file_paths = [os.path.join(fdir, f"data_batch_{i}") for i in range(1, 5)]
        elif subset.name == "VALIDATION":
            file_paths = [os.path.join(fdir, "data_batch_5")]
        elif subset.name == "TEST":
            file_paths = [os.path.join(fdir, "test_batch")]
        else:
            raise ValueError("Invalid subset. Expected 'training', 'validation', or 'test'.")
        
        self.data = []
        self.labels = []
        
        # Load the data from the files
        for file_path in file_paths:
            # if not os.path.isfile(file_path):
            #     raise ValueError(f"Missing file: {file_path}")
            
            with open(file_path, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                self.data.extend(batch[b'data'])
                self.labels.extend(batch[b'labels'])
        
        # Reshape the data to the correct format
        self.data = np.array(self.data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        # Apply transformations if provided
        if self.transform:
            self.data = [self.transform(image) for image in self.data]

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        ## TODO implement
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple:
        '''
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        '''
        ## TODO implement
        if idx >= len(self.data) or idx < 0:
            raise IndexError("Index out of bounds")
        image = self.data[idx]
        label = self.labels[idx]
        return image, label

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''
        ## TODO implement
        return len(self.classes)

