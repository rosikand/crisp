"""
File: datasets.py
------------------
Defines the PyTorch dataset classes.
"""


import torch
from torch.utils.data import Dataset
import os 
from PIL import Image
import pandas as pd
import numpy as np
import rsbox  # custom package I wrote to handle some common ML tasks such as loading images into (C, H, W) form. 
from rsbox import ml



class INaturalistClassification(Dataset):
    def __init__(self, csv_file_path, images_dir_path, data_split="train"):
        """
        Initialize the dataset by reading the csv file and creating a mapping from image name to label. 
        Args:
        - csv_file_path (string): path to csv file
        - images_dir_path (string): path to directory with all the images 
        """
        self.csv_file = csv_file_path
        self.images_dir = images_dir_path
        self.df = pd.read_csv(csv_file_path)
        self.data_split = data_split

        # TODO: sift out non-research grade observations. 
        # self.df = self.df[self.df['Quality_grade'] == 'research']

        # TODO: need to edit this to ensure species level classification labels. 

        # Create a mapping from image name to numeric label and add this as a column to the df 
        
        self.label_map = {}  
        self.unique_labels = self.df['name'].unique()  
        print(f"Initializing dataset... num unique labels is {len(self.unique_labels)}")
        for i, label in enumerate(self.unique_labels):
            self.label_map[label] = i  
            
        self.df['label'] = self.df['name'].map(self.label_map)

        self.default_img_shape = (3, 224, 224)
        

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return the image and label at the given index. 
        """
        
        # label 
        label = self.df.loc[idx, 'label']

        remote_sensing_img_name = str(self.df.loc[idx, 'remote_sensing'])
        rs_path = os.path.join(self.remote_sensing_dir, remote_sensing_img_name)

        # get uuid to search for in npz based on dataset
        uuid_npy = None
        if self.data_split == "train" or self.data_split == "validation":
          uuid_npy = str(self.df.loc[idx, 'photo_uuid'])
        elif self.data_split == "test":
          uuid_npy = str(self.df.loc[idx, 'observation_uuid'])
        else:
           raise Exception("invalid data split")
           

        try:
          rs_img_obj = np.load(rs_path)
          # rs_files = rs_img_obj.files
          if (uuid_npy in rs_img_obj):
            image_array = rs_img_obj[uuid_npy]
          else:
             image_array = rs_img_obj[rs_img_obj.files[0]]
          image_array = image_array[:3,:,:]  # (4, 256, 256) -- > (3, 256, 256)
        except:
          # generate random remote sensing image
          image_array = torch.randn(3, 256, 256)
          # raise Exception("remote sensing image not found")
        

        # tensorize 
        image_array = torch.tensor(image_array, dtype=torch.float32)
        label = torch.tensor(label)

        return image_array, label
    