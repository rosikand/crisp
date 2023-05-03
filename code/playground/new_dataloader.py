"""
File: new_dataloader.py
------------------
Testing new (fixed) dataset class. 
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
    def __init__(self, csv_file_path, images_dir_path):
        """
        Initialize the dataset by reading the csv file and creating a mapping from image name to label. 
        Args:
        - csv_file_path (string): path to csv file
        - images_dir_path (string): path to directory with all the images 
        """
        self.csv_file = csv_file_path
        self.images_dir = images_dir_path
        self.df = pd.read_csv(csv_file_path)

        # TODO: sift out non-research grade observations. 

        # Create a mapping from image name to numeric label and add this as a column to the df 
        # TODO: need to edit this to ensure species level classification labels. 
        self.label_map = {}  
        unique_labels = self.df['name'].unique()  
        for i, label in enumerate(unique_labels):
            self.label_map[label] = i  
            
        self.df['label'] = self.df['name'].map(self.label_map)
        

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

        # image
        img_name = str(self.df.loc[idx, 'photo_id'])

        # if extension column is not present, then default to png? 
        if 'extension' not in self.df.columns:
            extension_str = "png"
        else:
            extension_str = str(self.df.loc[idx, 'extension'])
        suffix_path = img_name[:3] + '/' + img_name + "." + extension_str
        img_path = os.path.join(self.images_dir, suffix_path)
        
        try:
          image_array = ml.load_image(img_path, resize=None, normalize=True)
        except:
          raise Exception("The current image path does not point to a valid file")

        return image_array, label



# test 

csv_file_path = '../../data/inaturalist_12K/train_val/train_val.csv'
images_dir_path = '../../data/inaturalist_12K/train_val/images'

ds = INaturalistClassification(csv_file_path, images_dir_path)
dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

for i, (image, label) in enumerate(dataloader):
    pass
