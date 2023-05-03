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
        # self.df = self.df[self.df['Quality_grade'] == 'research']

        # TODO: need to edit this to ensure species level classification labels. 

        # Create a mapping from image name to numeric label and add this as a column to the df 
        
        self.label_map = {}  
        self.unique_labels = self.df['name'].unique()  
        for i, label in enumerate(self.unique_labels):
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

        # # if extension column is not present, then default to png? 
        # if 'extension' not in self.df.columns:
        #     extension_str = "png"
        #     print("(for debugging purposes) Warning: extension column not present in csv file. Defaulting to png")
        # else:
        #     extension_str = str(self.df.loc[idx, 'extension'])
        
        # temporay since it seems that all the images are png despite mismatching extension 
        extension_str = "png"

        suffix_path = img_name[:3] + '/' + img_name + "." + extension_str
        img_path = os.path.join(self.images_dir, suffix_path)
        
        try:
          image_array = ml.load_image(img_path, resize=None, normalize=True)
        except:
          print("The current image path does not point to a valid file...., skipping")
          return torch.tensor(-1), torch.tensor(-1)
          # raise Exception("The current image path does not point to a valid file")
        

        # tensorize 
        image_array = torch.tensor(image_array, dtype=torch.float32)
        label = torch.tensor(label)

        return image_array, label
    