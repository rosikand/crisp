"""
File: data.py
------------------
Implement supervised dataset for classification baseline. 
"""


import torch
from torch.utils.data import Dataset
import os 
from PIL import Image
import pandas as pd
import numpy as np
import rsbox 
from rsbox import ml



class INaturalistClassification(Dataset):
    def __init__(self, csv_file_path, images_dir_path, label_map_path, image_resize=(256, 256), normalize=True):
        """
        Initialize the dataset by reading the csv file and creating a mapping from image name to label. 
        Args:
        - csv_file_path (string): path to csv file
        - images_dir_path (string): path to directory with all the images 
        """
        self.csv_file = csv_file_path
        self.images_dir = images_dir_path
        self.df = pd.read_csv(csv_file_path)

        # sift out non-research grade observations. 
        self.df = self.df[self.df["quality_grade"] == 'research']

        # sift out non-species level 
        self.df = self.df[self.df["rank"] == 'species']


        # load label map 
        with open(label_map_path, 'rb') as file:
            self.label_map = pickle.load(file)


        self.default_img_shape = (3, 224, 224)

        self.image_resize = image_resize  # can be None 
        self.normalize = normalize
        


    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.df)


    def __getitem__(self, idx):
        """
        Return the image and label at the given index. 
        """

        row_obj = self.df.iloc[idx]

        species_name = row_obj['name']
        
        assert species_name in self.label_map
        label = self.label_map[species_name]

        # img 
        extension_str = "png"
        img_name = row_obj['photo_id']
        suffix_path = img_name[:3] + '/' + img_name + "." + extension_str
        img_path = os.path.join(self.images_dir, suffix_path)


        try:
          image_array = ml.load_image(img_path, resize=self.image_resize, normalize=self.normalize)
        except:
          print(f"The current image path ({img_path}) does not point to a valid file...., using random tensor.")
          image_array = torch.randn(3, 256, 256)
        
        # tensorize 
        image_array = torch.tensor(image_array, dtype=torch.float)
        label = torch.tensor(label)

        return image_array, label
    