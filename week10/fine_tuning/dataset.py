"""
File: datasets.py
------------------
Defines the PyTorch dataset class for fine-tuning. 
"""


import torch
from torch.utils.data import Dataset
import os 
from PIL import Image
import pandas as pd
import numpy as np
import rsbox 
from rsbox import ml
import pdb
import os
import json 


class FineTuningCrispDataset(Dataset):
    def __init__(self, csv_file_path, images_dir_path, remote_sensing_dir_path, label_map_path, split, ground_level_image_size=(240, 180), normalize=True):
        """
        Initialize the dataset by reading the csv file and creating a mapping from image name to label. 
        Args:
        - csv_file_path (string): path to csv file
        - images_dir_path (string): path to directory with all the images 
        - remote_sensing_dir_path (string): path to directory with all the remote sensing images
        """
        self.csv_file = csv_file_path
        self.images_dir = images_dir_path
        self.remote_sensing_dir = remote_sensing_dir_path
        self.df = pd.read_csv(csv_file_path)
        self.split = split

        # load label map 
        with open(label_map_path, 'r') as file:
            self.label_map = json.load(file)
        self.num_classes = len(self.label_map.keys())

        faulty_rows = self.df[~self.df['name'].isin(self.label_map.keys())]
        print("Number of rows not filtered correctly (should be 0): ", len(faulty_rows))

        self.gl_image_size = ground_level_image_size 
        self.normalize = normalize

        print("Length of dataset: ", len(self.df))
        

    def __len__(self):
        """
        Return the length of the dataset
        """
        return int(len(self.df) / 10.0)

    def __getitem__(self, idx):
        """
        Return the image and label at the given index. 
        """

        row_obj = self.df.iloc[idx]

        # label 
        species_name = row_obj['name']
        assert species_name in self.label_map
        label = self.label_map[species_name]
        label = torch.tensor(label)

        # ground level img 
        extension_str = "png"
        img_name = str(row_obj['photo_id'])
        suffix_path = img_name[:3] + '/' + img_name + "." + extension_str
        img_path = os.path.join(self.images_dir, suffix_path)

        try:
          image_array = ml.load_image(img_path, resize=self.gl_image_size, normalize=self.normalize)
          image_array = torch.tensor(image_array, dtype=torch.float)
          if image_array.shape[0] == 1:
            image_array = image_array.repeat(3, 1, 1)
        except:
          print(f"The current image path ({img_path}) does not point to a valid file for ground level image...., using random tensor.")
          image_array = torch.randn(3, 240, 180)

        # just do a shape triple check to avoid haulting training loop 
        if image_array.shape != (3, 240, 180):
          print("image array shape error... using random tensor")
          image_array = torch.randn(3, 240, 180)


        # remote sensing image 
        npz_file = str(row_obj['remote_sensing'])
        if self.split == "train":
            uuid = str(row_obj['photo_uuid'])  
        else:
            uuid = str(row_obj['observation_uuid'])  # for val and test 
        rs_path = os.path.join(self.remote_sensing_dir, npz_file)

        try:
          npz_array = np.load(rs_path)
          if uuid in npz_array.files:
            image_array_rs = npz_array[uuid]
            image_array_rs = image_array_rs[:3,:,:]
            if self.normalize:
              image_array_rs = image_array_rs / 255.0
          else:
            image_array_rs = torch.randn(3, 256, 256)
            print("uuid not in npz_array, defaulting to random tensor...")
        except:
          print(f"Error loading in this remote sensing sample occurred. Defaulting to random array...")
          image_array_rs = torch.randn(3, 256, 256)
        
        # tensorize 
        if not torch.is_tensor(image_array_rs):
          image_array_rs = torch.tensor(image_array_rs, dtype=torch.float)

        # just do a shape triple check to avoid haulting training 
        if image_array_rs.shape != (3, 256, 256):
          print("image array shape for remote sensing image is incorrect... using random tensor")
          image_array_rs = torch.randn(3, 256, 256)

        
        return image_array, image_array_rs, label
