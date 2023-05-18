"""
File: datasets.py
------------------
Defines the PyTorch dataset classes.

THIS NEEDS TO BE UPDATED FOR CRISP TO TAKE IN REMOTE SENSING AS WELL! 
"""


import torch
from torch.utils.data import Dataset
import os 
from PIL import Image
import pandas as pd
import numpy as np
import rsbox  # custom package I wrote to handle some common ML tasks such as loading images into (C, H, W) form. 
from rsbox import ml
import pdb
import os



class CrispDataset(Dataset):
    def __init__(self, csv_file_path, images_dir_path, remote_sensing_dir_path):
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
        

    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return the image and label at the given index. 
        """

        # ground level image 
        
        img_name = str(self.df.loc[idx, 'photo_id'])
        extension_str = str(self.df.loc[idx, 'extension'])
        suffix_path = img_name[:3] + '/' + img_name + "." + extension_str
        img_path = os.path.join(self.images_dir, suffix_path)

        try:
          image_array = ml.load_image(img_path, resize=None, normalize=True)
        except:
          suffix_path = img_name[:3] + '/' + img_name + "." + "png"
          img_path = os.path.join(self.images_dir, suffix_path)
          try:
            image_array = ml.load_image(img_path, resize=None, normalize=True)
          except: 
            image_array = torch.randn(3, 256, 256)

        gl_img = image_array

        
        # remote sensing image
        remote_sensing_img_name = str(self.df.loc[idx, 'remote_sensing'])
        rs_path = os.path.join(self.remote_sensing_dir, remote_sensing_img_name)

        try:
          rs_img_obj = np.load(rs_path)
          rs_files = rs_img_obj.files
          rs_img = rs_img_obj[rs_files[0]]
          rs_img = rs_img[:3,:,:]  # (4, 256, 256) -- > (3, 256, 256)
        except:
          # generate random remote sensing image
          rs_img = torch.randn(3, 256, 256)
          # raise Exception("remote sensing image not found")
        
        

        # (3, 170, 247), (3, 256, 256)
        return (gl_img, rs_img)    



base_path = "../../data/may17data"
csv_path = os.path.join(base_path, "filtered.csv")
images_dir_path = os.path.join(base_path, "images")
remote_sensing_dir_path = os.path.join(base_path, "remote_sensing")

ds = CrispDataset(csv_path, images_dir_path, remote_sensing_dir_path)


# dataloader 
dataloader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

# for batch in dataloader:
#   gl, rs = batch
#   pdb.set_trace()

# _ = ds[5]

# pdb.set_trace()

