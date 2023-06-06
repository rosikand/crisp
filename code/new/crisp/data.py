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
import crisp


class CrispDataset(Dataset):
    def __init__(self, csv_file_path, images_dir_path, remote_sensing_dir_path, data_split="train"):
        """
        Initialize the dataset by reading the csv file and creating a mapping from image name to label. 
        Args:
        - csv_file_path (string): path to csv file
        - images_dir_path (string): path to directory with all the images 
        - remote_sensing_dir_path (string): path to directory with all the remote sensing images
        - data_split (string): train, validation, or test
        """
        self.csv_file = csv_file_path
        self.images_dir = images_dir_path
        self.remote_sensing_dir = remote_sensing_dir_path
        self.df = pd.read_csv(csv_file_path)
        self.data_split = data_split

        self.ground_level_image_size = (256, 256)  # resize to this 
        

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
          image_array = ml.load_image(img_path, resize=self.ground_level_image_size, normalize=True)
        except:
          suffix_path = img_name[:3] + '/' + img_name + "." + "png"
          img_path = os.path.join(self.images_dir, suffix_path)
          try:
            image_array = ml.load_image(img_path, resize=self.ground_level_image_size, normalize=True)
          except: 
            image_array = torch.randn(3, 256, 256)

        gl_img = image_array

        
        # remote sensing image
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
            rs_img = rs_img_obj[uuid_npy]
          else:
             rs_img = rs_img_obj[rs_img_obj.files[0]]
          rs_img = rs_img[:3,:,:]  # (4, 256, 256) -- > (3, 256, 256)
        except:
          # generate random remote sensing image
          rs_img = torch.randn(3, 256, 256)
          # raise Exception("remote sensing image not found")


        # tensorize and typecast 
        if not torch.is_tensor(gl_img):
          gl_img = torch.tensor(gl_img, dtype=torch.float32)
        else: 
          gl_img = gl_img.to(torch.float32)

        if not torch.is_tensor(rs_img):
          rs_img = torch.tensor(rs_img, dtype=torch.float32)
        else:
          rs_img = rs_img.to(torch.float32)
        

        # (3, 170, 247), (3, 256, 256)
        return (gl_img, rs_img)    



# base_path = "../../data/may17data"
# csv_path = os.path.join(base_path, "filtered.csv")
# images_dir_path = os.path.join(base_path, "images")
# remote_sensing_dir_path = os.path.join(base_path, "remote_sensing")

# ds = CrispDataset(csv_path, images_dir_path, remote_sensing_dir_path)
# model = crisp.CrispModel(
#             encoder_name = "resnet50",
#             embedding_dim = 512, 
#             pretrained_weights = None
#         )


# gl, rs = ds[5]
# gl = torch.unsqueeze(gl, 0)
# rs = torch.unsqueeze(rs, 0)
# gr_logits, rs_logits = model(gl, rs)

# crisp_loss = crisp.ClipLoss()
# # loss = crisp_loss(gr_logits, rs_logits)



# # dataloader 
# dataloader = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=False)

# # iterate through dataloader and print out the first 10 losses 
# for i, batch in enumerate(dataloader):
#   gl, rs = batch
#   gr_logits, rs_logits = model(gl, rs)
#   loss = crisp_loss(gr_logits, rs_logits)
#   print(loss)
#   if i == 10:
#     break

# # _ = ds[5]

# # pdb.set_trace()

