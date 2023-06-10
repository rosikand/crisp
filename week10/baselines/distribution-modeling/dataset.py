"""
File: data.py
------------------
Implement supervised PyTorch dataset for distribution modeling baseline. 
"""


import torch
from torch.utils.data import Dataset
import os 
from PIL import Image
import pandas as pd
import numpy as np
import rsbox 
from torch.utils.data import DataLoader
from rsbox import ml
import pdb
import json 
from tqdm import tqdm



class SpeciesDistributionDataset(Dataset):
    def __init__(self, csv_file_path, rs_dir_path, label_map_path, normalize=True):
        self.csv_file = csv_file_path
        self.images_dir = rs_dir_path
        self.df = pd.read_csv(csv_file_path)

        # load label map 
        with open(label_map_path, 'r') as file:
            self.label_map = json.load(file)
        self.num_classes = len(self.label_map.keys())
        

        # faulty rows 
        faulty_rows = self.df[~self.df['name'].isin(self.label_map.keys())]
        print("Number of rows not filtered correctly (should be 0): ", len(faulty_rows))

        self.default_img_shape = (3, 256, 256)

        self.normalize = normalize

        print("Length of dataset: ", len(self.df))
        print("Num classes: ", self.num_classes) 
        
    def __len__(self):
        """
        Return the length of the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Return the image and label at the given index.
        Note: 256, 256 is ground image dimensions.  
        """

        row_obj = self.df.iloc[idx]

        species_name = row_obj['name']
        
        assert species_name in self.label_map
        label = self.label_map[species_name]

        # img 
        npz_file = str(row_obj['remote_sensing'])
        uuid = str(row_obj['observation_uuid'])
        rs_path = os.path.join(self.images_dir, npz_file)

        try:
          npz_array = np.load(rs_path)
          if uuid in npz_array.files:
            image_array = npz_array[uuid]
            image_array = image_array[:3,:,:]
            if self.normalize:
              image_array = image_array / 255.0
        except:
          print(f"Error loading in this sample occurred. Defaulting to random array...")
          image_array = torch.randn(3, 256, 256)
        
        # tensorize 
        if not torch.is_tensor(image_array):
          image_array = torch.tensor(image_array, dtype=torch.float)

        label = torch.tensor(label)

        # just do a shape triple check to avoid haulting training 
        if image_array.shape != (3, 256, 256):
          print("image array shape error... using random tensor")
          image_array = torch.randn(3, 256, 256)

        return image_array, label
    


# # test 

# split = 'test'
# csv_file_path = f"/mnt/disks/mountDir/metadata/filtered_{split}.csv"
# images_dir_path = f"/mnt/disks/mountDir/{split}/remote_sensing/"
# label_map_path = "/mnt/disks/mountDir/metadata/label_map.json"
# normalize = True


# ds = SpeciesDistributionDataset(
#   csv_file_path = csv_file_path,
#   rs_dir_path = images_dir_path,
#   label_map_path = label_map_path,
#   normalize = normalize
# )


# loader = DataLoader(ds, batch_size=1, shuffle=False)

# progress_bar = tqdm(total=len(loader))
# mishaper = 0
# for i, data in enumerate(loader):
#   x, y = data
#   progress_bar.update(1)
#   if x[0].shape != (3, 256, 256):
#     print(x.shape)
#     mishaper += 1
  
