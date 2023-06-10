"""
File: data.py
------------------
Implement supervised PyTorch dataset for classification baseline. 
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


test_ds = True  # for debugging 
num_paths_missing = 0


class INaturalistClassification(Dataset):
    def __init__(self, csv_file_path, images_dir_path, label_map_path, image_resize=None, normalize=False):
        """
        Initialize the dataset by reading the csv file and creating a mapping from image name to label. 
        Args:
        - csv_file_path (string): path to csv file
        - images_dir_path (string): path to directory with all the images 
        """
        self.csv_file = csv_file_path
        self.images_dir = images_dir_path
        self.df = pd.read_csv(csv_file_path)

        # load label map 
        with open(label_map_path, 'r') as file:
            self.label_map = json.load(file)
        self.num_classes = len(self.label_map.keys())
        

        # faulty rows 
        faulty_rows = self.df[~self.df['name'].isin(self.label_map.keys())]
        print("Number of rows not filtered correctly (should be 0): ", len(faulty_rows))

        self.default_img_shape = (3, 224, 224)

        self.image_resize = image_resize  # can be None 
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
        """

        row_obj = self.df.iloc[idx]

        species_name = row_obj['name']
        
        assert species_name in self.label_map
        label = self.label_map[species_name]

        # img 
        extension_str = "png"
        img_name = str(row_obj['photo_id'])
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
    





def test_ds():
  csv_file_path = "/mnt/disks/mountDir/metadata/filtered_train.csv"
  images_dir_path = "/mnt/disks/mountDir/train/images"
  label_map_path = "/mnt/disks/mountDir/metadata/label_map.json"
  image_resize = None
  normalize = True

  trds = INaturalistClassification(
    csv_file_path = csv_file_path,
    images_dir_path = images_dir_path,
    label_map_path = label_map_path,
    image_resize = image_resize,
    normalize = normalize
  )

  loader = DataLoader(trds, batch_size=1, shuffle=False)

  progress_bar = tqdm(total=len(loader))
  for i, data in enumerate(loader):
    x, y = data
    progress_bar.update(1)

    # pdb.set_trace()

    # # Print the shape of the input and label tensors
    # print(f"Element {i+1}:")
    # print("Input shape:", inputs.shape)
    # print("Label shape:", labels.shape)
    # # print("Label: ", label)
  
  print("Done looping through dataset. Did any error messages appear?")
  print("num paths missing: ", num_paths_missing)



if test_ds:
  test_ds()
