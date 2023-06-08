"""
File: datasets.py
------------------
Defines the PyTorch dataset class for auto arborist dataset. 
"""


import torch
from torch.utils.data import Dataset
import os 
import numpy as np 
import pandas as pd 
import pickle 
import pdb
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image



class MiniAutoArborist(Dataset):
    def __init__(self, root_dir, csv_file_path, split='train', normalize=False, aerial_resize=None, sv_resize=None):
        self.df = pd.read_csv(csv_file_path)
        self.root = root_dir
        self.normalize = normalize
        self.aerial_resize = aerial_resize
        self.sv_resize = sv_resize


        # filter based on the split 
        assert split == "train" or split == "test"
        self.df = self.df[self.df['split'] == split]

    
    def load_pickle_file(self, file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    
    def save_img(self, x, save_path):
        save_image(x, save_path)
        print("saved to: ", save_path)


    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        datum = self.df.iloc[idx]
        file = datum.file
        label = datum.label 

        path_ = os.path.join(self.root, file)
        sample = self.load_pickle_file(path_)
        sv_image = sample['sv_image']
        aerial_image = sample['aerial_image']


        sv_image = torch.tensor(sv_image, dtype=torch.float)
        aerial_image = torch.tensor(aerial_image, dtype=torch.float)

        # (H, W, C) --> (C, H, W)
        sv_image = torch.movedim(sv_image, -1, 0)
        aerial_image = torch.movedim(aerial_image, -1, 0)

        # resize 
        if self.sv_resize is not None:
            sv_image = T.Resize(size=self.sv_resize)(sv_image)

        if self.aerial_resize is not None:
            aerial_image = T.Resize(size=self.aerial_resize)(aerial_image)
        
        label = torch.tensor(label)


        # normalize
        if self.normalize:
            sv_image = sv_image / 255.0
            aerial_image = aerial_image / 255.0


        return sv_image, aerial_image, label 
        
