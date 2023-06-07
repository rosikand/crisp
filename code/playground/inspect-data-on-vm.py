import pdb
import numpy as np 
import torch
import os 
from rsbox import ml 
import pandas as pd


disk_dir = "/mnt/disks/mountDir/"
split = 'train'
root_path = os.path.join(disk_dir, split)
csv_path = os.path.join(root_path, "observations.csv")
img_dir = os.path.join(root_path, "images")


# --- 

df = pd.read_csv(csv_path)
df = df[df['quality_grade'] == 'research']
row = df.iloc[45]
pdb.set_trace()

column_list = df['name'].tolist()
available_nums = []
for elem in column_list:
    num = str(elem)[:3]
    if num not in available_nums:
        available_nums.append(num)

available_nums = list(set(available_nums))

pdb.set_trace()

# img_name = str(self.df.loc[idx, 'photo_id'])
#         extension_str = str(self.df.loc[idx, 'extension'])
#         suffix_path = img_name[:3] + '/' + img_name + "." + extension_str
#         img_path = os.path.join(self.images_dir, suffix_path)

# pdb.set_trace()




