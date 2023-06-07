"""
The labels between train, test, and val splits
may not all have the same classes. For example, 
the test split might have samples of labels
never seen before during training and vise versa. 
To get around this, here we take the intersection between
the unique labels between the three splits to get the 
labels that persist throughout all splits. 
"""

import pdb
import numpy as np 
import torch
import os 
from rsbox import ml 
import pandas as pd
import pickle


train_csv = "/mnt/disks/mountDir/train/observations.csv"
test_csv = "/mnt/disks/mountDir/test/observations.csv"
val_csv = "/mnt/disks/mountDir/validation/observations.csv"

save_prefix_path = "/mnt/disks/mountDir/filtered"

tr_df = pd.read_csv(train_csv)
te_df = pd.read_csv(test_csv)
val_df = pd.read_csv(val_csv)

tr_labels = set(tr_df['name'].tolist())
te_labels = set(te_df['name'].tolist())
val_labels = set(val_df['name'].tolist())

inter = tr_labels.intersection(te_labels, val_labels)

label_map = {}

for i, elem in enumerate(inter):
    label_map[elem] = str(i)

# pickle 
save_path = os.path.join(save_prefix_path, "label_map.pkl")
with open(save_path, "wb") as file:
    pickle.dump(label_map, file)

print(f"saved at {save_path}")


