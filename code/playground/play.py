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


tr_df = pd.read_csv(train_csv)
te_df = pd.read_csv(test_csv)
val_df = pd.read_csv(val_csv)

tr_labels = set(tr_df['name'].tolist())
te_labels = set(te_df['name'].tolist())
val_labels = set(val_df['name'].tolist())

inter = tr_labels.intersection(te_labels, val_labels)

label_map = {}

for i, elem in enumerate(inter):
    label_map[str(i)] = elem

# pickle 
with open("label_map.pkl", "wb") as file:
    pickle.dump(label_map, file)



