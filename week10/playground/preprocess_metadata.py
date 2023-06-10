"""
All in one file to create new observations.csv's with the desired contents. 
For all three train/test/val observations.csv files, we do: 
- Filter out any rows where "rank" != "species". We note that we don't use "rank_level" since this appears to have a mismatch. For example: 
rank_level                                                          10.0
rank                                                              hybrid
name                                                  Cistus × purpureus  # should be a species at rank_level 10.0! 
- Filter out non-research grade rows 
- create label map based on unique "name"'s from train csv. 
- filter out all val and test samples where the "name" does not appear in the above label map. 
"""

import pdb
import numpy as np 
import torch
import os 
from rsbox import ml 
import sys
import pandas as pd
import pickle
import json 


train_csv = "/mnt/disks/mountDir/train/observations.csv"
test_csv = "/mnt/disks/mountDir/test/observations.csv"
val_csv = "/mnt/disks/mountDir/validation/observations.csv"

# save_prefix_path = "/mnt/disks/mountDir/metadata"
save_prefix_path = 'test'

trdf = pd.read_csv(train_csv)
tedf = pd.read_csv(test_csv)
vadf = pd.read_csv(val_csv)

print("og length of trdf:", len(trdf))
print("og length of tedf:", len(tedf))
print("og length of vadf:", len(vadf))


# filter for research grade only 
trdf = trdf[trdf["quality_grade"] == 'research']
tedf = tedf[tedf["quality_grade"] == 'research']
vadf = vadf[vadf["quality_grade"] == 'research']

print("research filtered length of trdf:", len(trdf))
print("research filtered length of tedf:", len(tedf))
print("research filtered length of vadf:", len(vadf))


# filter out non-species level names 
trdf = trdf[trdf["rank"] == 'species']
tedf = tedf[tedf["rank"] == 'species']
vadf = vadf[vadf["rank"] == 'species']

print("species filtered length of trdf:", len(trdf))
print("species filtered length of tedf:", len(tedf))
print("species filtered length of vadf:", len(vadf))

# create label map based on unique names from training set 

tr_labels = list(set(trdf['name'].tolist()))

label_map = {}
for i, class_name in enumerate(tr_labels):
    if class_name not in label_map:
        label_map[class_name] = i
    else:
        raise ValueError("you've done something wrong!")
        sys.exit()



# now filter out the labels in test and val not in tr_labels 
tedf = tedf[tedf['name'].isin(tr_labels)]
vadf = vadf[vadf['name'].isin(tr_labels)]
print("label filtered length of tedf:", len(tedf))
print("label filtered length of vadf:", len(vadf))


# save label_map and new csv's 

# csv's  
tr_new_path = os.path.join(save_prefix_path, "filtered_train.csv")
te_new_path = os.path.join(save_prefix_path, "filtered_test.csv")
val_new_path = os.path.join(save_prefix_path, "filtered_validation.csv")
trdf.to_csv(tr_new_path)
tedf.to_csv(te_new_path)
vadf.to_csv(val_new_path)
print(f'csvs saved correctly! train path is {tr_new_path}')

# Save label_map to JSON file
suffix = "label_map.json"
label_map_path = os.path.join(save_prefix_path, suffix)
with open(label_map_path, 'w') as file:
    json.dump(label_map, file)

print(f"JSON file {label_map_path} created successfully!")


# load the csv's back in to verify 
trdf_ = pd.read_csv(tr_new_path)
tedf_ = pd.read_csv(te_new_path)
vadf_ = pd.read_csv(val_new_path)

print('-------------------')
print("final length of trdf:", len(trdf))
print("final length of tedf:", len(tedf))
print("final length of vadf:", len(vadf))
