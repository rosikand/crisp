# filters out non-intersecting labels across splits 

import os
import pandas as pd 
import pickle 
import pdb


train_csv = "/mnt/disks/mountDir/train/observations.csv"
test_csv = "/mnt/disks/mountDir/test/observations.csv"
val_csv = "/mnt/disks/mountDir/validation/observations.csv"


csv_save_dir = "/mnt/disks/mountDir/filtered" 

tr_df = pd.read_csv(train_csv)
te_df = pd.read_csv(test_csv)
val_df = pd.read_csv(val_csv)


prefix = "/mnt/disks/mountDir"
suffix = "label_map.pkl"
path = os.path.join(prefix, suffix)


def get_label_map(path):
    with open(path, 'rb') as file:
        loaded_object = pickle.load(file)    
    
    return loaded_object


label_map = get_label_map(path)
valid_names = list(label_map.values())
filtered_tr = tr_df[tr_df['name'].isin(valid_names)]
filtered_te = te_df[te_df['name'].isin(valid_names)]
filtered_val = val_df[val_df['name'].isin(valid_names)]

# save 
filtered_tr.to_csv(os.path.join(csv_save_dir, "filtered_train.csv"))
filtered_te.to_csv(os.path.join(csv_save_dir, "filtered_test.csv"))
filtered_val.to_csv(os.path.join(csv_save_dir, "filtered_validation.csv"))

pdb.set_trace()

