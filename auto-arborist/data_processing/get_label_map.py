"""
File: get_label_map.py
------------------
Get all labels from vancouver_sample data 
since the official label map would be for the
entire dataset which we don't have right now.  
"""


import pickle 
import os 
import pdb
from tqdm import tqdm


root_path = "/mnt/disks/MOUNT_DIR/vancouver_sample/"


def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


all_labels = []


for filename in tqdm(os.listdir(root_path)):
    if filename.endswith(".pkl"):  
        file_path = os.path.join(root_path, filename)  
        datum = load_pickle_file(file_path)
        label = datum['label_text']
        all_labels.append(label)


unique_labels = list(set(all_labels))
print(f"Number of unique classes: {unique_labels}")


pdb.set_trace()


