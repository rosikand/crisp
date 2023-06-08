"""
File: load_datum.py
------------------
The purpose of this module is to show how to
load one of the .pkl files and visualize the sample. 
"""

import pickle 
import os 
import pdb


root_path = "/mnt/disks/MOUNT_DIR/vancouver_sample/"
datum_name = "vancouver-train-3745102878536441617.pkl"
path = os.path.join(root_path, datum_name)


def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)



datum = load_pickle_file(path)

for elem in datum.keys():
    print(elem)


