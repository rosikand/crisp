"""
File: create_split.py
------------------
Creates a dictionary which maps
file path to split (train or test)
for the vancouver_sample dataset.  
"""

import os
import random
import pdb
from tqdm import tqdm
import pickle 
import csv
import json


data_directory = "/mnt/disks/MOUNT_DIR/vancouver_sample/"
train_split = 0.9


def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


# utility 
def get_unique_labels(paths):
    all_labels = []

    for filename in tqdm(paths):
        if filename.endswith(".pkl"):  
            file_path = os.path.join(data_directory, filename) 
            datum = load_pickle_file(file_path)
            label = datum['label_text']
            all_labels.append(label)


    unique_labels = list(set(all_labels))
    
    return unique_labels



filepaths = os.listdir(data_directory)

random.shuffle(filepaths)
train_size = int(len(filepaths) * train_split)

train_data = filepaths[:train_size]
test_data = filepaths[train_size:]

train_labels = get_unique_labels(train_data)
test_labels = get_unique_labels(test_data)

intersection = list(set(train_labels) & set(test_labels))
int_len = len(intersection)
og_label_len = max(len(train_labels), len(test_labels))
difference = og_label_len - int_len

# if difference > 10:
#     sys.exit()

print(f"intersecting labels diff: {difference}")

def sift_out_nonintersecting(intersection_labels, train_data, test_data):

    new_train_files = []
    for filename in tqdm(train_data):
        if filename.endswith(".pkl"):  
            file_path = os.path.join(data_directory, filename) 
            datum = load_pickle_file(file_path)
            label = datum['label_text']
            if label in intersection_labels:
                new_train_files.append(filename)

    
    new_test_files = []
    for filename in tqdm(test_data):
        if filename.endswith(".pkl"):  
            file_path = os.path.join(data_directory, filename) 
            datum = load_pickle_file(file_path)
            label = datum['label_text']
            if label in intersection_labels:
                new_test_files.append(filename)

    return new_train_files, new_test_files


og_tr_len = len(train_data)
og_te_len = len(test_data)


train_data, test_data = sift_out_nonintersecting(intersection, train_data, test_data)

print("diff tr: ", og_tr_len - len(train_data))
print("diff te: ", og_te_len - len(test_data))


# create label map 
label_map = {}
for i, label in enumerate(intersection):
    if label not in label_map:
        label_map[label] = i

# save 
# Save dictionary to JSON file
with open("label_map.json", 'w') as file:
    json.dump(label_map, file)

print(f"JSON file 'label_map.json' created successfully!")


print("train len: ", len(train_data))
print("test len: ", len(test_data))
# construct df 

master_list = []

for filepath in train_data:
    curr_datum = []
    curr_datum.append(filepath)

    # get label 
    datum_obj = load_pickle_file(os.path.join(data_directory, filepath))
    label = datum_obj['label_text']
    label_num = label_map[label]
    curr_datum.append(label)
    curr_datum.append(label_num)
    curr_datum.append("train")

    master_list.append(curr_datum)


for filepath in test_data:
    curr_datum = []
    curr_datum.append(filepath)

    # get label 
    datum_obj = load_pickle_file(os.path.join(data_directory, filepath))
    label = datum_obj['label_text']
    label_num = label_map[label]
    curr_datum.append(label)
    curr_datum.append(label_num)
    curr_datum.append("test")

    master_list.append(curr_datum)



# Specify the file name
save_filename = 'ds.csv'
headers = ['file', 'name', 'label', 'split']


# Write data to the CSV file
with open(save_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the headers
    writer.writerow(headers)
    
    # Write the data rows
    writer.writerows(master_list)

print(f"CSV file '{save_filename}' created successfully!")

# # strings_not_in_train = list(set([string for string in test_labels if string not in train_labels]))



# data_split = {}

# for filepath in train_data:
#     data_split[os.path.join(data_directory, filepath)] = "train"

# for filepath in test_data:
#     data_split[os.path.join(data_directory, filepath)] = "test"

# def save_dict_to_csv(dictionary, filename):
#     with open(filename, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Key', 'Value'])  # Write the header row
#         for key, value in dictionary.items():
#             writer.writerow([key, value])

# save_dict_to_csv(data_split, "ds.csv")

