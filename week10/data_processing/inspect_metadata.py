import json
import pandas as pd 
import os 
import sys 
import pdb


split = "validation"  # should be one of [train, validation, test]
print_label_counts = True
save_counts = False
save_counts_path = f"/mnt/disks/mountDir/metadata/counts/{split}_counts.csv"

csv_ = f"/mnt/disks/mountDir/metadata/filtered_{split}.csv"
label_map_path = "/mnt/disks/mountDir/metadata/label_map.json"
df = pd.read_csv(csv_)
print("Length of dataset: ", len(df))


# label map 
with open(label_map_path, 'r') as file:
    label_map = json.load(file)
num_classes = len(label_map.keys())
print("Num classes: ", num_classes) 


faulty_rows = df[~df['name'].isin(label_map.keys())]
print("Number of rows not filtered correctly (should be 0): ", len(faulty_rows))


counts = df['name'].value_counts()
if print_label_counts:
    print(counts)
    print("len of counts: ", len(counts))

if save_counts:
    counts.to_csv(save_counts_path, header=True) 
    print("Counts saved to: ", save_counts_path)