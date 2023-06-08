"""
File: get_dataset_stats.py
------------------
Gets number of samples per class in dataset. 
"""


import pandas as pd 
import pdb 


csv_path = 'ds.csv'
split = 'test'

df = pd.read_csv(csv_path)
df = df[df['split'] == split]
label_counts = df['name'].value_counts()
print(label_counts)

