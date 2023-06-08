"""
File: load_ds_and_visualize.py
------------------
Load in the ds.csv file and visualize a few samples! 
"""

import pandas as pd 
import pdb 


df = pd.read_csv('ds.csv')
train = df[df['split'] == 'train']
test = df[df['split'] == 'test']

pdb.set_trace()