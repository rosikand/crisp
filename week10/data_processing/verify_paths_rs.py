import os
import pandas as pd
from tqdm import tqdm
import pdb
import numpy as np  # for np.load 


split = "train"  # should be one of [train, validation, test]
csv_ = f"/mnt/disks/mountDir/metadata/filtered_{split}.csv"
df = pd.read_csv(csv_)
rs_dir = f"/mnt/disks/mountDir/{split}/remote_sensing/"
num_missing = 0
for index, row in tqdm(df.iterrows(), total=len(df)):
    name_ = str(row['remote_sensing'])
    path_ = os.path.join(rs_dir, name_)
    if os.path.exists(path_) == False:
        print(f"Path does not exist for row {index} at: {path_}")
        num_missing += 1

print("total num missing: ", num_missing)



