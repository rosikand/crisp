import os
import pandas as pd
from tqdm import tqdm
import pdb
import numpy as np  # for np.load 


split = "validation"  # should be one of [train, validation, test]
csv_ = f"/mnt/disks/mountDir/metadata/filtered_{split}.csv"
df = pd.read_csv(csv_)
images_dir = f"/mnt/disks/mountDir/{split}/images"
rs_dir = f"/mnt/disks/mountDir/{split}/remote_sensing/"
rs = True

pdb.set_trace()

num_missing = 0
num_missing_rs = 0
for index, row in tqdm(df.iterrows(), total=len(df)):
    img_name = str(row['photo_id'])
    suffix_path = img_name[:3] + '/' + img_name + ".png"
    img_path = os.path.join(images_dir, suffix_path)
    if rs:
        rs_name = str(row['remote_sensing'])
        rs_path = os.path.join(rs_dir, rs_name)
        if os.path.exists(rs_path) == False:
            print("rs path doesn't exist here...")
            num_missing_rs += 1
    if os.path.exists(img_path) == False:
        print(f"Path does not exist for row {index} at: {img_path}")
        num_missing += 1

print("total num missing: ", num_missing)
print("total num missing rs: ", num_missing_rs)




