import os
import pandas as pd
from tqdm import tqdm

split = "validation"  # should be one of [train, validation, test]
csv_ = f"/mnt/disks/mountDir/metadata/filtered_{split}.csv"
df = pd.read_csv(csv_)
images_dir = f"/mnt/disks/mountDir/{split}/images"

num_missing = 0

for index, row in tqdm(df.iterrows(), total=len(df)):
    img_name = str(row['photo_id'])
    suffix_path = img_name[:3] + '/' + img_name + ".png"
    img_path = os.path.join(images_dir, suffix_path)
    if os.path.exists(img_path) == False:
        print(f"Path does not exist for row {index} at: {img_path}")
        num_missing += 1

print("total num missing: ", num_missing)



