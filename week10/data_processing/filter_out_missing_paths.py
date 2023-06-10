import os
import pandas as pd
from tqdm import tqdm

split = "test"  # should be one of [train, validation, test]
csv_ = f"/mnt/disks/mountDir/metadata/filtered_{split}.csv"
df = pd.read_csv(csv_)
images_dir = f"/mnt/disks/mountDir/{split}/images"
save_path = f"/mnt/disks/mountDir/metadata/filtered_{split}.csv"
save_bool = False

def filter_df(df):
    paths_to_remove = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        img_name = str(row['photo_id'])
        suffix_path = img_name[:3] + '/' + img_name + ".png"
        img_path = os.path.join(images_dir, suffix_path)
        if not os.path.exists(img_path):
            paths_to_remove.append(index)
    
    df = df.drop(paths_to_remove)
    return df


print("old df len: ", len(df))
new_df = filter_df(df)
print("new df len: ", len(new_df))

# save 
if save_bool:
    new_df.to_csv(save_path)
    print(f'new csv saved correctly! the saved path is {save_path}')

