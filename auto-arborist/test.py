import dataset
import torch 
import numpy as np
import pdb

root_path = "/mnt/disks/MOUNT_DIR/vancouver_sample/"

ds = dataset.MiniAutoArborist(
        root_path, 
        "ds.csv", 
        "train", 
        normalize=True, 
        aerial_resize=(256, 256), 
        sv_resize=(256, 256)
    )



for i in range(60, 100):
    sv_image, aerial_image, label = ds[i]

# ds.save_img(sv_image, "plots/sv.png")
# ds.save_img(aerial_image, "plots/aerial.png")

    ds.save_img([sv_image, aerial_image], f"plots/{i}.png")