import os
import pandas as pd 
import pdb


csv_save_dir = "/mnt/disks/mountDir/filtered" 

tr = os.path.join(csv_save_dir, "filtered_train.csv")
te = os.path.join(csv_save_dir, "filtered_test.csv")
val = os.path.join(csv_save_dir, "filtered_validation.csv")


tr_df = pd.read_csv(tr)
te_df = pd.read_csv(te)
val_df = pd.read_csv(val)

pdb.set_trace()