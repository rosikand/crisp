import pandas as pd 
import pdb

# load csv
df = pd.read_csv('test_observations.csv')

# filter out rows based on the first three digits of photo_id
valid_prefixes = [str(num) for num in range(100, 288)]
mask = df['photo_id'].astype(str).str.startswith(tuple(valid_prefixes))
filtered_df = df.loc[mask]

# save 
filtered_df.to_csv('filtered_test_observations.csv')

