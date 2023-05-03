import pandas as pd 
import pdb

data = {'species_name': ['dog', 'cat', 'dog', 'bird', 'cat'],
        'image_path': ['path/to/dog.jpg', 'path/to/cat.jpg', 'path/to/dog2.jpg', 'path/to/bird.jpg', 'path/to/cat2.jpg']}
df = pd.DataFrame(data)

label_map = {}  
unique_labels = df['species_name'].unique()  
for i, label in enumerate(unique_labels):
    label_map[label] = i  
    
df['label'] = df['species_name'].map(label_map)
print(df)

pdb.set_trace()