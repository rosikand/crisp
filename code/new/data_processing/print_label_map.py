# shows how to load and print label map 

import os 
import pickle 


# prefix = "/mnt/disks/mountDir"
# suffix = "label_map.pkl"
# path = os.path.join(prefix, suffix)
path = "/mnt/disks/mountDir/filtered/label_map.pkl"

def get_label_map(path):
    with open(path, 'rb') as file:
        loaded_object = pickle.load(file)    
    
    return loaded_object


res = get_label_map(path)
print(res)
