# just to test file paths 

import rsbox 
from rsbox import ml 
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pdb

# load data

root_data_path = '../../data'

img_path = 'playground/images/100/100211007.png'

joined_path = os.path.join(root_data_path, img_path)

ml.plot_png(joined_path, color=True)
img = np.array(Image.open(joined_path))
ml.plot(img)


# # load image with PIL
# img = Image.open(joined_path)

# pdb.set_trace()