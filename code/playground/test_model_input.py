
import numpy as np
import torch 
import torchvision 
import torch.nn as nn
import rsbox 
from rsbox import ml 
import os
import matplotlib.pyplot as plt
from PIL import Image
import pdb


# model 
num_classes = 10 

model = torchvision.models.resnet50(weights=None)
d = model.fc.in_features
model.fc = nn.Linear(d, num_classes)


# load data point 

root_data_path = '../../data'
img_path = 'playground/images/100/100211007.png'
joined_path = os.path.join(root_data_path, img_path)

img = torch.unsqueeze(torch.tensor(ml.load_image(joined_path, resize=(224, 224), normalize=True), dtype=torch.float32), dim=0)
logits = model(img)
pdb.set_trace()

