import torch
import torchvision
import matplotlib.pyplot as plt
import pdb
import numpy as np 
from rsbox import ml


# Load the LFWPAIRS dataset
lfw_pairs = torchvision.datasets.LFWPairs(root='./data', download=True)

# Print some information about the dataset
print('LFWPAIRS dataset')
print('Number of pairs:', len(lfw_pairs))

num_examples = [3000, 1000, 2550, 4500]


for i in num_examples:
    # Get a pair of images and their label
    img1, img2, label = lfw_pairs[i]

    img1 = np.array(img1)
    img2 = np.array(img2)

    ml.plot(img1)
    ml.plot(img2)
    print(label)
