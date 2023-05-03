"""
File: configs.py 
----------------------
Holds experiment configuration classes. 
i.e., specifies parameter choices. 
To run experiment, select on of these
classes and pass it to the experiment class. 
See runner.py. 
"""


import experiments
import torchplate
import rsbox 
import wandb
from rsbox import ml, misc
import torch.optim as optim
import torch
from torch import nn
import pickle 
import models
import datasets


"""
Config spec:

(Required)
- experiment_class: specifies which experiment class to use from experiments.py. 
- val_freq: how often to run validation (in epochs).
- num_classes: number of species classes (i.e., how many outputs the model should have for logits).
- csv_file_path: relative path to obersvations.csv file. 
- images_dir_path: relative path to images directory.
- train_batch_size: batch size for training.
- val_batch_size: batch size for validation.

(Optional)
...
"""


class TemplateConfig:
    # just for show 
    experiment_class = experiments.BaseExp
    val_freq = 200000
    num_classes = None
    train_csv_file_path = None
    train_images_dir_path = None
    val_csv_file_path = None
    val_images_dir_path = None
    test_csv_file_path = None
    test_images_dir_path = None
    train_batch_size = None
    val_batch_size = None


class BaseConfig:
    # ... 
    experiment_class = experiments.BaseExp
    val_freq = 200000
    num_classes = None
    train_csv_file_path = "../../../data/playground/filtered_observations.csv"
    train_images_dir_path = "../../../data/playground/images"
    val_csv_file_path = "../../../data/playground/filtered_observations.csv"
    val_images_dir_path = "../../../data/playground/images"
    test_csv_file_path = "../../../data/playground/filtered_observations.csv"
    test_images_dir_path = "../../../data/playground/images"
    train_batch_size = 1
    val_batch_size = 1
    