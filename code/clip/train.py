"""
File: train.py
------------------
Runner file to train the model. 
"""


import argparse
import torchplate
import crisp
from torchplate import experiment
from torchplate import utils
from torchplate import metrics as tp_metrics
from tqdm import tqdm
import torch
import data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
from rsbox import ml, misc
import datasets 
import os


class CrispExperiment(torchplate.experiment.Experiment):
     
    def __init__(self, model_checkpoint=None):
        print("Initializing base crisp experiment...")


        # vars 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        print(f"Using device: {self.device}")



        # model initialization 
        self.model = crisp.CrispModel(
            encoder_name = "resnet50",
            embedding_dim = 512, 
            pretrained_weights = None
        )
        self.model.to(self.device)

        # loss and optimizer 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = crisp.ClipLoss()

        # data 
        batch_size = 10
        base_path = "../../data/may17data"
        csv_path = os.path.join(base_path, "filtered.csv")
        images_dir_path = os.path.join(base_path, "images")
        remote_sensing_dir_path = os.path.join(base_path, "remote_sensing")
        ds = data.CrispDataset(csv_path, images_dir_path, remote_sensing_dir_path)
        self.trainloader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False)


        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            save_weights_every_n_epochs = None, 
            wandb_logger = None,
            verbose = True,
            experiment_name = misc.timestamp()
        )


    def evaluate(self, batch):
        gl, rs = batch
        gl = gl.to(self.device)
        rs = rs.to(self.device)

        gl_logits, rs_logits = self.model(gl, rs)
        loss = self.criterion(gl_logits, rs_logits)

        return loss 




experiment = CrispExperiment()
experiment.train(num_epochs=1, display_batch_loss=True)


