"""
File: experiments.py
------------------
This file holds the experiments which are
subclasses of torchplate.experiment.Experiment. 
torchplate is a small python package I wrote which
handles the training loop and various other utilities
such as logging, checkpointing, etc. 
"""

import numpy as np
import torchplate
from torchplate import (
        experiment,
        utils
    )
from torchplate import metrics as tp_metrics
import models
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import datasets
import torchvision
import configs
from tqdm import tqdm



class BaseExp(experiment.Experiment):
    """
    Base experiment class. Simple supervised classification. 
    ---
    - Model: ResNet50 (imagenet pretrained)
    - Loss: CrossEntropyLoss
    - Optimizer: Adam 
    """
    
    def __init__(self, config=None):
        self.cfg = config
        if self.cfg is None:
            self.cfg = configs.BaseConfig()
        self.model = self.construct_model(pretrained=True, num_classes=self.cfg.num_classes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # data loading 
        self.train_ds = datasets.INaturalistClassification(
            csv_file_path=self.cfg.csv_file_path, 
            images_dir_path=self.cfg.images_dir_path
            )
        self.val_ds = datasets.INaturalistClassification(
            csv_file_path=self.cfg.csv_file_path, 
            images_dir_path=self.cfg.images_dir_path
            )
        self.test_ds = datasets.INaturalistClassification(
            csv_file_path=self.cfg.csv_file_path, 
            images_dir_path=self.cfg.images_dir_path
            )
        
        self.trainloader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.cfg.train_batch_size, shuffle=True)
        self.valloader = torch.utils.data.DataLoader(self.val_ds, batch_size=self.cfg.val_batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_ds, batch_size=1, shuffle=False)
        

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = None,
            verbose = True
        )
    

    def construct_model(self, pretrained=True, num_classes=2):
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, num_classes)
        return model
    

    # provide this abstract method (any subclass of torchplate.Experiment class must provide this) 
    # to calculate loss and optionally, other metrics to print over the course of training. 
    def evaluate(self, batch):
        # batch will be of form 
        # (x, y)

        x, y = batch
        logits = self.model(x)
        loss_val = self.criterion(logits, y)
        acc = tp_metrics.calculate_accuracy(logits, y)

        metrics_dict = {'loss': loss_val, 'accuracy': acc}
        
        return metrics_dict
    


    def validate(self):
        # get accuracy on val set
        acc = tp_metrics.Accuracy()
        tqdm_loader = tqdm(self.valloader)
        i = 0
        for batch in tqdm_loader:
            i += 1
            x, y = batch
            logits = self.model(x)
            acc.update(logits, y)
            tqdm_loader.set_description(f'Accuracy: {acc.get()}')

        
        print(f'Validation accuracy: {acc.get()}')

    
    def test(self):
        # get accuracy on test set
        acc = tp_metrics.Accuracy()
        tqdm_loader = tqdm(self.testloader)
        i = 0
        for batch in tqdm_loader:
            i += 1
            x, y = batch
            logits = self.model(x)
            acc.update(logits, y)
            tqdm_loader.set_description(f'Accuracy: {acc.get()}')

        
        print(f'Test accuracy: {acc.get()}')
    
    
    # override default callbacks 
    def on_epoch_start(self):
        self.model.train()

    
    def on_epoch_end(self):
        if self.epoch_num % self.cfg.val_freq == 0:
            self.model.eval()
            self.validate()



class MLPExp(BaseExp):
    """
    MLP experiment class. Everything is the same as BaseExp except
    the model is a simple MLP instead of ResNet50. Thus, only code
    that needs to be overriden is the model construction. Everything
    else is the inherited from BaseExp. 
    ---
    model: MLP
    """
    def __init__(self, config=None):
        super().__init__(config=config)

        # input shape is 
        input_shape_ = self.train_ds[0][0].shape
        self.model = models.MLP(
            input_shape=input_shape_, 
            num_classes=self.cfg.num_classes
            )
