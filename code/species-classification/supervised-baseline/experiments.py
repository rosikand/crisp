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
import pdb
import configs
from tqdm import tqdm
import train



class BaseExp(train.Trainer):
    """
    Base experiment class. Simple supervised classification. 
    ---
    - Model: ResNet50 (imagenet pretrained)
    - Loss: CrossEntropyLoss
    - Optimizer: Adam 
    """
    
    def __init__(self, config=None, model_checkpoint=None):
        print("Initializing experiment...")
        self.cfg = config
        if self.cfg is None:
            self.cfg = configs.BaseConfig()

        
        self.device = self.cfg.device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        print(f"Using device: {self.device}")
        
        

        # data loading stuff 
        print("Initializing train dataset")
        self.train_ds = datasets.INaturalistClassification(
            csv_file_path=self.cfg.train_csv_file_path, 
            images_dir_path=self.cfg.train_images_dir_path
            )
        print("Initializing validation dataset")
        self.val_ds = datasets.INaturalistClassification(
            csv_file_path=self.cfg.val_csv_file_path, 
            images_dir_path=self.cfg.val_images_dir_path
            )
        print("Initializing test dataset")
        self.test_ds = datasets.INaturalistClassification(
            csv_file_path=self.cfg.test_csv_file_path, 
            images_dir_path=self.cfg.test_images_dir_path
            )
        

        self.trainloader = torch.utils.data.DataLoader(self.train_ds, batch_size=self.cfg.train_batch_size, shuffle=True)
        self.valloader = torch.utils.data.DataLoader(self.val_ds, batch_size=self.cfg.val_batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_ds, batch_size=1, shuffle=False)
        
        print(f"Length of training dataset: {len(self.trainloader)}")
        print(f"Length of val dataset: {len(self.valloader)}")
        print(f"Length of test dataset: {len(self.testloader)}")

        # model init stuff 
        
        self.num_classes = self.cfg.num_classes
        if self.num_classes is None:
            # placeholder for the time being. 
            self.num_classes = len(self.train_ds.unique_labels)

        self.model = self.construct_model(weights=None, num_classes=self.num_classes)

        self.model.to(self.device)

        if model_checkpoint is not None:
            self.load_weights(model_checkpoint)

        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # extra stuff 
        self.num_to_validate = self.cfg.num_to_validate
        self.num_to_test = self.cfg.num_to_test
        
        

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = None,
            verbose = True
        )
    

    def construct_model(self, weights=None, num_classes=2):
        print(f"Initializing model with {self.num_classes} num classes")
        model = torchvision.models.resnet50(weights=weights)
        d = model.fc.in_features
        model.fc = nn.Linear(d, num_classes)
        return model
    

    def is_negative(self, x, y):
        # if x or y is -1, return None
        x_neg = False
        y_neg = False
        for sample in x:
            if sample.eq(-1).all().item():
                x_neg = True
                break
        
        for sample in y:
            if sample.eq(-1).all().item():
                y_neg = True
                break

        return x_neg and y_neg
    

    # provide this abstract method (any subclass of torchplate.Experiment class must provide this) 
    # to calculate loss and optionally, other metrics to print over the course of training. 
    def evaluate(self, batch):
        # batch will be of form 
        # (x, y)


        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        if self.is_negative(x, y):
            return None

        # if x.eq(-1).all().item() and y.eq(-1).all().item():
        #    # print("Data file not found, skipping batch...")
        #     return None
        
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
            if i > self.num_to_validate:
                break
            x, y = batch
            if self.is_negative(x, y):
                # print("Data file not found, skipping batch...")
                continue
            x = x.to(self.device)
            y = y.to(self.device)
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
            if i > self.num_to_test:
                break
            x, y = batch
            if self.is_negative(x, y):
                # print("Data file not found, skipping batch...")
                continue
            x = x.to(self.device)
            y = y.to(self.device)
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


    def on_run_end(self):
        self.save_weights()



class MLPExp(BaseExp):
    """
    MLP experiment class. Everything is the same as BaseExp except
    the model is a simple MLP instead of ResNet50. Thus, only code
    that needs to be overriden is the model construction. Everything
    else is the inherited from BaseExp. 
    ---
    model: MLP
    """
    def __init__(self, config=None,  model_checkpoint=None):
        super().__init__(config=config, model_checkpoint=model_checkpoint)

        print("Initializing MLP experiment...")

        # input shape is 
        input_shape_ = self.train_ds[0][0].shape
        self.model = models.MLP(
            input_shape=input_shape_, 
            num_classes=self.num_classes
            )
        self.model.to(self.device)
