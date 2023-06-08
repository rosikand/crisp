"""
File: resnet-baseline.py
------------------
Implements a ResNet-50 CNN baseline model for STL-10 dataset
"""


import torchplate
from torchplate import experiment
from torchplate import utils
from torchplate import metrics as tp_metrics
import sys
import dataset
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
from urllib.request import urlopen
from tqdm import tqdm
import pdb
import rsbox 
import wandb
import argparse
from rsbox import ml, misc
from torchvision import transforms
import json



# ----------------- Global config vars ----------------- #

root_path = "/mnt/disks/proto/vancouver_sample/"
csv_path = "ds.csv"
label_map_json_file = 'label_map.json'
normalize = True
# aerial_resize = (512, 512)
# sv_resize = (512, 512)
aerial_resize = None
sv_resize = None
batch_size = 16
logger = None  # eventually wandb 
experiment_name = "baseline" + "-" + misc.timestamp()


# ----------------- Utilities ----------------- #

def get_num_classes(json_file_path):
    # takes in label map json path and returns num classes 
    with open(json_file_path, 'r') as file:
        label_map = json.load(file)

    num_keys = len(label_map.keys())

    return num_keys


# ----------------- Training Experiments ----------------- #




class StreetLevelExperiment(torchplate.experiment.Experiment):
    def __init__(self): 
        """
        In this baseline experiment, we train a resnet-18 model
        to predict the species of the tree using only street level
        imagery only. 
        """

        self.print_exp_description()
        

        self.train_ds = dataset.MiniAutoArborist(
                root_path, 
                csv_path, 
                "train", 
                normalize=normalize, 
                aerial_resize=aerial_resize, 
                sv_resize=sv_resize
            )
        
        self.test_ds = dataset.MiniAutoArborist(
                root_path, 
                csv_path, 
                "test", 
                normalize=normalize, 
                aerial_resize=aerial_resize, 
                sv_resize=sv_resize
            )

        self.trainloader = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_ds, batch_size=1, shuffle=False)

        print("Length of trainset: ", len(self.trainloader))
        print("Length of testset: ", len(self.testloader))


        self.num_classes = get_num_classes(label_map_json_file)

        self.model = self.construct_model(num_classes=self.num_classes)
        

        # device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Experiment running on device {self.device}") 
        self.model.to(self.device)


        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()


        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = logger,
            verbose = True
        )
    

    def print_exp_description(self):
        print(f"Running StreetLevelExperiment. Run name: {experiment_name}")


    def construct_model(self, weights=None, num_classes=10):
        print(f"Initializing model with {self.num_classes} num classes")
        model = torchvision.models.resnet18(weights=weights)
        d = model.fc.in_features
        model.fc = nn.Linear(d, num_classes)
        return model


    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        sv_image, aerial_image, y = batch
        sv_image = sv_image.to(self.device)
        y = y.to(self.device)
        logits = self.model(sv_image)
        loss_val = self.criterion(logits, y)
        acc = tp_metrics.calculate_accuracy(logits, y)
        metrics_dict = {'loss': loss_val, 'accuracy': acc}
        return metrics_dict
    

    def test(self):
        self.model.eval()
        with torch.no_grad():
            acc = tp_metrics.Accuracy()
            tqdm_loader = tqdm(self.testloader)
            for batch in tqdm_loader:
                sv_image, aerial_image, y = batch
                sv_image = sv_image.to(self.device)
                y = y.to(self.device)
                logits = self.model(sv_image)
                acc.update(logits, y)
                tqdm_loader.set_description(f'Accuracy: {acc.get()}')
            
            final_acc = acc.get()
            print(f'Test accuracy: {final_acc}')
            acc.reset()

        return final_acc


    def on_epoch_end(self):
        test_acc = self.test()
        if self.wandb_logger is not None:
            self.wandb_logger.log({"Test accuracy": test_acc})

    
    def on_run_end(self):
        self.save_weights()


    def on_epoch_start(self):
        self.model.train()



class AerialExperiment(StreetLevelExperiment):
    """
    In this baseline experiment, we train a resnet-18 model
    to predict the species of the tree using only aerial
    imagery only. 
    """
    def __init__(self):
        super().__init__()

    

    def print_exp_description(self):
        print(f"Running AerialExperiment. Run name: {experiment_name}")


    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        sv_image, aerial_image, y = batch
        aerial_image = aerial_image.to(self.device)
        y = y.to(self.device)
        logits = self.model(aerial_image)
        loss_val = self.criterion(logits, y)
        acc = tp_metrics.calculate_accuracy(logits, y)
        metrics_dict = {'loss': loss_val, 'accuracy': acc}
        return metrics_dict
    

    def test(self):
        self.model.eval()
        with torch.no_grad():
            acc = tp_metrics.Accuracy()
            tqdm_loader = tqdm(self.testloader)
            for batch in tqdm_loader:
                sv_image, aerial_image, y = batch
                aerial_image = aerial_image.to(self.device)
                y = y.to(self.device)
                logits = self.model(aerial_image)
                acc.update(logits, y)
                tqdm_loader.set_description(f'Accuracy: {acc.get()}')
            
            final_acc = acc.get()
            print(f'Test accuracy: {final_acc}')
            acc.reset()

        return final_acc




class ConcatModel(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        
        # hyperparameters 
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Base encoder model f(x)
        self.encoder = torchvision.models.resnet18(num_classes = self.latent_dim)

        self.linear_head = nn.Linear(self.latent_dim * 2, self.num_classes)


    def forward(self, x1, x2):
        latent_1 = self.encoder(x1)
        latent_2 = self.encoder(x2)
        concatenated = torch.cat((latent_1, latent_2), dim=1)
        return self.linear_head(concatenated)



class ConcatExperiment(torchplate.experiment.Experiment):
    """
    Use resnet18 to project both aerial and sv image
    into 512-dimensional vectors. Concat them into
    1024-vector. Pass this into two layer MLP.  
    """
    def __init__(self): 

        self.print_exp_description()
        
        self.train_ds = dataset.MiniAutoArborist(
                root_path, 
                csv_path, 
                "train", 
                normalize=normalize, 
                aerial_resize=aerial_resize, 
                sv_resize=sv_resize
            )
        
        self.test_ds = dataset.MiniAutoArborist(
                root_path, 
                csv_path, 
                "test", 
                normalize=normalize, 
                aerial_resize=aerial_resize, 
                sv_resize=sv_resize
            )

        self.trainloader = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_ds, batch_size=1, shuffle=False)

        print("Length of trainset: ", len(self.trainloader))
        print("Length of testset: ", len(self.testloader))


        # model 
        self.hidden_dim = 512
        self.num_classes = get_num_classes(label_map_json_file)
        self.model = ConcatModel(self.hidden_dim, self.num_classes)

        # device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Experiment running on device {self.device}") 
        self.model.to(self.device)


        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()


        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = logger,
            verbose = True
        )
    

    def print_exp_description(self):
        print(f"Running ConcatExperiment. Run name: {experiment_name}")


    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        sv_image, aerial_image, y = batch
        aerial_image = aerial_image.to(self.device)
        sv_image = sv_image.to(self.device)
        y = y.to(self.device)
        logits = self.model(sv_image, aerial_image)
        loss_val = self.criterion(logits, y)
        acc = tp_metrics.calculate_accuracy(logits, y)
        metrics_dict = {'loss': loss_val, 'accuracy': acc}
        return metrics_dict
    

    def test(self):
        self.model.eval()
        with torch.no_grad():
            acc = tp_metrics.Accuracy()
            tqdm_loader = tqdm(self.testloader)
            for batch in tqdm_loader:
                sv_image, aerial_image, y = batch
                aerial_image = aerial_image.to(self.device)
                sv_image = sv_image.to(self.device)
                y = y.to(self.device)
                logits = self.model(sv_image, aerial_image)                
                acc.update(logits, y)
                tqdm_loader.set_description(f'Accuracy: {acc.get()}')
            
            final_acc = acc.get()
            print(f'Test accuracy: {final_acc}')
            acc.reset()

        return final_acc


    def on_epoch_end(self):
        test_acc = self.test()
        if self.wandb_logger is not None:
            self.wandb_logger.log({"Test accuracy": test_acc})

    
    def on_run_end(self):
        self.save_weights()


    def on_epoch_start(self):
        self.model.train()

    


# ----------------- Runner ----------------- #


def main(args):
    # update globals from cli args 
    if args.name is not None:
        global experiment_name 
        experiment_name = args.name + "-" + experiment_name

    if args.log:
        global logger
        logger = wandb.init(project = "auto-arborist-vancouver", name = experiment_name)
        
    
    # train 
    if args.experiment == "aerial":
        exp = AerialExperiment()
    elif args.experiment == "street":
        exp = StreetLevelExperiment()
    elif args.experiment == "concat":
        exp = ConcatExperiment()
    else:
        exp = StreetLevelExperiment()
   
    exp.train(num_epochs=args.epochs, display_batch_loss=args.batch_loss)
    exp.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-name", type=str, help='Experiment name for wandb logging purposes.', default=None) 
    parser.add_argument("-experiment", type=str, help='Which baseline do you want to run? Options are (aerial, street)', default="street") 
    parser.add_argument('-epochs', type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument('-log', action='store_true', help='Do you want to log this run to wandb?', default=False)
    parser.add_argument('-batch_loss', action='store_true', help='Do you want to display loss at each batch in the training bar?')
    args = parser.parse_args()
    main(args)