"""
File: train.py
------------------
Experiment trainer file for supervised baseline for species classification. 
"""


import torchplate
from torchplate import experiment
from torchplate import utils
from torchplate import metrics as tp_metrics
import sys
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
import pdb
import rsbox 
from rsbox import ml, misc
from torchvision import transforms
import dataset
import wandb


# ----------------- Global config vars ----------------- #

# paths 
train_csv_file_path = "/mnt/disks/mountDir/metadata/filtered_train.csv"
train_images_dir_path = "/mnt/disks/mountDir/train/images"
validation_csv_file_path = "/mnt/disks/mountDir/metadata/filtered_validation.csv"
validation_images_dir_path = "/mnt/disks/mountDir/validation/images"
test_csv_file_path = "/mnt/disks/mountDir/metadata/filtered_test.csv"
test_images_dir_path = "/mnt/disks/mountDir/test/images"
label_map_path = "/mnt/disks/mountDir/metadata/label_map.json"

# hyperparameters 
image_resize = (240, 180)
normalize = True
batch_size = 256

# misc. 
experiment_name = "default" + "-" + misc.timestamp()
logger = None   # don't change... use argparse argument 

print("batch_size: ", batch_size)
print("normalize?: ", normalize)
print("image_resize: ", image_resize)


# ----------------- Training Experiments ----------------- #


class BaselineExperiment(torchplate.experiment.Experiment):
    def __init__(self): 

        print(f"Running BaselineExperiment. Run name: {experiment_name}")

        # load the data 
        self.train_ds = dataset.INaturalistClassification(
                csv_file_path = train_csv_file_path,
                images_dir_path = train_images_dir_path,
                label_map_path = label_map_path,
                image_resize = image_resize,
                normalize = normalize
            )
    
        self.val_ds = dataset.INaturalistClassification(
                csv_file_path = validation_csv_file_path,
                images_dir_path = validation_images_dir_path,
                label_map_path = label_map_path,
                image_resize = image_resize,
                normalize = normalize
            )

        self.test_ds = dataset.INaturalistClassification(
                csv_file_path = test_csv_file_path,
                images_dir_path = test_images_dir_path,
                label_map_path = label_map_path,
                image_resize = image_resize,
                normalize = normalize
            )

        self.trainloader = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.valloader = torch.utils.data.DataLoader(self.val_ds, batch_size=1, shuffle=False)
        self.testloader = torch.utils.data.DataLoader(self.test_ds, batch_size=1, shuffle=False)

        self.num_classes = self.train_ds.num_classes

        # model 
        self.model = self.construct_model(num_classes=self.num_classes)
        
        # device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Experiment running on device {self.device}") 
        self.model.to(self.device)


        # training variables 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()


        # inherit from torchplate.experiment.Experiment and pass in
        # model, optimizer, and dataloader 
        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = logger,
            verbose = True
        )
    
    
    def construct_model(self, weights=None, num_classes=10):
        print(f"Initializing model with {self.num_classes} num classes")
        model = torchvision.models.resnet18(weights=weights)
        d = model.fc.in_features
        model.fc = nn.Linear(d, num_classes)
        return model


    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model(x)
        loss_val = self.criterion(logits, y)
        acc = tp_metrics.calculate_accuracy(logits, y)
        metrics_dict = {'loss': loss_val, 'accuracy': acc}
        return metrics_dict
    

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            acc = tp_metrics.Accuracy()
            tqdm_loader = tqdm(self.valloader)
            for batch in tqdm_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                acc.update(logits, y)
                tqdm_loader.set_description(f'Accuracy: {acc.get()}')
            
            final_acc = acc.get()
            print(f'Validation accuracy: {final_acc}')
            acc.reset()

        return final_acc


    def test(self):
        self.model.eval()
        with torch.no_grad():
            acc = tp_metrics.Accuracy()
            tqdm_loader = tqdm(self.testloader)
            for batch in tqdm_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                acc.update(logits, y)
                tqdm_loader.set_description(f'Accuracy: {acc.get()}')
            
            final_acc = acc.get()
            print(f'Test accuracy: {final_acc}')
            acc.reset()

        return final_acc


    def on_epoch_end(self):
        val_acc = self.validate()   
        test_acc = self.test()     
        if self.wandb_logger is not None:
            self.wandb_logger.log({"Validation accuracy": val_acc})
            self.wandb_logger.log({"Test accuracy": test_acc})
        self.save_weights()
        print('--------------------------')

    
    def on_run_end(self):
        self.save_weights()
        print('testing...')
        test_acc = self.test()
        if self.wandb_logger is not None:
            self.wandb_logger.log({"Test accuracy": test_acc})


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
        logger = wandb.init(project = "crisp197", name = experiment_name)
        
    
    # train 
    exp = BaselineExperiment()
    print(f"training for {args.epochs} epochs")
    exp.train(num_epochs=args.epochs, display_batch_loss=args.batch_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-name", type=str, help='Experiment name for wandb logging purposes.', default=None) 
    parser.add_argument('-epochs', type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument('-log', action='store_true', help='Do you want to log this run to wandb?', default=False)
    parser.add_argument('-batch_loss', action='store_true', help='Do you want to display loss at each batch in the training bar?')
    args = parser.parse_args()
    main(args)
