"""
File: probe.py
------------------
Option (1) of fine-tuning: train a linear probe.
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
import os
import argparse
import wandb


# ----------------- Global config vars ----------------- #

batch_size = 256
logger = None
experiment_name = "linearprobefinetuning" + "-" + misc.timestamp()

# paths 
train_csv_file_path = "/mnt/disks/mountDir/metadata/filtered_train.csv"
train_images_dir_path = "/mnt/disks/mountDir/train/images"
train_rs_dir_path = "/mnt/disks/crisp/mountDir/train/remote_sensing"

validation_csv_file_path = "/mnt/disks/mountDir/metadata/filtered_validation.csv"
validation_images_dir_path = "/mnt/disks/mountDir/validation/images"
validation_rs_dir_path = "/mnt/disks/crisp/mountDir/validation/remote_sensing"

test_csv_file_path = "/mnt/disks/mountDir/metadata/filtered_test.csv"
test_images_dir_path = "/mnt/disks/mountDir/test/images"
test_rs_dir_path = "/mnt/disks/crisp/mountDir/test/remote_sensing"

label_map_path = "/mnt/disks/mountDir/metadata/label_map.json"



# ----------------- Training Experiments ----------------- #


class FineTuneProbe(torchplate.experiment.Experiment):
     
    def __init__(self, gl_encoder_path, rs_encoder_path):
        print("Initializing base crisp linear probe fine-tuning experiment...")


        # vars 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        print(f"Using device: {self.device}")

        # data 
        ds = data.CrispDataset(csv_path, images_dir_path, remote_sensing_dir_path)
        self.trainloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


        # model initialization 
        self.model = torchvision.models.resnet18(weights=None)
        for param in self.model.parameters():
            param.requires_grad = False
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, self.num_classes)

        # load weights 

        self.model.to(self.device)

        # loss and optimizer 
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = logger,
            verbose = True,
        )


    def evaluate(self, batch):
        gl, rs = batch
        gl = gl.to(self.device)
        rs = rs.to(self.device)

        gl_logits, rs_logits = self.model(gl, rs)
        loss = self.criterion(gl_logits, rs_logits)

        return loss 


    def on_run_start(self):
        self.model.train()
        # just to test
        self.save_weights()
        self.model.save_gl_encoder_weights()
        self.model.save_rs_encoder_weights()

    def on_epoch_end(self):
        self.save_weights()
        self.model.save_gl_encoder_weights()
        self.model.save_rs_encoder_weights()
        print('--------------------------')

    def on_run_end(self):
        self.save_weights()
        self.model.save_gl_encoder_weights()
        self.model.save_rs_encoder_weights()
        print("Pre-training run complete!")


# ----------------- Runner ----------------- #


def main(args):
    # update globals from cli args 
    global experiment_name 
    if args.name is not None:
        experiment_name = args.name + "-" + experiment_name

    if args.log:
        global logger
        logger = wandb.init(project = "crisp197-crisp-pretraining", name = experiment_name)
    
    # train 
    exp = CrispExperiment()
    print(f"training for {args.epochs} epochs")
    exp.train(num_epochs=args.epochs, display_batch_loss=args.batch_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-name", type=str, help='Experiment name for wandb logging purposes.', default=None) 
    parser.add_argument('-epochs', type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument('-log', action='store_true', help='Do you want to log this run to wandb?', default=False)
    parser.add_argument('-batch_loss', action='store_true', help='Do you want to display loss at each batch in the training bar?', default=True)
    args = parser.parse_args()
    main(args)

