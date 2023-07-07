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
import os
import argparse
import wandb
import json


# ----------------- Global config vars ----------------- #

root_path = "../dataset/data"
csv_path = "../dataset/ds.csv"
label_map_json_file = '../dataset/label_map.json'
normalize = True
# aerial_resize = (512, 512)
# sv_resize = (512, 512)
aerial_resize = None
sv_resize = None
batch_size = 128
print("batch size: ", batch_size)
logger = None  
experiment_name = "crisp-pretraining" + "-" + misc.timestamp()
save_weights = True
encoder_architecture = 'resnet18'


# ----------------- Utilities ----------------- #

def get_num_classes(json_file_path):
    # takes in label map json path and returns num classes 
    with open(json_file_path, 'r') as file:
        label_map = json.load(file)

    num_keys = len(label_map.keys())

    return num_keys


# ----------------- Training Experiments ----------------- #


class CrispExperiment(torchplate.experiment.Experiment):
     
    def __init__(self, model_checkpoint=None):
        self.print_exp_description()

        self.train_ds = data.MiniAutoArborist(
                root_path, 
                csv_path, 
                "train", 
                normalize=normalize, 
                aerial_resize=aerial_resize, 
                sv_resize=sv_resize
            )
        
        self.trainloader = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)

        print("Length of trainset: ", len(self.trainloader) * batch_size)

        self.num_classes = get_num_classes(label_map_json_file)

        # vars 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        print(f"Using device: {self.device}")
        

        # model initialization 
        self.model = crisp.CrispModel(
            encoder_name = encoder_architecture,
            embedding_dim = 512, 
            pretrained_weights = None
        )
        self.model.to(self.device)

        # loss and optimizer 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = crisp.ClipLoss()


        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = logger,
            verbose = True,
        )


    def print_exp_description(self):
        print(f"Initializing base crisp pre-training experiment. Run name: {experiment_name}")


    def evaluate(self, batch):
        gl, rs, y = batch
        gl = gl.to(self.device)
        rs = rs.to(self.device)

        gl_logits, rs_logits = self.model(gl, rs)
        loss = self.criterion(gl_logits, rs_logits)

        return loss 


    def on_run_start(self):
        self.model.train()
        if save_weights:
            self.save_weights()
            self.model.save_gl_encoder_weights()
            self.model.save_rs_encoder_weights()


    def on_epoch_end(self):
#        if save_weights:
           # self.save_weights()
           # self.model.save_gl_encoder_weights()
           # self.model.save_rs_encoder_weights()
        print('--------------------------')


    def on_run_end(self):
        if save_weights:
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
        logger = wandb.init(project = "auto-arborist-vancouver-summer-2023", name = experiment_name)
    
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

