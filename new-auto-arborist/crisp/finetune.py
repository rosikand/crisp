"""
File: finetune.py
------------------
CRISP fine-tuning after pre-training. 
"""


import torchplate
from torchplate import experiment
from torchplate import utils
from torchplate import metrics as tp_metrics
import sys
import data
import torch
import crisp
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

root_path = "../dataset/data"
csv_path = "../dataset/ds.csv"
label_map_json_file = '../dataset/label_map.json'
normalize = True
# aerial_resize = (512, 512)
# sv_resize = (512, 512)
aerial_resize = None
sv_resize = None
batch_size = 16
print("batch size: ", batch_size)
logger = None  
experiment_name = "finetuning" + "-" + misc.timestamp()
freeze_gl_encoder = True
freeze_rs_encoder = True
encoder_architecture = 'resnet18'
embedding_dim = 512
combine_method = 'only_rs'
pretrained_gl_encoder_path = "savedencoders/ground-12-38-AM-Jun-28-2023.pth" 
pretrained_rs_encoder_path = "savedencoders/remotesensing-12-38-AM-Jun-28-2023.pth"
print("embed dim: ", embedding_dim)
print("combine method: ", combine_method)
print("encoder arch: ", encoder_architecture)
print("Normalize: ", normalize)
print("freeze_gl_encoder?: ", freeze_gl_encoder)
print("freeze_rs_encoder?: ", freeze_rs_encoder)



# ----------------- Utilities ----------------- #

def get_num_classes(json_file_path):
    # takes in label map json path and returns num classes 
    with open(json_file_path, 'r') as file:
        label_map = json.load(file)

    num_keys = len(label_map.keys())

    return num_keys


# -----------------Models ----------------- #


class LinearProbe(nn.Module):
    def __init__(self, 
                 encoder_name,
                 embedding_dim, 
                 num_classes, 
                 combine_method = "concat",
                 pretrained_gl_encoder_path = None, 
                 pretrained_rs_encoder_path = None,
                 freeze_gl_encoder = True,
                 freeze_rs_encoder = True
                ):
        super().__init__()

        valid_combine_methods = ['add', 'concat', 'only_gl', 'only_rs']
        assert combine_method in valid_combine_methods
        self.combine_method = combine_method


        if combine_method == "only_gl" or combine_method == "only_rs":
            self.uses_both_images = False
        else:
            self.uses_both_images = True

        # encoders 
        self.crisp_model = crisp.CrispModel(
            encoder_name = encoder_name,
            embedding_dim = embedding_dim
        )

        # load pre-trained weights (if provided)
        if pretrained_gl_encoder_path is not None:
            self.crisp_model.load_ground_image_encoder_weights(pretrained_gl_encoder_path)
        if pretrained_rs_encoder_path is not None:
            self.crisp_model.load_remote_sensing_encoder_weights(pretrained_rs_encoder_path)

        # freeze 
        if freeze_gl_encoder:
            self.crisp_model.lock("ground_image")
        if freeze_rs_encoder:
            self.crisp_model.lock("remote_sensing")

        # linear head 
        if combine_method == "concat":
            self.linear_head = nn.Linear(embedding_dim * 2, num_classes)
        else:
            self.linear_head = nn.Linear(embedding_dim, num_classes)

    
    def forward(self, gl=None, rs=None):
        # note: user must provide named arguments 
        if self.uses_both_images:
            ground_image_latent = self.crisp_model.encode_ground_image(gl)
            remote_sensing_latent = self.crisp_model.encode_remote_sensing_image(rs)
            if self.combine_method == "concat":
                concatenated = torch.cat((ground_image_latent, remote_sensing_latent), dim=1)
                return self.linear_head(concatenated)
        elif self.combine_method == "only_gl":
            ground_image_latent = self.crisp_model.encode_ground_image(gl)
            return self.linear_head(ground_image_latent)
        elif self.combine_method == "only_rs":
            remote_sensing_latent = self.crisp_model.encode_remote_sensing_image(rs)
            return self.linear_head(remote_sensing_latent)

    

# ----------------- Training Experiments ----------------- #


class BaseFineTuneExperiment(torchplate.experiment.Experiment):
    """
    Use resnet18 to project both aerial and sv image
    into 512-dimensional vectors. Concat them into
    1024-vector. Pass this into two layer MLP.  
    """
    def __init__(self): 

        self.print_exp_description()
        
        self.train_ds = data.MiniAutoArborist(
                root_path, 
                csv_path, 
                "train", 
                normalize=normalize, 
                aerial_resize=aerial_resize, 
                sv_resize=sv_resize
            )
        
        self.test_ds = data.MiniAutoArborist(
                root_path, 
                csv_path, 
                "test", 
                normalize=normalize, 
                aerial_resize=aerial_resize, 
                sv_resize=sv_resize
            )

        self.trainloader = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.test_ds, batch_size=1, shuffle=False)

        print("Length of trainset: ", len(self.trainloader) * batch_size)
        print("Length of testset: ", len(self.testloader))


        # model 
        self.num_classes = get_num_classes(label_map_json_file)
        self.model = LinearProbe(
                                    encoder_name=encoder_architecture,
                                    embedding_dim=embedding_dim, 
                                    num_classes=self.num_classes, 
                                    combine_method = combine_method,
                                    pretrained_gl_encoder_path = pretrained_gl_encoder_path, 
                                    pretrained_rs_encoder_path = pretrained_rs_encoder_path,
                                    freeze_gl_encoder = freeze_gl_encoder,
                                    freeze_rs_encoder = freeze_rs_encoder
                                )

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
        print(f"Running BaseFineTuneExperiment. Run name: {experiment_name}")


    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        sv_image, aerial_image, y = batch
        aerial_image = aerial_image.to(self.device)
        sv_image = sv_image.to(self.device)
        y = y.to(self.device)
        logits = self.model(gl=sv_image, rs=aerial_image)
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
                logits = self.model(gl=sv_image, rs=aerial_image)                
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
        logger = wandb.init(project = "auto-arborist-vancouver-summer-2023", name = experiment_name)
        
    
    # train 
    if args.experiment == "base":
        exp = BaseFineTuneExperiment()
    else:
        exp = BaseFineTuneExperiment()
   
    exp.train(num_epochs=args.epochs, display_batch_loss=args.batch_loss)
    exp.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-name", type=str, help='Experiment name for wandb logging purposes.', default=None) 
    parser.add_argument("-experiment", type=str, help='Which baseline do you want to run? Options are (base)', default="base") 
    parser.add_argument('-epochs', type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument('-log', action='store_true', help='Do you want to log this run to wandb?', default=False)
    parser.add_argument('-batch_loss', action='store_true', help='Do you want to display loss at each batch in the training bar?')
    args = parser.parse_args()
    main(args)
