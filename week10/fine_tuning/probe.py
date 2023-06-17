"""
File: probe.py
------------------
Option (1) of fine-tuning: train a linear probe.
"""


import argparse
import torchplate
from torchplate import experiment
from torchplate import utils
from torchplate import metrics as tp_metrics
from tqdm import tqdm
import torch
import dataset
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
embedding_dim = 512
logger = None
experiment_name = "linearprobefinetuning" + "-" + misc.timestamp()

# paths 
train_csv_file_path = "/mnt/disks/crisp/mountDir/metadata/filtered_train.csv"
train_images_dir_path = "/mnt/disks/crisp/mountDir/train/images"
train_rs_dir_path = "/mnt/disks/crisp/mountDir/train/remote_sensing"

validation_csv_file_path = "/mnt/disks/crisp/mountDir/metadata/filtered_validation.csv"
validation_images_dir_path = "/mnt/disks/crisp/mountDir/validation/images"
validation_rs_dir_path = "/mnt/disks/crisp/mountDir/validation/remote_sensing"

test_csv_file_path = "/mnt/disks/crisp/mountDir/metadata/filtered_test.csv"
test_images_dir_path = "/mnt/disks/crisp/mountDir/test/images"
test_rs_dir_path = "/mnt/disks/crisp/mountDir/test/remote_sensing"

label_map_path = "/mnt/disks/crisp/mountDir/metadata/label_map.json"


gl_enc_path = "/mnt/disks/crisp/mountDir/weights/savedencoders/ground-10-41-AM-Jun-13-2023.pth"
rs_enc_path = "/mnt/disks/crisp/mountDir/weights/savedencoders/remotesensing-10-41-AM-Jun-13-2023.pth"


# ----------------- Training Experiments ----------------- #


class FineTuneProbe(torchplate.experiment.Experiment):
     
    def __init__(self, gl_encoder_path, rs_encoder_path):
        print("Initializing base crisp linear probe fine-tuning experiment...")


        # vars 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        print(f"Using device: {self.device}")
        self.embedding_dim = embedding_dim

        # data 
        self.train_ds = dataset.FineTuningCrispDataset(
            train_csv_file_path, 
            train_images_dir_path, 
            train_rs_dir_path, 
            label_map_path, 
            "train"
        )
        self.val_ds = dataset.FineTuningCrispDataset(
            validation_csv_file_path, 
            validation_images_dir_path, 
            validation_rs_dir_path, 
            label_map_path, 
            "validation"
        )
        self.test_ds = dataset.FineTuningCrispDataset(
            test_csv_file_path, 
            test_images_dir_path, 
            test_rs_dir_path, 
            label_map_path, 
            "test"
        )
        

        self.trainloader = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.valloader = torch.utils.data.DataLoader(self.val_ds, batch_size=1, shuffle=False)
        self.testloader = torch.utils.data.DataLoader(self.test_ds, batch_size=1, shuffle=False)


        # model initialization 
        self.gl_encoder = self.construct_encoder(embedding_dim=self.embedding_dim)
        self.rs_encoder = self.construct_encoder(embedding_dim=self.embedding_dim)
        # load pre-trained checkpoints
        self.gl_encoder.load_state_dict(torch.load(gl_encoder_path))
        self.rs_encoder.load_state_dict(torch.load(rs_encoder_path))

        self.gl_encoder.to(self.device)
        self.rs_encoder.to(self.device)

        print("encoders with weights ready!")

        # linear probe 
        self.num_classes = self.train_ds.num_classes
        self.model = nn.Linear(self.embedding_dim * 2, self.num_classes) 
        print(self.model)
        self.model.to(self.device)


        # loss and optimizer 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = logger,
            verbose = True,
        )


    def construct_encoder(self, embedding_dim=512):
        model = torchvision.models.resnet18(weights=None)
        d = model.fc.in_features
        model.fc = nn.Linear(d, embedding_dim)
        return model



    def evaluate(self, batch):
        gl, rs, label = batch
        gl = gl.to(self.device)
        rs = rs.to(self.device)
        label = label.to(self.device)
        gl_latent = self.gl_encoder(gl)
        rs_latent = self.rs_encoder(rs)
        latent_concat = torch.cat((gl_latent, rs_latent), -1)
        logits = self.model(latent_concat)
        loss_val = self.criterion(logits, label)
        acc = tp_metrics.calculate_accuracy(logits, label)
        metrics_dict = {'loss': loss_val, 'accuracy': acc}
        return metrics_dict

    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            acc = tp_metrics.Accuracy()
            tqdm_loader = tqdm(self.valloader)
            for batch in tqdm_loader:
                gl, rs, label = batch
                gl = gl.to(self.device)
                rs = rs.to(self.device)
                label = label.to(self.device)
                gl_latent = self.gl_encoder(gl)
                rs_latent = self.rs_encoder(rs)
                latent_concat = torch.cat((gl_latent, rs_latent), -1)
                logits = self.model(latent_concat)
                acc.update(logits, label)
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
                gl, rs, label = batch
                gl = gl.to(self.device)
                rs = rs.to(self.device)
                label = label.to(self.device)
                gl_latent = self.gl_encoder(gl)
                rs_latent = self.rs_encoder(rs)
                latent_concat = torch.cat((gl_latent, rs_latent), -1)
                logits = self.model(latent_concat)
                acc.update(logits, label)
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

        

    def on_epoch_start(self):
        self.model.train()

    def on_run_start(self):
        val_acc = self.validate()   
        test_acc = self.test()
        print('---') 



# ----------------- Runner ----------------- #


def main(args):
    # update globals from cli args 
    global experiment_name 
    if args.name is not None:
        experiment_name = args.name + "-" + experiment_name

    if args.log:
        global logger
        logger = wandb.init(project = "crisp197-crisp-finetuning", name = experiment_name)
    
    # train 
    exp = FineTuneProbe(gl_enc_path, rs_enc_path)
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

