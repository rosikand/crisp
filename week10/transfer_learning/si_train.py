"""
File: si_train.py
------------------
Fine-tuning where we use pre-training as weight initialization
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
import si_data
import wandb


# ----------------- Global config vars ----------------- #

pretrained_encoder_path = "/mnt/disks/crisp/mountDir/weights/savedencoders/ground-10-41-AM-Jun-13-2023.pth"

# paths 
train_csv_file_path = "/mnt/disks/crisp/mountDir/metadata/filtered_train.csv"
train_images_dir_path = "/mnt/disks/crisp/mountDir/train/images"

validation_csv_file_path = "/mnt/disks/crisp/mountDir/metadata/filtered_validation.csv"
validation_images_dir_path = "/mnt/disks/crisp/mountDir/validation/images"

test_csv_file_path = "/mnt/disks/crisp/mountDir/metadata/filtered_test.csv"
test_images_dir_path = "/mnt/disks/crisp/mountDir/test/images"

label_map_path = "/mnt/disks/crisp/mountDir/metadata/label_map.json"


# hyperparameters 
image_resize = (240, 180)
normalize = True
batch_size = 256
freeze_encoder = True

# misc. 
experiment_name = "finetuneweightinitTuesday" + "-" + misc.timestamp()
logger = None   # don't change... use argparse argument 

print("batch_size: ", batch_size)
print("normalize?: ", normalize)
print("image_resize: ", image_resize)
print("Resnet encoder weights frozen?: ", freeze_encoder)

# ----------------- Training Experiments ----------------- #


class BaselineExperiment(torchplate.experiment.Experiment):
    def __init__(self): 

        print(f"Running Weight init BaselineExperiment. Run name: {experiment_name}")

        # load the data 
        self.train_ds = si_data.INaturalistClassification(
                csv_file_path = train_csv_file_path,
                images_dir_path = train_images_dir_path,
                label_map_path = label_map_path,
                image_resize = image_resize,
                normalize = normalize
            )
    
        self.val_ds = si_data.INaturalistClassification(
                csv_file_path = validation_csv_file_path,
                images_dir_path = validation_images_dir_path,
                label_map_path = label_map_path,
                image_resize = image_resize,
                normalize = normalize
            )

        self.test_ds = si_data.INaturalistClassification(
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
        self.model = self.construct_pretrained_encoder(encoder_weights=pretrained_encoder_path)


        # path_ = "saved/11-56-PM-Jun-13-2023.pth"


        # edit 
        self.model.fc = nn.Sequential(
            self.model.fc,  # encoder output 
            nn.ReLU(inplace=True),
            nn.Linear(512, 2681),
            nn.ReLU(inplace=True),
            nn.Linear(2681, self.num_classes)
        )

        if freeze_encoder:
            print("freezing encoder weights...")
            for param in self.model.parameters():
                param.requires_grad = False
            
            for param in self.model.fc.parameters():
                param.requires_grad = True

        print(self.model)


        # device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Experiment running on device {self.device}") 
        self.model.to(self.device)


        # training variables 
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
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


    def construct_pretrained_encoder(self, encoder_weights, embedding_dim=512):
        model = torchvision.models.resnet18(weights=None)
        d = model.fc.in_features
        model.fc = nn.Linear(d, embedding_dim)
        model.load_state_dict(torch.load(encoder_weights))
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


    def test_top_k(self, k=5):
        k_accuracy_count = 0
        self.model.eval()
        with torch.no_grad():
            top_1_acc = tp_metrics.Accuracy()
            tqdm_loader = tqdm(self.testloader)
            for batch in tqdm_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model(x)
                top_1_acc.update(logits, y)
                _, top_pred_indices = torch.topk(F.softmax(logits, dim=1), k=k)
                top_pred_indices = top_pred_indices.tolist()[0] 
                if y.item() in top_pred_indices:
                    k_accuracy_count += 1

        

        final_acc = top_1_acc.get()
        print(f'Top-1 accuracy: {final_acc}')
        top_1_acc.reset()

        
        k_acc = k_accuracy_count/len(self.testloader)
        print(f'Top-{k} Test accuracy: {k_acc}')

        return final_acc, k_acc


    def on_epoch_end(self):
        val_acc = self.validate()   
        top_1_test, top_5_test = self.test_top_k(k=5)    
        if self.wandb_logger is not None:
            self.wandb_logger.log({"Validation accuracy": val_acc})
            self.wandb_logger.log({"Test top-1 accuracy": top_1_test})
            self.wandb_logger.log({"Test top-5 accuracy": top_5_test})
        self.save_weights()
        print('--------------------------')

    
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
        logger = wandb.init(project = "crisp197", name = experiment_name)
        
    
    # train 
    exp = BaselineExperiment()
    print(f"training for {args.epochs} epochs")
    # test_acc_5 = exp.test_top_k(k=5)
    exp.train(num_epochs=args.epochs, display_batch_loss=args.batch_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-name", type=str, help='Experiment name for wandb logging purposes.', default=None) 
    parser.add_argument('-epochs', type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument('-log', action='store_true', help='Do you want to log this run to wandb?', default=False)
    parser.add_argument('-batch_loss', action='store_true', help='Do you want to display loss at each batch in the training bar?')
    args = parser.parse_args()
    main(args)