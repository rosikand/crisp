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


# ----------------- Global config vars ----------------- #




train_folder = "/mnt/disks/proto/stl10_images_rotated/" 
test_folder = "/mnt/disks/proto/stl10_test"
batch_size = 64
logger = None
experiment_name = "resnet50" + "-" + misc.timestamp()
normalize = False
resize = None


# ----------------- Dataset ----------------- #


class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        if ".DS_Store" in self.classes:
            self.classes.remove(".DS_Store")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)
                if not os.path.isfile(image_path):
                    continue

                images.append((image_path, self.class_to_idx[class_name]))

        return images

        

    def __getitem__(self, index):
        image_path, label = self.images[index]
        image = ml.load_image(image_path, resize=resize, normalize=normalize) 

        if not torch.is_tensor(image):
            image = torch.tensor(image, dtype=torch.float)
        
        if not torch.is_tensor(label):
            label = torch.tensor(label)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.dtype != torch.float:
            image = image.to(torch.float)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)



# ----------------- Training Experiments ----------------- #


class ResNetExp(torchplate.experiment.Experiment):
    def __init__(self): 

        print(f"Running ResNetExp. Run name: {experiment_name}")
        
        self.num_classes = 10
        self.model = self.construct_model(num_classes=self.num_classes)
        
        # device 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Experiment running on device {self.device}") 
        self.model.to(self.device)


        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # data 
        self.transforms = None
        self.trainset = ImageClassificationDataset(train_folder, transform=self.transforms)
        self.testset = ImageClassificationDataset(test_folder, transform=self.transforms)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=1, shuffle=False)


        # inherit from torchplate.experiment.Experiment and pass in
        # model, optimizer, and dataloader 
        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = logger,
            verbose = True
        )
    
    
    def plot_samples(self):
        # plot the first few samples from the dataset to visualize 
        for i, (x, y) in enumerate(self.trainloader):
            ml.plot(x[0])  # plot first in batch  
            print(y[0])
            if i == 5:
                break   


    def construct_model(self, weights=None, num_classes=10):
        print(f"Initializing model with {self.num_classes} num classes")
        model = torchvision.models.resnet50(weights=weights)
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
        logger = wandb.init(project = "proto231n", name = experiment_name)
        
    
    # train 
    exp = ResNetExp()
    exp.train(num_epochs=args.epochs, display_batch_loss=args.batch_loss)
    exp.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-name", type=str, help='Experiment name for wandb logging purposes.', default=None) 
    parser.add_argument('-epochs', type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument('-log', action='store_true', help='Do you want to log this run to wandb?', default=False)
    parser.add_argument('-batch_loss', action='store_true', help='Do you want to display loss at each batch in the training bar?')
    args = parser.parse_args()
    main(args)