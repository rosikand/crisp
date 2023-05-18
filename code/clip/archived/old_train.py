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
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pdb
from rsbox import ml, misc
import datasets 




class CrispExperiment(torchplate.experiment.Experiment):
    """
    Base crisp experiment class which inherits from torchplate.experiment.Experiment.  
    """
    
    def __init__(self, model_checkpoint=None):
        print("Initializing base crisp experiment...")

        # vars 
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
        self.model = crisp.CrispModel(
            encoder_name = "resnet50",
            embedding_dim = 512,  # eventually put all this in cfg. 
            pretrained_weights = None
        )
        self.model.to(self.device)

        if model_checkpoint is not None:
            self.load_weights(model_checkpoint)

        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = crisp.ClipLoss()

        # extra stuff 
        self.num_to_validate = self.cfg.num_to_validate
        self.num_to_test = self.cfg.num_to_test

        self.num_missed_dur_train = 0
        self.num_missed_dur_val = 0
        self.num_missed_dur_test = 0

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            save_weights_every_n_epochs = None, 
            wandb_logger = None,
            verbose = True,
            experiment_name = misc.timestamp()
        )




class CrispExperiment(torchplate.experiment.Experiment):
    """
    Base crisp experiment class which inherits from torchplate.experiment.Experiment.  
    """
    
    def __init__(self, config=None, model_checkpoint=None):
        print("Initializing base crisp experiment...")
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
        self.model = crisp.CrispModel(
            encoder_name = "resnet50",
            embedding_dim = 512,  # eventually put all this in cfg. 
            pretrained_weights = None
        )
        self.model.to(self.device)

        if model_checkpoint is not None:
            self.load_weights(model_checkpoint)

        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = crisp.ClipLoss()

        # extra stuff 
        self.num_to_validate = self.cfg.num_to_validate
        self.num_to_test = self.cfg.num_to_test

        self.num_missed_dur_train = 0
        self.num_missed_dur_val = 0
        self.num_missed_dur_test = 0

        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            save_weights_every_n_epochs = None, 
            wandb_logger = None,
            verbose = True,
            experiment_name = misc.timestamp()
        )
    

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
    

    def evaluate(self, batch):

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        if self.is_negative(x, y):
            print("Train data point file not found, skipping batch...")
            self.num_missed_dur_train += 1
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
                print("Val data file not found, skipping batch...")
                self.num_missed_dur_val += 1
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
                print("Test data file not found, skipping batch...")
                self.num_missed_dur_test += 1
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
        print("-------- (Training complete!) --------")
        print(f"Number of missed train batches: {self.num_missed_dur_train}")
        print(f"Number of missed val batches: {self.num_missed_dur_val}")
        print(f"Number of missed test batches: {self.num_missed_dur_test}")
        print("--------------------------------------")




def main(args):
    
    # run the experiment
    exp = CrispExperiment()
    exp.train(num_epochs=args.epochs, display_batch_loss=True)
    exp.test()



if __name__ == "__main__":
    """
    Example run: 
    $ python run.py --dataset_path cubs-10-pretrain --imagenet_pretrained --epochs 11 --classes 10
    """
    
    # Create the argument parser

    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument("--dataset_path", type=str, default="../../../datasets/cubs-pretrain-10/classification")
    parser.add_argument("--imagenet_pretrained", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=11)
    parser.add_argument("--classes", type=int, default=10)

    # Parse the arguments
    args = parser.parse_args()

    # call main function 
    main(args)