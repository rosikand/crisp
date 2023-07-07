"""
File: crisp.py
------------------
Our implementation of CLIP classes and functions for CRISP. 
Uses openclip code. 
"""


import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import pdb
import os
from rsbox import misc



# ----------------- Model class ----------------- #

VALID_ENCODERS = ["resnet18", "resnet50", "vit"]

class CrispModel(nn.Module):
    """
    Trainable PyTorch module for CLIP/CRISP pre-training. 
    Has two submodules:
    - image encoder 1 
    - image encoder 2
    User must pass in a PyTorch encoder module for each submodule. 
    """

    def __init__(self, encoder_name, embedding_dim=512, pretrained_weights=None):
        super().__init__()

        # assert encoder name is valid
        assert encoder_name in VALID_ENCODERS, f"encoder name {encoder_name} is not valid. Valid encoders are {VALID_ENCODERS}"
    
        # construct the encoder modules
        self.ground_image_encoder = self.construct_encoder(encoder_name, embedding_dim, pretrained_weights)
        self.remote_sensing_encoder = self.construct_encoder(encoder_name, embedding_dim, pretrained_weights)

        # extra 
        self.embedding_dim = embedding_dim

        # see https://github.com/mlfoundations/open_clip/blob/6ee59e10510ec9761b8b9871b9fd1eeb8e28627d/src/open_clip/model.py#L202
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def construct_encoder(self, encoder_name, embedding_dim, pretrained_weights=None):
        """
        Construct the encoder module. 
        Specify the encoder name and optionally pass in pretrained weights
        name from torchvision docs. 

        scratched: 
            base_model = torchvision.models.resnet50(weights=pretrained_weights)
            base_layers = list(base_model.children())[:-1]
            projection_layer = torch.nn.Linear(base_model.fc.in_features, embedding_dim)
            modules = base_layers + [projection_layer]
            encoder = torch.nn.Sequential(*modules)
            return encoder
        """
        if encoder_name == "resnet50":
            model = torchvision.models.resnet50(weights=pretrained_weights)
            d = model.fc.in_features
            model.fc = nn.Linear(d, embedding_dim)
            return model
        elif encoder_name == "resnet18":
            model = torchvision.models.resnet18(weights=pretrained_weights)
            d = model.fc.in_features
            model.fc = nn.Linear(d, embedding_dim)
            return model
        else:
            print("Encoder name invalid... defaulting to resnet18...")
            model = torchvision.models.resnet18(weights=pretrained_weights)
            d = model.fc.in_features
            model.fc = nn.Linear(d, embedding_dim)
            return model
        

    def lock(self, encoder_to_lock):
        """
        Lock certain parameters of the model so that it cannot be trained.
        See CoCa paper and https://github.com/mlfoundations/open_clip/blob/
        6ee59e10510ec9761b8b9871b9fd1eeb8e28627d/src/open_clip/modified_resnet.py#L154.  
        """
        if encoder_to_lock == "ground_image":
            # freeze the ground image encoder parameters
            for param in self.ground_image_encoder.parameters():
                param.requires_grad = False
        elif encoder_to_lock == "remote_sensing":
            # freeze the remote sensing encoder parameters
            for param in self.remote_sensing_encoder.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"encoder_to_lock must be either 'ground_image' or 'remote_sensing'")
        

    def encode_remote_sensing_image(self, x):
        return self.remote_sensing_encoder(x)
    

    def encode_ground_image(self, x):
        return self.ground_image_encoder(x)

    
    def save_rs_encoder_weights(self, save_path=None):
        if save_path is None:
            if not os.path.exists("savedencoders"):
                os.makedirs("savedencoders")
            save_path = "savedencoders/" + "remotesensing-" + misc.timestamp() + ".pth"
        torch.save(self.remote_sensing_encoder.state_dict(), save_path)
        print("Remote sensing encoder weights saved at: " + str(save_path))

    def save_gl_encoder_weights(self, save_path=None):
        if save_path is None:
            if not os.path.exists("savedencoders"):
                os.makedirs("savedencoders")
            save_path = "savedencoders/" + "ground-" + misc.timestamp() + ".pth"
        torch.save(self.ground_image_encoder.state_dict(), save_path)
        print("Ground image encoder weights saved at: " + str(save_path))


    def load_remote_sensing_encoder_weights(self, encoder_path):
        """
        If you have weights for the remote sensing encoder, load them here. 
        """
        self.remote_sensing_encoder.load_state_dict(torch.load(encoder_path))
        print(f"Successfully loaded remote sensing encoder weights from {encoder_path}")

    
    def load_ground_image_encoder_weights(self, encoder_path):
        """
        If you have weights for the ground image encoder, load them here. 
        """
        self.ground_image_encoder.load_state_dict(torch.load(encoder_path))
        print(f"Successfully loaded ground image encoder weights from {encoder_path}")


    def cosine_similarity_logits(self, ground_image, remote_sensing_image):
        """
        Maps images into latent space and computes cosine similarity between them as the logits. 
        We also have this in the loss. 
        Gets the CLIP matrix (one for each image input)
        """
        # see https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py#LL362C9-L362C9. 
        
        # get latent features 
        ground_image_latent = self.encode_ground_image(ground_image)
        remote_sensing_latent = self.encode_remote_sensing_image(remote_sensing_image)
        
        # normalized features
        ground_image_latent = ground_image_latent / ground_image_latent.norm(dim=1, keepdim=True)
        remote_sensing_latent = remote_sensing_latent / remote_sensing_latent.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_ground_image = logit_scale * ground_image_latent @ remote_sensing_latent.t()
        logits_per_remote_sensing_image = logits_per_ground_image.t()

        return logits_per_ground_image, logits_per_remote_sensing_image
    

    def alternate_cosine_similarity_logits(self, ground_image, remote_sensing_image):
        """
        for debugging. the first version uses openai official repo code.
        This uses https://github.com/mlfoundations/open_clip/blob/6ee59e10510ec9761b8b9871b9fd1eeb8e28627d/src/open_clip/loss.py#L102. 
        Note: after some debugging, this function returns the same value 
        (Pdb) cos_sim_1
        (tensor([[-0.7614]], grad_fn=<MmBackward0>), tensor([[-0.7614]], grad_fn=<TBackward0>))
        (Pdb) cos_sim_1.shape
        *** AttributeError: 'tuple' object has no attribute 'shape'
        (Pdb) cos_sim_2
        (tensor([[-0.7614]], grad_fn=<MmBackward0>), tensor([[-0.7614]], grad_fn=<MmBackward0>))
        (Pdb) cos_sim_1 == cos_sim_2
        True
        """
        # get latent features 
        ground_image_latent = self.encode_ground_image(ground_image)
        remote_sensing_latent = self.encode_remote_sensing_image(remote_sensing_image)
        
        # normalized features
        ground_image_latent = ground_image_latent / ground_image_latent.norm(dim=1, keepdim=True)
        remote_sensing_latent = remote_sensing_latent / remote_sensing_latent.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_ground_image = logit_scale * ground_image_latent @ remote_sensing_latent.T
        logits_per_remote_sensing_image = logit_scale * remote_sensing_latent @ ground_image_latent.T
        
        return logits_per_ground_image, logits_per_remote_sensing_image
    
    
    def forward(self, ground_image, remote_sensing_image):
        """
        Forward pass through the model. Produces cosine sim logits. 
        """
        gr_logits, rs_logits = self.cosine_similarity_logits(ground_image, remote_sensing_image)
        return gr_logits, rs_logits



# ----------------- Loss class ----------------- #


class ClipLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth
        return torch.arange(num_logits, device=device, dtype=torch.long)
        # my confusion is that this will be [0,1,..., batch-size] but why this the label? 
        # I think its cuz you are prediciting which cosine sim score goes with which image-image pair in the batch. 
        # that way, you are optimizing for the diagonal of the matrix to be the highest since i == j (positive pair). 


    def forward(self, logits_per_ground_image, logits_per_remote_sensing_image):
        device = logits_per_ground_image.device

        labels = self.get_ground_truth(device, logits_per_ground_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_ground_image, labels) +
            F.cross_entropy(logits_per_remote_sensing_image, labels)
        ) / 2

        return total_loss