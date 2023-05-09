"""
File: playground.py
------------------
For testing out code snippets.
"""


import torch
import numpy as np 
import crisp 
import pdb


# img_1 = ml.get_img(resize=True, size=(128, 128))
# img_2 = ml.get_img(resize=True, size=(224, 224)) / 1.1

# img_1 = torch.tensor(img_1, dtype=torch.float)
# img_2 = torch.tensor(img_2, dtype=torch.float)
# # unsqueeze to add batch dimension
# img_1 = img_1.unsqueeze(0)
# img_2 = img_2.unsqueeze(0)


# # batches 
# gr_batch = torch.randn(10, 3, 128, 128)
# rs_batch = torch.randn(10, 3, 224, 224)

# model = crisp.CrispModel(
#     encoder_name = "resnet50",
#     embedding_dim=512, 
#     pretrained_weights=None
#     )


# # gr_img_emb, rs_img_emb, logit_scale = model(gr_batch, rs_batch)
# logits_per_ground_image, logits_per_remote_sensing_image = model.cosine_similarity_logits(gr_batch, rs_batch)
# # cos_sim_2 = model.alternate_cosine_similarity_logits(gr_batch, rs_batch)

# # loss 
# crisp_loss = crisp.ClipLoss()
# loss = crisp_loss(logits_per_ground_image, logits_per_remote_sensing_image)
# print(loss)



# show that crisp implementation functions correctly 

# batches 
gr_batch = torch.randn(10, 3, 128, 128)
rs_batch = torch.randn(10, 3, 224, 224)

model = crisp.CrispModel(
    encoder_name = "resnet50",
    embedding_dim=512, 
    pretrained_weights=None
    )

gr_logits, rs_logits = model(gr_batch, rs_batch)

crisp_loss = crisp.ClipLoss()
loss = crisp_loss(gr_logits, rs_logits)
print(loss)
