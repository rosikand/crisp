# test the dataset/dataloader 


import crisp
import data 
import os
import torch 
import rsbox 
from rsbox import ml 


base_path = "../../data/may17data"
csv_path = os.path.join(base_path, "filtered.csv")
images_dir_path = os.path.join(base_path, "images")
remote_sensing_dir_path = os.path.join(base_path, "remote_sensing")

ds = data.CrispDataset(csv_path, images_dir_path, remote_sensing_dir_path)
model = crisp.CrispModel(
            encoder_name = "resnet50",
            embedding_dim = 512, 
            pretrained_weights = None
        )


print("Model dtype: ", next(model.parameters()).dtype)

# import sys
# sys.exit() 

# gl, rs = ds[5]
# gl = torch.unsqueeze(gl, 0)
# rs = torch.unsqueeze(rs, 0)
# gr_logits, rs_logits = model(gl, rs)

crisp_loss = crisp.ClipLoss()
# loss = crisp_loss(gr_logits, rs_logits)



# dataloader 
dataloader = torch.utils.data.DataLoader(ds, batch_size=5, shuffle=False)

# iterate through dataloader and print out the first 10 losses 
for i, batch in enumerate(dataloader):
  gl, rs = batch

  gr_logits, rs_logits = model(gl, rs)
  loss = crisp_loss(gr_logits, rs_logits)
  print(loss)
  if i == 10:
    break