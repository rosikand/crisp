Two ways: 

- Use pre-trained encoders as feature extactors and train linear probe on top of this to serve as a probe. 
- Use pre-trained encoder weights as initialization weights and then fine-tune whole model. 

See "When transferring a pretrained model to a downstream task, two popular methods are full fine-tuning
(updating all the model parameters) and linear probing (updating only the last linear layer—the “head”)." of https://arxiv.org/pdf/2202.10054.pdf. 