"""
File: configs.py 
----------------------
Specifies experiment configuration parameters as
python classes. Pass one of these to the 
experiment class in experiments.py! This makes it
easier to specify things like hyperparameters. 
"""



class BaseConfig:
    experiment = experiments.BaseExp
    trainloader, test_set = data.get_dataloaders()
    logger = None