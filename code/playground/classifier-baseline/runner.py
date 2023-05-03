"""
File: runner.py
------------------
Runner script to train a model. This is the script that calls the other modules.
Execute this one to execute the program! See argparse help for more info on commands. 
Basically, run: 

$ python runner.py -config <config_class_name> -experiment <experiment_class_name> -train -test -num_epochs <num_epochs> -grad_accum <grad_accum>

the config class name specifies the experiment you are running and the parameters you are using. 
"""


import argparse
import warnings
import pdb 
import configs
import experiments


def main(args):
    # load config 
    if args.config is None:
        config_class = 'BaseConfig'
    else:
        config_class = args.config

    cfg = getattr(configs, config_class)

    # checkpoint 
    if args.checkpoint is not None:
        model_checkpoint_ = args.checkpoint
    else:
        model_checkpoint_ = None

    # load experiment 
    assert cfg.experiment_class is not None, "Must specify experiment class in config."
    exp = cfg.experiment_class(cfg, model_checkpoint = model_checkpoint_)

	# train the model
    if args.train:
        exp.train(args.num_epochs, gradient_accumulate_every_n_batches=args.grad_accum, display_batch_loss=args.display_batch_loss)
    
    # test the model
    if args.test:
        exp.test()



if __name__ == '__main__':
    # example run 
    # python runner.py -e BaseExp -c BaseConfig -train -test -num_epochs 10 -grad_accum 1
    # configure args 
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-experiment", type=str, help='specify experiment.py class to use.') 
    parser.add_argument("-config", type=str, help='specify config.py class to use.') 
    parser.add_argument('-train', action='store_true', help='Do you want to train the model?', default=True)
    parser.add_argument('-test', action='store_true', help='Do you want to test the model?', default=False)
    parser.add_argument("-checkpoint", type=str, help='Optionally specify model checkpoint to start with.') 
    parser.add_argument('-num_epochs', type=int, help='Number of epochs to train for.', default=10)
    parser.add_argument('-grad_accum', type=int, help='Number of gradient accumulations per batch.', default=1)
    parser.add_argument('-display_batch_loss', action='store_true', help='Display batch loss during training at each step.', default=True)
    args = parser.parse_args()
    main(args)
	