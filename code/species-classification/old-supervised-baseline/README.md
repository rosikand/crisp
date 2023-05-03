# Species Classification 

Program to train and test species classification experiments. 

> `(input, output)` is `(INaturalist ground level image, species label)`

## Directory structure 

- `configs.py`: configuration classes which define the parameters that are inputted into the experiment class. 
- `experiments.py`: define experiment classes here. Specifies the main logic behind each run (i.e., train/test loops, model, new methods to try). Each new thing we want to try could be an experiment class. 
- `models.py`: define NN architectures here. 
- `runner.py`: runner script to handle all of the setup logic. For example, takes in CLI args which specify the config class and experiment class and then runs the `.train`/`.test` methods. Execute this to run the program! 


To run the program, run the `runner.py` script which handles all of the configuration logic.  

```
python3 runner.py -config <config_class_name>
```

More CLI args can be specified (see argparse in `runner.py`): 

```
python3 runner.py -config <config_class_name> -experiment <experiment_class_name> -train -test -num_epochs <num_epochs> -grad_accum <grad_accum>
```

