# Training Instructions

#### In the simplest, case, training can be started with:
python3 main_multi.py --epochs 10

#### To change the model that you want to train with:
* All the models are in the `models` directory.
* The list of models being trained can be found in the `main_multi.py` file in the "List of Models To Train" section.
* To add a model, just include it in the list as a string.
* The activation function can also be changed for the models that have "_act1" or "_act2" as part of their names (see "Activation Functions" below)
* Some examples:

`model_list = [
    'ResNet18()',
    'DPN92()',
    'DPN92_act1(activation=swish)',
    'ResNet_act2(activation=F.selu'),
	]`

#### Activation Functions
* Besides the native activation functions in PyTorch that can be added as arguments to some of the models, there are also several custom activation functions, some from recent research, and some of my own invention, that can be found in the `custome_activations.py` file.

## Training Arguments Reference

### General settings
--epochs
* The number of epochs to train 
* Default:	 30
* Must be an integer

--data-path
* The path where you store training data - e.g. cifar, mnist etc.
* Default:	 '../data'
* Must be a string with a valid filepath

--train-batch-size', type=int, default=128)
--test-batch-size', type=int, default=100)
* Self-explanatory

--seed
* Random seed for initializations
* Default is 999
* Must be an integer

### Hardware-related settings
--gpus
* Which specific gpus to use.
* The default is `None` because the model is set to automatically use all available GPUs. Using this setting will set the model to use only the specified gpus
* Must be a tuple - i.e. integers separated by commas

--threads
* The number of CPU threads for the data loader to use.
* Default is 4
* Can sometimes cause system resource errors if set too high.

--optimize-cudnn
* Default is `True`
* Generally fine to keep as `True`, but changing to `False` for many different short runs, or runs where input sizes are changing constantly could help speed things up if you are training multiple models and datasets over short runs. Not a setting that most people need to pay attention to.

### Optimization and learning rate settings for training

--optimizer
* Which optimizer to use. Currently only supports native PyTorch optimizers.
* Default is 'SGD'
* Must be a string containing the name of a valid PyTorch optimizer such as "Adam", "SGD", "RMSprop" etc

--lr
* Learning rate. 
* Default is 0.01
* Must be a float

--lr-stages
* The number of times to decrease the learning rate.
* Set to a number n to decrease the learning rate by factor of lr_increment every 1/n epochs')
* Must be an int

--lr-increment
* Used with --lr-stages. This is the amount to divide the learning rate by at each stage.
* Default is 10, but should probably be set much lower for long trainings with lots of stages.

--lr-momentum
* Learning rate momentum, if you need it.
* Default=None, but can use 0.9 etc.

--lr-decay
* This can be used instead of --lr-stages, but not together.
* Default is set to 1 - i.e. no decay. 
* Set to lower value such as .98 (i.e. 2% decay) to automatically decay each epoch.
* Must be a float

--lr-floor
* Minimum learning rate below which it cannot decline
* Default is 0.000001
* Must be a float

### Logging and checkpoint settings

--save-checkpoints
* Whether to save model checkpoints or not.
* Default is `False`

--checkpoint-dir
* Directory to save checkpoints in.
* Default is 'results/checkpoints'

--checkpoint-accuracy-threshold'
* The minimum test accuracy that the model must achieve before saving checkpoints.
* Default is 50.1
* Must be a float

--min-checkpoint-improvement
* The minimum improvement in test accuracy that the model must see before saving the next checkpoint.
* Default is 0.1
* Must be a float

--save-training-info
* Whether to save information about each training into csv files or not
* Default is `True`

--training-info-dir
* The directory into which training result csv files are saved
* Default is 'results/training_info'
