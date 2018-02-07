# Pytorch Implmementation of Dual Path Networks with Tweaks

This is a complete training example for Dual Path Networks (https://arxiv.org/abs/1707.01629) some other residual architctures.
Currently, only CIFAR10 is supported, but I may at some point extend this to various other datasets.

## Main contributions of this repo are:
* Several interesting tweaks to Dual Path Networks, some of which could potentially lead to improved performance. I will be writing these up in a blog post sometime over the coming weeks.
* Easy testing of different activation functions.
* Automated training of several models in succession.
* Complete logging of trained experiment in CSV files, as well as model checkpoints.

## How To Run:
python3 main_multi.py --epochs 10

Detailed instructions and more example training scripts can be found in the TRAINING_INSTRUCTIONS.md file along with detailed descriptions of the arguments

## Models and Model Configuration
* All of the main model architectures are stored in the **models** directory, each in their own separate file.
* Each model file must be registered in <code>models/\_\_init\_\_.py</code>
* For the moment, the actual model names are at the bottom of each model file, and must be typed in to the `model_list` array in main_multi.py, but I will eventually rewrite so that they can simply be selected as an argument during training.

## Activation Function Tweaks
* Several (but not all) of the models support using custom activation functions. The ones that do are noted in TRAINING_INSTRUCTIONS. 
* The file custom_activations.py includes my implementations of a few new activation functions from recent research that are not yet implemented in PyTorch. There are also a few experiments of my own, some of them promising, to be written up in a blog post soon.

## Logging and Checkpoints
* This project includes a reasonably informative progress bar in the terminal window.
* There is also logging of training data for each model trained, with historical records saved to csv files.
* The command line arguments that control this are detailed in TRAINING_INSTRUCTIONS.md

## Credits
* This code borrows extensively from https://github.com/kuangliu/pytorch-cifar and https://github.com/eladhoffer/convNet.pytorch - much thanks for the great projects!
