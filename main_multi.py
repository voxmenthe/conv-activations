################################################################
###################### PyTorch Imports ###########################
################################################################
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

################################################################
#################### General imports ###########################
################################################################

import os
import time
import argparse
from tqdm import tqdm

##################################################################
############ Imports that are part of this project ###############
##################################################################

from models import *
import utils
from imageprocessing import get_transform
from utils import progress_bar, print_model_params
from data_utils import get_cifar10, get_dataset

###################################################################
################# argparser for training options ##################
###################################################################

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--data-path', type=str, default='../data',help="path to store data in")
parser.add_argument('--train-batch-size', type=int, default=128)
parser.add_argument('--test-batch-size', type=int, default=100)
parser.add_argument('--seed', default=999, type=int, help='random seed (default: 999)')

######## Hardware-related settings #########
parser.add_argument('--gpus', default=None,
                    help='gpus used for training set to specific numbers to specify- e.g 0,1,3, default is to use all available')
parser.add_argument('--threads', type=int, default=4,
                    help='number of threads for data loader to use. default=4')
parser.add_argument('--optimize-cudnn', action='store_true', default=True, 
                    help='change to false for many different short runs, or runs where input sizes are changing constantly')

######## Optimization and learning rate settings for training #########

parser.add_argument('--optimizer', type=str, default='SGD',
                    help='Optimizer: choose "Adam", "SGD", "RMSprop" etc')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate. default=0.01')
parser.add_argument('--lr-stages', type=int, default=False,
                    help='set to a number n to decrease lr by factor of lr_increment every 1/n epochs')
parser.add_argument('--lr-increment', type=float, default=10.,
                    help='used with --lr-stages. amount to divide learning rate by. default=10')
parser.add_argument('--lr-momentum', type=float, default=None,
                    help='learning rate momentum. default=None, but can use 0.9 etc.')
parser.add_argument('--lr-decay', type = float, default=1, 
                    help='learning rate decay. default is 1. set to below one for decay (i.e. .99, .95 etc)')
parser.add_argument('--lr-floor', type = float, default=0.000001, 
                    help='learning rate floor = minimum learning rate')

########### Logging and checkpoint settings #############

parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--checkpoint-file', type=str, default=None, help='which checkpoint file to use (filename)')
parser.add_argument('--save-checkpoints', action='store_true', default=False)
parser.add_argument('--checkpoint-dir', type=str, default='results/checkpoints')
parser.add_argument('--checkpoint-accuracy-threshold', type=float, default=50.1,
                    help='accuracy threshold for saving checkpoints. 0-100, default=50.1')
parser.add_argument('--min-checkpoint-improvement', type=float, default=0.1,
                    help='minimum accuracy improvement before saving next checkpoint')
parser.add_argument('--save-training-info', action='store_true', default=True)
parser.add_argument('--training-info-dir', type=str, default='results/training_info')
parser.add_argument('--debug-mode', action='store_true', default=False,
                    help='shows print statements to aid debugging. default=false')

args = parser.parse_args()

###################### Load data ###########################
print('==> Preparing data..')
trainset, testset, trainloader, testloader, classes = get_cifar10(args)

#############################################################
################## List of Models To Train ##################
#############################################################
"""
To test a different set of models, just change these manually.
Refer to TRAINING_INSTRUCTIONS.md for details.
"""

model_list = [
    'ResNet18()',
    'DPN92()',
    'DPN92_act1(activation=swish)'
]

############################################################
################# Set up files for logging #################
############################################################

# Set up combo results files outside the loop
if args.save_training_info:
    combo_train_folder = os.path.join(args.training_info_dir, "combo_train")
    combo_test_folder = os.path.join(args.training_info_dir, "combo_test")

    if not os.path.isdir(combo_train_folder):
        os.makedirs(combo_train_folder)
    if not os.path.isdir(combo_test_folder):
        os.makedirs(combo_test_folder)

    combo_train_filename = "combo_train_" + "_".join([str(x) for x in time.localtime()[0:5]]) + '.csv'
    combo_train_filename = os.path.join(combo_train_folder,combo_train_filename)

    combotrainfile = open(combo_train_filename,"w",encoding="utf-8")
    combotrainfile.write(str(model_list)+'\n\n')
    combotrainfile.write(str(args)+'\n\n')
    
    combo_test_filename = "combo_test_" + "_".join([str(x) for x in time.localtime()[0:5]]) + '.csv'
    combo_test_filename = os.path.join(combo_test_folder,combo_test_filename)

    combotestfile = open(combo_test_filename,"w",encoding="utf-8")
    combotestfile.write('\n\n'+str(model_list)+'\n\n')
    combotestfile.write(str(args)+'\n\n')

for model_position, current_model in enumerate(model_list):
    best_test_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('\n\n==> Building model: {}. This is model #{} out of a total of {} models'.format(current_model, model_position+1, len(model_list)+1))
    net = eval(current_model)

    # print model parameters using function from utils
    total_model_params = print_model_params(net)

    ################################################################
    ###################### CUDA settings ###########################
    ################################################################

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        net.cuda()
        if args.gpus:
            args.gpus = [int(i) for i in args.gpus.split(',')]
            torch.cuda.set_device(args.gpus[0])
        else:
            net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

        cudnn.benchmark = args.optimize_cudnn
        torch.cuda.manual_seed_all(args.seed)


    #######################################################
    ############# Loss and Optimizer Settings #############
    #######################################################

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    lr = args.lr
    if args.lr_momentum:
        optimizer = eval('optim.' + args.optimizer + '(net.parameters(), lr=' + str(lr) + \
            ', momentum=' + str(args.lr_momentum) + ', weight_decay=' + str(1-args.lr_decay) + ')')
    else:
        optimizer = eval('optim.' + args.optimizer + '(net.parameters(), lr=' + str(lr) + \
            ', weight_decay=' + str(1-args.lr_decay) + ')')

    # Initialize the learning rate settings
    lr = args.lr

    if args.lr_stages:
        epoch_increment = args.epochs // args.lr_stages
        epoch_threshold = epoch_increment


    #######################################################################
    ############### Set up files to save training info into ###############
    #######################################################################

    # Prepare to save training info to csv file
    if args.save_training_info:
        train_folder = os.path.join(args.training_info_dir,"train")
        test_folder = os.path.join(args.training_info_dir,"test")

        if not os.path.isdir(train_folder):
            os.makedirs(train_folder)

        if not os.path.isdir(test_folder):
            os.makedirs(test_folder)

        training_info_filename = "train" + "_".join([str(x) for x in time.localtime()[0:5]]) + "_" + str(current_model) + '.csv'
        training_info_filename = os.path.join(train_folder,training_info_filename)

        testing_info_filename = "test" + "_".join([str(x) for x in time.localtime()[0:5]]) + "_" + str(current_model) + '.csv'
        testing_info_filename = os.path.join(test_folder,testing_info_filename)        

        trainfile = open(training_info_filename,"w",encoding="utf-8")
        testfile = open(testing_info_filename,"w",encoding="utf-8")

        header_row = 'trainortest,epoch,batch_idx,num_batches,loss,accuracy,correct,total,batch_loss,lr,batch_accuracy,max_batch_acc,min_batch_acc,total_model_params'

        trainfile.write(str(current_model)+'\n\n')
        trainfile.write(header_row)

        testfile.write('\n' + str(current_model) + '\n')
        testfile.write(header_row + '\n')

        combotrainfile.write('\n'+str(current_model)+'\n')
        combotrainfile.write(header_row)

        combotestfile.write(str(current_model)+'\n')
        combotestfile.write(header_row)

    #####################################################
    ############### Define train function ###############
    #####################################################

    # Training
    def train(epoch):
        global optimizer
        global lr
        global epoch_increment
        global epoch_threshold
        print('\nEpoch: %d  out of total %d  |    Model #%d (out of %d): %s' 
            % (epoch, args.epochs, model_position + 1, len(model_list)+1, current_model))

        net.train()
        train_loss = 0
        correct = 0
        total = 0
        max_batch_acc = 0
        min_batch_acc = 100

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
                criterion.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            batch_correct = predicted.eq(targets.data).cpu().sum()
            batch_accuracy = 100.*batch_correct / args.train_batch_size
            correct += batch_correct
            train_accuracy = 100.*correct/total

            if batch_accuracy > max_batch_acc:
                max_batch_acc = batch_accuracy
            if batch_accuracy < min_batch_acc:
                min_batch_acc = batch_accuracy 

            #all_training_info = 'Epoch, %d, batch_idx, %d, num_batches, %d, Loss, %.3f , Acc, %.3f%% , correct, %d, total, %d,  batch_loss, %.3f, lr, %.4f'
            all_training_info_line = '%d,%d,%d,%.3f,%.3f,%d,%d,%.3f,%.4f,%.3f,%.3f,%.3f,%d' \
                                    % (epoch, batch_idx, len(trainloader), train_loss/(batch_idx+1), train_accuracy, correct, total, loss.data[0], 
                                        optimizer.defaults['lr'],batch_accuracy, max_batch_acc, min_batch_acc,total_model_params)

            # start the command line progress bar
            train_progress_bar_output = progress_bar(batch_idx, len(trainloader), 
                'Loss: %.3f | Acc: %.2f%% (%d/%d) | Batch Loss: %.2f | lr: %.5f | Batch Acc: %.2f%% (Max: %.2f%%, Min: %.2f%%)'
                                    % (train_loss/(batch_idx+1), train_accuracy, correct, total, loss.data[0], optimizer.defaults['lr'],
                                        batch_accuracy, max_batch_acc, min_batch_acc))

            # Save Training Info to CSV file
            if args.save_training_info:
                trainfile.write('train,' + all_training_info_line+'\n')             
                if batch_idx == len(trainloader)-1 or batch_idx == len(trainloader):
                    combotrainfile.write('train,' + all_training_info_line+'\n')

            ################################################################
            ################## Updating the learning rate ##################
            ################################################################

            if args.lr_decay and args.lr_stages:
                print('\n\n')
                print("Learning decay and learning stages cannot be set simultaneously.")
                print("Learning rate will not be updated.")
                print('\n\n')

            if args.lr_decay and not args.lr_stages:
                if lr > args.lr_floor:
                    lr *= args.lr_decay
                    if args.lr_momentum:
                        optimizer = eval('optim.' + args.optimizer + '(net.parameters(), lr=' + str(lr) + \
                            ', momentum=' + str(args.lr_momentum) + ', weight_decay=' + str(1-args.lr_decay) + ')')
                    else:
                        optimizer = eval('optim.' + args.optimizer + '(net.parameters(), lr=' + str(lr) + \
                            ', weight_decay=' + str(1-args.lr_decay) + ')')
            
            if args.lr_stages and not args.lr_decay:
                if lr > args.lr_floor:
                    if epoch > epoch_threshold:
                        lr /= args.lr_increment
                        #print("\nLearning rate changed to : %.4f \n" % (lr))
                        epoch_threshold += epoch_increment

                        if args.lr_momentum:
                            optimizer = eval('optim.' + args.optimizer + '(net.parameters(), lr=' + str(lr) + \
                                ', momentum=' + str(args.lr_momentum) + ', weight_decay=' + str(1-args.lr_decay) + ')')
                        else:
                            optimizer = eval('optim.' + args.optimizer + '(net.parameters(), lr=' + str(lr) + \
                                ', weight_decay=' + str(1-args.lr_decay) + ')')

    ####################################################
    ############### Define test function ###############
    ####################################################

    def test(epoch):
        global best_test_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        max_batch_acc = 0
        min_batch_acc = 100

        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            batch_correct = predicted.eq(targets.data).cpu().sum()
            batch_accuracy = 100.*batch_correct / args.test_batch_size
            correct += batch_correct
            test_accuracy = 100.*correct/total

            if batch_accuracy > max_batch_acc:
                max_batch_acc = batch_accuracy
            if batch_accuracy < min_batch_acc:
                min_batch_acc = batch_accuracy 

            #all_testing_info = 'Epoch, %d, batch_idx, %d, num_batches, %d, Loss, %.3f , Acc, %.3f%% , correct, %d, total, %d,  batch_loss, %.3f, lr, %.4f'
            all_testing_info_line = '%d,%d,%d,%.3f,%.3f,%d,%d,%.3f,%.4f,%.3f,%.3f,%.3f,%d' \
                                    % (epoch, batch_idx, len(testloader), test_loss/(batch_idx+1), test_accuracy, correct, total, 
                                        loss.data[0], optimizer.defaults['lr'],batch_accuracy, max_batch_acc, min_batch_acc,total_model_params)

            test_progress_bar_output = progress_bar(batch_idx, len(testloader), 
                'Loss: %.3f | Acc: %.2f%% (%d/%d) |Batch Loss: %.2f | lr: %.5f | Batch Acc: %.2f%% (Max: %.2f%%, Min: %.2f%%)'
                                    % (test_loss/(batch_idx+1), test_accuracy, correct, total, loss.data[0],
                                        optimizer.defaults['lr'],batch_accuracy, max_batch_acc, min_batch_acc))

            # Save Test Info to to the training info CSV file
            if args.save_training_info:
                testfile.write('test,' + all_testing_info_line+'\n')
                if batch_idx == len(testloader)-1 or batch_idx == len(testloader):
                    combotestfile.write('test,' + all_testing_info_line+'\n')                    

            # Save checkpoint.
            if args.save_checkpoints:
                if test_accuracy > args.checkpoint_accuracy_threshold and test_accuracy > best_test_acc + args.min_checkpoint_improvement:

                    if not os.path.isdir(args.checkpoint_dir):
                        os.makedirs(args.checkpoint_dir)

                    checkpoint_filename = str(current_model) + '_' + "acc{:.0f}".format(test_accuracy) + '_ckpt.t' + str(epoch)
                    checkpoint_filename = os.path.join(args.checkpoint_dir, checkpoint_filename)
                    print('Saving checkpoint {}'.format(checkpoint_filename))

                    state = {
                        'net': net.module if use_cuda else net,
                        'acc': test_accuracy,
                        'epoch': epoch,
                    }

                    torch.save(state, checkpoint_filename)

                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy

    ##################################################
    ############### Main training loop ###############
    ##################################################

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch)
        test(epoch)


    if args.save_training_info:
        trainfile.close()
        testfile.close()

# outside main loop
if args.save_training_info:
    combotrainfile.close()
    combotestfile.close()