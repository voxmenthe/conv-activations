import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

__DATASETS_DEFAULT_PATH = '/notebooks/111_Pytorch/data'
default_train_batch = 128
default_test_batch = 100
default_threads = 2

def get_cifar10(args=None):

  if args:
    datapath = args.data_path
    trainbatch = args.train_batch_size
    testbatch = args.test_batch_size
    threads = args.threads
  else:
    datapath = __DATASETS_DEFAULT_PATH
    trainbatch = default_train_batch
    testbatch = default_test_batch
    threads = default_threads

  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])


  trainset = torchvision.datasets.CIFAR10(root=datapath, train=True, download=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainbatch, shuffle=True, num_workers=threads)

  testset = torchvision.datasets.CIFAR10(root=datapath, train=False, download=True, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=testbatch, shuffle=False, num_workers=threads)

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  return trainset, testset, trainloader, testloader, classes

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
    train = (split == 'train')
    root = os.path.join(datasets_path, name)
    if name == 'cifar10':
        return datasets.CIFAR10(root=root,
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=root,
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'stl10':
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'imagenet':
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
