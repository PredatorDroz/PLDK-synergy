"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import random
import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def fetch_dataloader(types, params):
    DATA_ROOT = r"F:\Other\data2"
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    train_ds = torchvision.datasets.CIFAR100(os.path.join(DATA_ROOT, 'cifar100'), train=True, download=True)
    test_ds = torchvision.datasets.CIFAR100(os.path.join(DATA_ROOT, 'cifar100'), train=False, download=True)

    _data_train = np.concatenate([np.array(train_ds[i][0]) for i in range(len(train_ds))])
    _data_test = np.concatenate([np.array(test_ds[i][0]) for i in range(len(test_ds))])

    train_mean = _data_train.mean(axis=(0, 1))
    train_std = _data_train.std(axis=(0, 1))

    test_mean = _data_test.mean(axis=(0, 1))
    test_std = _data_test.std(axis=(0, 1))

    print(f'Hard code CIFAR100 train/test mean/std for next time')

    # using random crops and horizontal flip for train set
#     if params.augmentation == "yes":
#         train_transformer = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

#     # data augmentation can be turned off
#     else:
#         train_transformer = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    train_transformer = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transform.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std),
])

    test_transformer = transforms.Compose([
    transform.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(test_mean, test_std),
])

# %% Choose model para

    # transformer for dev set
    dev_transformer = transforms.Compose([
         transforms.ToTensor(),
    transforms.Normalize(test_mean, test_std)])
    
    trainset = torchvision.datasets.CIFAR100(os.path.join(DATA_ROOT, 'cifar100'), train=True, download=True,transform=train_transformer)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        shuffle=True, num_workers=params.num_workers, pin_memory=params.cuda)

    devset = torchvision.datasets.CIFAR100(os.path.join(DATA_ROOT, 'cifar100'), train=False, download=True ,transform=dev_transformer)
    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl


def fetch_subset_dataloader(types, params):
    print("ZZZZZZZZZZZ")
    DATA_ROOT = r"F:\Other\data2"
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    
    """
    train_ds = torchvision.datasets.CIFAR100(os.path.join(DATA_ROOT, 'cifar100'), train=True, download=True)
    test_ds = torchvision.datasets.CIFAR100(os.path.join(DATA_ROOT, 'cifar100'), train=False, download=True)
    
    _data_train = np.concatenate([np.array(train_ds[i][0]) for i in range(len(train_ds))])
    _data_test = np.concatenate([np.array(test_ds[i][0]) for i in range(len(test_ds))])

    train_mean = _data_train.mean(axis=(0, 1))
    train_std = _data_train.std(axis=(0, 1))

    test_mean = _data_test.mean(axis=(0, 1))
    test_std = _data_test.std(axis=(0, 1))


    # using random crops and horizontal flip for train set
    if params.augmentation == "yes":
        train_transformer = transforms.Compose([
 transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
           transforms.Normalize(train_mean, train_std),
        ])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transform.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transform.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(test_mean, test_std)])

    trainset = torchvision.datasets.CIFAR100(os.path.join(DATA_ROOT, 'cifar100'), train=True,download=True,transform=train_transformer)

    devset = torchvision.datasets.CIFAR100(os.path.join(DATA_ROOT, 'cifar100'), train=False,
        download=True, transform=dev_transformer)

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)

    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=params.cuda)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl