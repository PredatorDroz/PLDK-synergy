

import argparse
import logging
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm

import utilse
import model.net as net
import model.data_loader as data_loader
import model.resnet as resnet
import model.wrn as wrn
import model.densenet as densenet
import model.resnext as resnext
import model.preresnet as preresnet
from evaluate import evaluate, evaluate_kd


parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'

import model.alexnet as alexnet

teacher_model = alexnet.AlexNet(num_classes=10)
teacher_checkpoint = 'experiments/base_cnn/epoch399.pth.tar'
utilse.load_checkpoint(teacher_checkpoint, teacher_model)