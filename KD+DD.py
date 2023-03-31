# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 17:41:57 2023

@author: fahim
"""

"""Main entrance for train/eval with/without KD on CIFAR-10"""

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
#import model.resnext as resnext
import model.preresnet as preresnet
from evaluate import evaluate, evaluate_kd


parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'


# Load the parameters from json file
args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utilse.Params(json_path)

import model.alexnet as alexnets

def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utilse.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            #print(train_batch.shape)
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(non_blocking=False), labels_batch.cuda(non_blocking=False)
                
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data.cpu()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data.cpu())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - name of file to restore from (without its extension .pth.tar)
    """
    save_dir = 'D:/research_2022/PyTorch-GAN-master/PyTorch-GAN-master/implementations/cgan/data/model/'
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utilse.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    # learning rate schedulers for different models:
    if params.model_version == "resnet18":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn":
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2)

    scheduler = StepLR(optimizer, step_size=100, gamma=0.2)
    for epoch in range(params.num_epochs):
     
        scheduler.step()
     
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)        

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utilse.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)
       

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utilse.save_dict_to_json(val_metrics, best_json_path)
            torch.save(model.state_dict(), os.path.join(save_dir, 'TeacherR101.pth')) 

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utilse.save_dict_to_json(val_metrics, last_json_path)

# Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()
    teacher_model.eval()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utilse.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(non_blocking=False), \
                                            labels_batch.cuda(non_blocking=False)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output, fetch teacher output, and compute KD loss
            output_batch = model(train_batch)

            # get one batch output from teacher_outputs list

            with torch.no_grad():
                output_teacher_batch = teacher_model(train_batch)
            if params.cuda:
                output_teacher_batch = output_teacher_batch.cuda(non_blocking=False)

            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data.cpu()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data.cpu())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                       loss_fn_kd, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - file to restore (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utilse.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    
    # Tensorboard logger setup
    # board_logger = utils.Board_Logger(os.path.join(model_dir, 'board_logs'))

    # learning rate schedulers for different models:
    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    elif params.model_version == "cnn_distill": 
        scheduler = StepLR(optimizer, step_size=100, gamma=0.2) 

    for epoch in range(80):
        scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_kd(model, teacher_model, optimizer, loss_fn_kd, train_dataloader,
                 metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate_kd(model, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        print("saving weights")
        utilse.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)
        utilse.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint='experiments/base_cnn')

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utilse.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utilse.save_dict_to_json(val_metrics, last_json_path)


        # #============ TensorBoard logging: uncomment below to turn in on ============#
        # # (1) Log the scalar values
        # info = {
        #     'val accuracy': val_acc
        # }

        # for tag, value in info.items():
        #     board_logger.scalar_summary(tag, value, epoch+1)

        # # (2) Log values and gradients of the parameters (histogram)
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     board_logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        #     # board_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)


# %% privacy_train
def privacy_train(trainloader, model, inference_model, criterion, optimizer, use_cuda, num_batchs=1000):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()

    inference_model.train()
    model.eval()
    # switch to evaluate mode

    end = time.time()
    first_id = -1
    for batch_idx, ((tr_input, tr_target), (te_input, te_target)) in trainloader:
        # measure data loading time
        if first_id == -1:
            first_id = batch_idx

        data_time.update(time.time() - end)

        if use_cuda:
            tr_input = tr_input.cuda()
            te_input = te_input.cuda()
            tr_target = tr_target.cuda()
            te_target = te_target.cuda()

        v_tr_input = torch.autograd.Variable(tr_input)
        v_te_input = torch.autograd.Variable(te_input)
        v_tr_target = torch.autograd.Variable(tr_target)
        v_te_target = torch.autograd.Variable(te_target)

        # compute output
        model_input = torch.cat((v_tr_input, v_te_input))

        pred_outputs = model(model_input)
        #y_hat

        infer_input = torch.cat((v_tr_target, v_te_target))
        #(y_hat)

        # TODO fix
        # mtop1, mtop5 = accuracy(pred_outputs.data, infer_input.data, topk=(1, 5))
        mtop1 = top_k_accuracy_score(y_true=infer_input.data.cpu(), y_score=pred_outputs.data.cpu(),
                                     k=1, labels=range(num_classes))

        mtop5 = top_k_accuracy_score(y_true=infer_input.data.cpu(), y_score=pred_outputs.data.cpu(),
                                     k=5, labels=range(num_classes))

        mtop1_a.update(mtop1, model_input.size(0))
        mtop5_a.update(mtop5, model_input.size(0))

        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), num_classes)) - 1)).cuda().type(torch.float)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.int64).view([-1, 1]).data, 1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        #ONE_hot y_hat

        attack_model_input = pred_outputs  # torch.cat((pred_outputs,infer_input_one_hot),1)
        member_output = inference_model(attack_model_input, infer_input_one_hot)
        #inf_model(y,y_hat)
        #member->?0/1

        is_member_labels = torch.from_numpy(
            np.reshape(
                np.concatenate((np.zeros(v_tr_input.size(0)), np.ones(v_te_input.size(0)))),
                [-1, 1]
            )
        ).cuda()

        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.float)
        #true_labels

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
        losses.update(loss.data.item(), model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx - first_id > num_batchs:
            break

        # plot progress
        if batch_idx % 10 == 0:
            print(report_str(batch_idx, data_time.avg, batch_time.avg, losses.avg, top1.avg, None))

    return losses.avg, top1.avg

##########################
### MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        #self.fc = nn.Linear(2048 * block.expansion, num_classes)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas



def resnet101(num_classes, grayscale):
    """Constructs a ResNet-101 model."""
    model = ResNet(block=Bottleneck, 
                   layers=[3, 4, 23, 3],
                   num_classes=10,
                   grayscale=False)
    return model

''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size = (32,32)):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        # print("MODEL DATA ON: ", x.get_device(), "MODEL PARAMS ON: ", self.classifier.weight.data.get_device())
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2


        return nn.Sequential(*layers), shape_feat 
    
    

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
nets= ConvNet(channel=3, num_classes= 10,net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling,im_size=(32,32))
nets2= ConvNet(channel=3, num_classes= 10,net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling,im_size=(32,32))

import numpy as np
import torch.utils.data as data
import model.net as net

# use GPU if available
params.cuda = torch.cuda.is_available()

# Set the random seed for reproducible experiments
# random.seed(230)
# torch.manual_seed(230)
# if params.cuda: torch.cuda.manual_seed(230)

# Set the logger
# utilse.set_logger(os.path.join(args.model_dir, 'train.log'))

# # Create the input data pipeline
# logging.info("Loading the datasets...")

# # fetch dataloaders, considering full-set vs. sub-set scenarios
# if params.subset_percent < 1.0:
#     train_dl = data_loader.fetch_subset_dataloader('train', params)
# else:
#     train_dl = data_loader.fetch_dataloader('train', params)
    
# import torchvision.transforms as transforms
# import torch
# import io

# x_train=torch.load('D:/research_2022/data-distillation/logged_files/CIFAR10/project/images_best.pt',map_location=lambda storage, loc: storage.cuda(0))
# y_train=torch.load('D:/research_2022/data-distillation/logged_files/CIFAR10/project/labels_best.pt',map_location=lambda storage, loc: storage.cuda(0))


# save_dir='logged_files/'
# with open('D:/research_2022/data-distillation/logged_files/CIFAR10/project/images_best.pt', 'rb') as f:
#     buffer = io.BytesIO(f.read())
#     print(buffer)
# torch.load(buffer)


# train_x =x_train
# #train_x = torch.stack([(x) for x in x_gen])
# train_y = y_train
# print(train_x.shape)
# print(train_y.shape)


# # train_x=train_x[0:60000,:,:,:]
# print(train_x.shape)
# print(train_y.shape)
# full_indices = np.arange(len(train_x))

# np.random.shuffle(full_indices)
# tensor_x = train_x[full_indices]
# tensor_y = train_y[full_indices]

# full_indices = np.arange(len(train_y))
# np.random.shuffle(full_indices)
# tensor_x = train_x[full_indices]
# tensor_y = train_y[full_indices]

# trainset = data.TensorDataset(tensor_x, tensor_y)  # create your datset
# train_dl = data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

# dev_dl = data_loader.fetch_dataloader('dev', params)

# torch.cuda.empty_cache()

# logging.info("- done.")

# """Based on the model_version, determine model/optimizer and KD training mode
#    WideResNet and DenseNet were trained on multi-GPU; need to specify a dummy
#    nn.DataParallel module to correctly load the model parameters
# """
# if "distill" in params.model_version:

#     # train a 5-layer CNN or a 18-layer ResNet with knowledge distillation
#     if params.model_version == "cnn_distill":
#         print("KD with CNN")
#         #model=alexnets.AlexNet().cuda() if params.cuda else alexnets.AlexNeT(params)
#         #model=densenet.DenseNet(num_classes=10).cuda()
#         #model=resnext50().cuda() if params.cuda else resnext50()
#         model=ConvNet(channel=3, num_classes= 10,net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling,im_size=(32,32)).cuda()
#         optimizer = optim.Adam(model.parameters(), lr=params.learning_rate*5)
#         student_checkpoint = 'net_conv_0.64964cc_test_mean.pth'
#         #model.load_state_dict(torch.load(student_checkpoint))
#         #model = resnet101().cuda()
        
#         # fetch loss function and metrics definition in model files
#         loss_fn_kd = net.loss_fn_kd
#         metrics = net.metrics

#     elif params.model_version == 'resnet18_distill':
#         model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
#         optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
#                               momentum=0.9, weight_decay=5e-4)
#         # fetch loss function and metrics definition in model files
#         loss_fn_kd =alexnets.loss_fn_kd
#         metrics = resnet.metrics

#     """ 
#         Specify the pre-trained teacher models for knowledge distillation
#         Important note: wrn/densenet/resnext/preresnet were pre-trained models using multi-GPU,
#         therefore need to call "nn.DaraParallel" to correctly load the model weights
#         Trying to run on CPU will then trigger errors (too time-consuming anyway)!
#     """
#     if params.teacher == "resnet18":
#         print("ALEX teacher")
#         #teacher_model = alexnets.AlexNet()
#         teacher_model = resnext50()
#         #model=nets.cuda()
#         #teacher_model=nets2
#         #print(teacher_model)
#         teacher_checkpoint = 'D:/research_2022/PyTorch-GAN-master/PyTorch-GAN-master/implementations/cgan/data/model/TeacherR50.pth'
#         teacher_checkpoint = 'D:/research_2022/PyTorch-GAN-master/PyTorch-GAN-master/implementations/cgan/data/model/TeacherRES_DD.pth'
#         #teacher_model = densenet.DenseNet(num_classes=10)
#         #teacher_model = densenet.DenseNet(num_classes=10)
#         #teacher_model = resnet.ResNet18().cuda()
#         #teacher_checkpoint = 'experiments/base_cnn/epoch399'
#         teacher_model = teacher_model.cuda()
#         teacher_model.load_state_dict(torch.load(teacher_checkpoint))
#         teacher_model = teacher_model.cuda()
        
#     elif params.teacher == "wrn":
#         teacher_model = wrn.WideResNet(depth=28, num_classes=10, widen_factor=10,
#                                        dropRate=0.3)
#         teacher_checkpoint = 'experiments/base_wrn/best.pth.tar'
#         teacher_model = nn.DataParallel(teacher_model).cuda()

#     elif params.teacher == "densenet":
#         teacher_model = densenet.DenseNet(depth=100, growthRate=12)
#         teacher_checkpoint = 'experiments/base_densenet/best.pth.tar'
#         teacher_model = nn.DataParallel(teacher_model).cuda()

#     elif params.teacher == "resnext29":
# #         teacher_model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=10)
# #         teacher_checkpoint = 'experiments/base_resnext29/best.pth.tar'
# #         teacher_model = nn.DataParallel(teacher_model).cuda()
# #         teacher_model = alexnets.AlexNet()
#         teacher_model = densenet.DenseNet(num_classes=10)
#         teacher_checkpoint = 'experiments/base_cnn/epoch399'
#         teacher_model = teacher_model.cuda()

#     elif params.teacher == "preresnet110":
#         teacher_model = preresnet.PreResNet(depth=110, num_classes=10)
#         teacher_checkpoint = 'experiments/base_preresnet110/best.pth.tar'
#         teacher_model = nn.DataParallel(teacher_model).cuda()
        
#     elif params.teacher == "AlexNet":
#         teacher_model = AlexNet(num_classes)
#         teacher_checkpoint = 'F:\Other\privacy_reg\results\epoch399.pth.tar'
#         teacher_model = nn.DataParallel(teacher_model).cuda()
        
#     #teacher_checkpoint = 'D:/research_2022/PyTorch-GAN-master/PyTorch-GAN-master/implementations/cgan/data/model/TeacherR50.pth'
#     #utilse.load_checkpoint(teacher_checkpoint, teacher_model)
    

#     # Train the model with KD
#     logging.info("Experiment - model version: {}".format(params.model_version))
#     logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
#     logging.info("First, loading the teacher model and computing its outputs...")
    
#     train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
#                           metrics, params, args.model_dir, args.restore_file)
# #     loss_fn=alexnets.loss_fn
# #     train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params,
# #                        args.model_dir, args.restore_file)
# else:
#     if params.model_version == "cnn":
#         model = net.Net(params).cuda() if params.cuda else net.Net(params)
#         #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
#         optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
#         # fetch loss function and metrics
#         loss_fn = net.loss_fn
#         metrics = net.metrics

#     elif params.model_version == "resnet18":
#         model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
#         optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
#                               momentum=0.9, weight_decay=5e-4)
#         # fetch loss function and metrics
#         loss_fn = resnet.loss_fn
#         metrics = resnet.metrics



#     # elif params.model_version == "wrn":
#     #     model = wrn.wrn(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
#     #     model = model.cuda() if params.cuda else model
#     #     optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
#     #                           momentum=0.9, weight_decay=5e-4)
#     #     # fetch loss function and metrics
#     #     loss_fn = wrn.loss_fn
#     #     metrics = wrn.metrics
        
#     # Train the model
#     logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
#     train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params,
#                        args.model_dir, args.restore_file)

'''Simplified version of DLA in PyTorch.

Note this implementation is not identical to the original paper version.
But it seems works fine.

See dla.py for the original paper version.
(simpleNet)
Reference:
    Deep Layer Aggregation. https://arxiv.org/abs/1707.06484
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, 1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super(Tree, self).__init__()
        self.root = Root(2*out_channels, out_channels)
        if level == 1:
            self.left_tree = block(in_channels, out_channels, stride=stride)
            self.right_tree = block(out_channels, out_channels, stride=1)
        else:
            self.left_tree = Tree(block, in_channels,
                                  out_channels, level=level-1, stride=stride)
            self.right_tree = Tree(block, out_channels,
                                   out_channels, level=level-1, stride=1)

    def forward(self, x):
        out1 = self.left_tree(x)
        out2 = self.right_tree(out1)
        out = self.root([out1, out2])
        return out


class SimpleDLA(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10):
        super(SimpleDLA, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.layer3 = Tree(block,  32,  64, level=1, stride=1)
        self.layer4 = Tree(block,  64, 128, level=2, stride=2)
        self.layer5 = Tree(block, 128, 256, level=2, stride=2)
        self.layer6 = Tree(block, 256, 512, level=1, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = SimpleDLA()
    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.size())
    
def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['net'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint


import numpy as np
import torch.utils.data as data
import model.net as net
import torch.backends.cudnn as cudnn

# use GPU if available
params.cuda = torch.cuda.is_available()

# Set the random seed for reproducible experiments
# random.seed(230)
# torch.manual_seed(230)
# if params.cuda: torch.cuda.manual_seed(230)

# Set the logger
utilse.set_logger(os.path.join(args.model_dir, 'train.log'))

# Create the input data pipeline
logging.info("Loading the datasets...")

# fetch dataloaders, considering full-set vs. sub-set scenarios
if params.subset_percent < 1.0:
    train_dl = data_loader.fetch_subset_dataloader('train', params)
else:
    train_dl = data_loader.fetch_dataloader('train', params)
    
import torchvision.transforms as transforms
import torch
import io

x_train=torch.load('D:/research_2022/data-distillation/mtt-distillation-main/mtt-distillation-main/logged_files/CIFAR10/project/images_best.pt',map_location=lambda storage, loc: storage.cuda(0))
y_train=torch.load('D:/research_2022//data-distillation/logged_files/CIFAR10/project/labels_best.pt',map_location=lambda storage, loc: storage.cuda(0))


save_dir='logged_files/'
with open('D:/research_2022/data-distillation/mtt-distillation-main/mtt-distillation-main/logged_files/CIFAR10/project/images_best.pt', 'rb') as f:
    buffer = io.BytesIO(f.read())
    print(buffer)
torch.load(buffer)


train_x =x_train
#train_x = torch.stack([(x) for x in x_gen])
train_y = y_train
print(train_x.shape)
print(train_y.shape)


# train_x=train_x[0:60000,:,:,:]
print(train_x.shape)
print(train_y.shape)
full_indices = np.arange(len(train_x))

np.random.shuffle(full_indices)
tensor_x = train_x[full_indices]
tensor_y = train_y[full_indices]

full_indices = np.arange(len(train_y))
np.random.shuffle(full_indices)
tensor_x = train_x[full_indices]
tensor_y = train_y[full_indices]

trainset = data.TensorDataset(tensor_x, tensor_y)  # create your datset
train_dl = data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

dev_dl = data_loader.fetch_dataloader('dev', params)

torch.cuda.empty_cache()

logging.info("- done.")

"""Based on the model_version, determine model/optimizer and KD training mode
   WideResNet and DenseNet were trained on multi-GPU; need to specify a dummy
   nn.DataParallel module to correctly load the model parameters
"""
if "distill" in params.model_version:

    # train a 5-layer CNN or a 18-layer ResNet with knowledge distillation
    if params.model_version == "cnn_distill":
        print("KD with CNN")
        #model=alexnets.AlexNet().cuda() if params.cuda else alexnets.AlexNeT(params)
        #model=densenet.DenseNet(num_classes=10).cuda()
        #model=resnext50().cuda() if params.cuda else resnext50()
        model=nets.cuda()
        #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate/5)
        optimizer = optim.SGD(model.parameters(), lr=0.06,
                      momentum=0.9, weight_decay=5e-4)
        optimizer = optim.SGD(model.parameters(), lr=0.00001,
                      momentum=0.9, weight_decay=5e-4)
        #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate/20)
        student_checkpoint = 'net_conv_0.64964cc_test_mean.pth'
        #model.load_state_dict(torch.load(student_checkpoint))
        #model = resnet101().cuda()
        
        # fetch loss function and metrics definition in model files
        loss_fn_kd = net.loss_fn_kd
        metrics = net.metrics

    elif params.model_version == 'resnet18_distill':
        model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                              momentum=0.9, weight_decay=5e-4)
        # fetch loss function and metrics definition in model files
        loss_fn_kd =alexnets.loss_fn_kd
        metrics = resnet.metrics

    """ 
        Specify the pre-trained teacher models for knowledge distillation
        Important note: wrn/densenet/resnext/preresnet were pre-trained models using multi-GPU,
        therefore need to call "nn.DaraParallel" to correctly load the model weights
        Trying to run on CPU will then trigger errors (too time-consuming anyway)!
    """
    if params.teacher == "resnet18":
        print("ALEX teacher")
        #teacher_model = alexnets.AlexNet()
        #teacher_model = resnext50()
        teacher_model=SimpleDLA()
        
        #teacher_model=nets2
        #print(teacher_model)
        teacher_checkpoint = 'D:/research_2022/PyTorch-GAN-master/PyTorch-GAN-master/implementations/cgan/data/model/TeacherR50.pth'
        #teacher_checkpoint = 'D:/research_2022/PyTorch-GAN-master/PyTorch-GAN-master/implementations/cgan/data/model/TeacherRES_DD.pth'
        #teacher_checkpoint = 'D:/research_2022/PyTorch-GAN-master/PyTorch-GAN-master/implementations/cgan/data/model/TeacherRES_DD101.pth'
        #teacher_model = densenet.DenseNet(num_classes=10)
        #teacher_model = densenet.DenseNet(num_classes=10)
        #teacher_model = resnet.ResNet18().cuda()
        #teacher_checkpoint = 'experiments/base_cnn/epoch399'
        teacher_checkpoint = 'D:/research_2022/pytorch-cifar-master/pytorch-cifar-master/checkpoint/Teacher_DD.pth'
        teacher_model = torch.nn.DataParallel(teacher_model)
        cudnn.benchmark = True
        teacher_model = teacher_model.cuda()
        teacher_model.load_state_dict(torch.load(teacher_checkpoint))
        #teacher_model = teacher_model.cuda()
        
    elif params.teacher == "wrn":
        teacher_model = wrn.WideResNet(depth=28, num_classes=10, widen_factor=10,
                                       dropRate=0.3)
        teacher_checkpoint = 'experiments/base_wrn/best.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()

    elif params.teacher == "densenet":
        teacher_model = densenet.DenseNet(depth=100, growthRate=12)
        teacher_checkpoint = 'experiments/base_densenet/best.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()

    elif params.teacher == "resnext29":
#         teacher_model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=10)
#         teacher_checkpoint = 'experiments/base_resnext29/best.pth.tar'
#         teacher_model = nn.DataParallel(teacher_model).cuda()
#         teacher_model = alexnets.AlexNet()
        teacher_model = densenet.DenseNet(num_classes=10)
        teacher_checkpoint = 'experiments/base_cnn/epoch399'
        teacher_model = teacher_model.cuda()

    elif params.teacher == "preresnet110":
        teacher_model = preresnet.PreResNet(depth=110, num_classes=10)
        teacher_checkpoint = 'experiments/base_preresnet110/best.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()
        
    elif params.teacher == "AlexNet":
        teacher_model = AlexNet(num_classes)
        teacher_checkpoint = 'F:\Other\privacy_reg\results\epoch399.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()
        
    #teacher_checkpoint = 'D:/research_2022/PyTorch-GAN-master/PyTorch-GAN-master/implementations/cgan/data/model/TeacherR50.pth'
    #utilse.load_checkpoint(teacher_checkpoint, teacher_model)
    

    # Train the model with KD
    logging.info("Experiment - model version: {}".format(params.model_version))
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    logging.info("First, loading the teacher model and computing its outputs...")
    
    train_and_evaluate_kd(model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
                          metrics, params, args.model_dir, args.restore_file)
#     loss_fn=alexnets.loss_fn
#     train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params,
#                        args.model_dir, args.restore_file)
else:
    if params.model_version == "cnn":
        model = net.Net(params).cuda() if params.cuda else net.Net(params)
        #optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        # fetch loss function and metrics
        loss_fn = net.loss_fn
        metrics = net.metrics

    elif params.model_version == "resnet18":
        model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                              momentum=0.9, weight_decay=5e-4)
        # fetch loss function and metrics
        loss_fn = resnet.loss_fn
        metrics = resnet.metrics



    # elif params.model_version == "wrn":
    #     model = wrn.wrn(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
    #     model = model.cuda() if params.cuda else model
    #     optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
    #                           momentum=0.9, weight_decay=5e-4)
    #     # fetch loss function and metrics
    #     loss_fn = wrn.loss_fn
    #     metrics = wrn.metrics
        
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, params,
                       args.model_dir, args.restore_file)


# use GPU if available
params.cuda = torch.cuda.is_available()
model = net.Net(params).cuda() if params.cuda else net.Net(params)
utilse.load_checkpoint('experiments/base_cnn/best_pth',model)

# teacher_model = alexnet.AlexNet(num_classes=10)
# teacher_checkpoint = 'experiments/base_cnn/epoch399'
# utilse.load_checkpoint(teacher_checkpoint, teacher_model)

# %% Inference Attack HZ Class


class InferenceAttack_HZ(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes

        super(InferenceAttack_HZ, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(self.num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )

        self.labels = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(64 * 2, 256),

            nn.ReLU(),
            nn.Linear(256, 128),

            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        for key in self.state_dict():
            print(f'\t {key}')
            if key.split('.')[-1] == 'weight':
                nn.init.normal_(self.state_dict()[key], std=0.01)

            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

        self.output = nn.Sigmoid()

    def forward(self, x, labels):

        out_x = self.features(x)
        out_l = self.labels(labels)

        is_member = self.combine(torch.cat((out_x, out_l), 1))

        return self.output(is_member)


# %% Status Func

def report_str(batch_idx, data_time, batch_time, losses, top1, top5):
    batch = f'({batch_idx:4d})'
    time = f'Data: {data_time:.2f}s | Batch: {batch_time:.2f}s'
    loss_ac1 = f'Loss: {losses:.3f} | Top1: {top1 * 100:.2f}%'

    res = f'{batch} {time} || {loss_ac1}'

    if top5 is None:
        return res
    else:
        return res + f' | Top5: {top5 * 100:.2f}%'


from sklearn.metrics import top_k_accuracy_score
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
# %% privacy_train
# torch.Size([128, 3, 32, 32]) torch.Size([128]) torch.Size([128, 3, 32, 32]) torch.Size([128])
# PRED torch.Size([256, 10])
# infer_in torch.Size([256])
def privacy_train(trainloader, model, inference_model, criterion, optimizer, use_cuda, num_batchs):
    num_classes=10
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()

    inference_model.train()
    model.eval()
    # switch to evaluate mode

    end = time.time()
    first_id = -1
    for batch_idx, ((tr_input, tr_target), (te_input, te_target)) in trainloader:
        # measure data loading time
        if first_id == -1:
            first_id = batch_idx

        data_time.update(time.time() - end)
        
        #print(tr_input.shape, tr_target.shape,te_input.shape, te_target.shape)

        if use_cuda:
            tr_input = tr_input.cuda()
            te_input = te_input.cuda()
            tr_target = tr_target.cuda()
            te_target = te_target.cuda()

        v_tr_input = torch.autograd.Variable(tr_input)
        v_te_input = torch.autograd.Variable(te_input)
        v_tr_target = torch.autograd.Variable(tr_target)
        v_te_target = torch.autograd.Variable(te_target)

        # compute output
        model_input = torch.cat((v_tr_input, v_te_input))

        pred_outputs = model(model_input)
        #print("PRED",pred_outputs.shape)
        #y_hat

        infer_input = torch.cat((v_tr_target, v_te_target))
        #print("infer_in",infer_input.shape)
        #(y_hat)

        # TODO fix
        # mtop1, mtop5 = accuracy(pred_outputs.data, infer_input.data, topk=(1, 5))
        mtop1 = top_k_accuracy_score(y_true=infer_input.data.cpu(), y_score=pred_outputs.data.cpu(),
                                     k=1, labels=range(num_classes))

        mtop5 = top_k_accuracy_score(y_true=infer_input.data.cpu(), y_score=pred_outputs.data.cpu(),
                                     k=5, labels=range(num_classes))

        mtop1_a.update(mtop1, model_input.size(0))
        mtop5_a.update(mtop5, model_input.size(0))

        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), num_classes)) - 1)).cuda().type(torch.float)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.int64).view([-1, 1]).data, 1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        #ONE_hot y_hat

        attack_model_input = pred_outputs  # torch.cat((pred_outputs,infer_input_one_hot),1)
        member_output = inference_model(attack_model_input, infer_input_one_hot)
        #inf_model(y,y_hat)
        #member->?0/1

        is_member_labels = torch.from_numpy(
            np.reshape(
                np.concatenate((np.zeros(v_tr_input.size(0)), np.ones(v_te_input.size(0)))),
                [-1, 1]
            )
        ).cuda()

        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.float)
        #true_labels

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
        losses.update(loss.data.item(), model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx - first_id > num_batchs:
            break

        # plot progress
        if batch_idx % 50 == 0:
            #print("STUCK")
            print( losses.avg, top1.avg)
            #print(report_str(batch_idx, data_time.avg, batch_time.avg, losses.avg, top1.avg, None))

    return losses.avg, top1.avg

from sklearn.metrics import top_k_accuracy_score
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# torch.Size([128, 3, 32, 32]) torch.Size([128]) torch.Size([128, 3, 32, 32]) torch.Size([128])
# PRED torch.Size([256, 10])
# infer_in torch.Size([256])
def privacy_train(trainloader, model, inference_model, criterion, optimizer, use_cuda, num_batchs):
    num_classes=10
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()

    inference_model.train()
    model.eval()
    # switch to evaluate mode

    end = time.time()
    first_id = -1
    for batch_idx, ((tr_input, tr_target), (te_input, te_target)) in trainloader:
        # measure data loading time
        if first_id == -1:
            first_id = batch_idx

        data_time.update(time.time() - end)
        
        #print(tr_input.shape, tr_target.shape,te_input.shape, te_target.shape)

        if use_cuda:
            tr_input = tr_input.cuda()
            te_input = te_input.cuda()
            tr_target = tr_target.cuda()
            te_target = te_target.cuda()

        v_tr_input = torch.autograd.Variable(tr_input)
        v_te_input = torch.autograd.Variable(te_input)
        v_tr_target = torch.autograd.Variable(tr_target)
        v_te_target = torch.autograd.Variable(te_target)

        # compute output
        model_input = torch.cat((v_tr_input, v_te_input))

        pred_outputs = model(model_input)
        #print("PRED",pred_outputs.shape)
        #y_hat

        infer_input = torch.cat((v_tr_target, v_te_target))
        #print("infer_in",infer_input.shape)
        #(y_hat)

        # TODO fix
        # mtop1, mtop5 = accuracy(pred_outputs.data, infer_input.data, topk=(1, 5))
        mtop1 = top_k_accuracy_score(y_true=infer_input.data.cpu(), y_score=pred_outputs.data.cpu(),
                                     k=1, labels=range(num_classes))

        mtop5 = top_k_accuracy_score(y_true=infer_input.data.cpu(), y_score=pred_outputs.data.cpu(),
                                     k=5, labels=range(num_classes))

        mtop1_a.update(mtop1, model_input.size(0))
        mtop5_a.update(mtop5, model_input.size(0))

        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), num_classes)) - 1)).cuda().type(torch.float)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.int64).view([-1, 1]).data, 1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
        #ONE_hot y_hat

        attack_model_input = pred_outputs  # torch.cat((pred_outputs,infer_input_one_hot),1)
        member_output = inference_model(attack_model_input, infer_input_one_hot)
        #inf_model(y,y_hat)
        #member->?0/1

        is_member_labels = torch.from_numpy(
            np.reshape(
                np.concatenate((np.zeros(v_tr_input.size(0)), np.ones(v_te_input.size(0)))),
                [-1, 1]
            )
        ).cuda()

        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.float)
        #true_labels

        loss = criterion(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
        losses.update(loss.data.item(), model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx - first_id > num_batchs:
            break

        # plot progress
        if batch_idx % 50 == 0:
            #print("STUCK")
            print( losses.avg, top1.avg)
            #print(report_str(batch_idx, data_time.avg, batch_time.avg, losses.avg, top1.avg, None))

    return losses.avg, top1.avg


import torchvision.transforms as transforms

import torchvision.datasets as datasets

train_mean = np.array([125.307, 122.950, 113.865])
train_std = np.array([62.993, 62.089, 66.705])
test_mean = np.array([126.025, 123.708, 114.854])
test_std = np.array([62.896, 61.937, 66.706])

# Normalize mean std to 0..1 from 0..255
train_mean /= 255
train_std /= 255
test_mean /= 255
test_std /= 255

print(f'Hard code CIFAR10 train/test mean/std for next time')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(test_mean, test_std),
])

# TODO check loader for trainloader_private
trainset_private =datasets.CIFAR10(root='./data-cifar10', train=True,
        download=True, transform=transform_test)
trainloader_private = torch.utils.data.DataLoader(trainset_private, batch_size=params.batch_size, shuffle=True)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in [20, 40]:
        state['lr'] *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


def save_checkpoint_adversary(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_adversary_best.pth.tar'))

# inference_model = torch.nn.DataParallel(inferenece_model).cuda()

LR = 0.05
EPOCHS = 10
print('\tTotal params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

criterion = nn.CrossEntropyLoss()

criterion_attack = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

inference_model = InferenceAttack_HZ(10).cuda()

private_train_criterion = nn.MSELoss()

optimizer_mem = optim.Adam(inference_model.parameters(), lr=0.00001)

best_acc = 0.0
start_epoch = 0

# Train and val
for epoch in range(start_epoch, EPOCHS):
    #adjust_learning_rate(optimizer, epoch)

    print(f'\nEpoch: [{epoch + 1:d} | {EPOCHS:d}] ')

    train_private_enum = enumerate(zip(trainloader_private, dev_dl))
    privacy_loss, privacy_acc = privacy_train(train_private_enum, model, inference_model, criterion_attack, optimizer_mem, True, 100)
    print(f'Privacy Res: {privacy_acc * 100:.2f}% ')


