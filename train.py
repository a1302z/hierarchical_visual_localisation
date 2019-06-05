"""
This script is processing user input and setting up training process accordingly. 
"""

import configparser
import argparse
import torch
import torchvision.models as pretrained_models
import torchvision.datasets as predefined_datasets
import torchvision.transforms as transforms

import common.train_model as train_model
import models.hfnet as hfnet

## Read command line arguments
parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', required=True, choices=['Aachen'], help='Dataset to choose')
parser.add_argument('--config', required=True, help='Config for training')
parser.add_argument('--device', default='0', help='Specify computing device')
parser.add_argument('--hostname', default='localhost', help='Hostname for visdom logging')
parser.add_argument('--port', default=8833, help='Port for visdom logging')
parser.add_argument('--exp_name', default='Experiment', help='Define experiment name')
args = parser.parse_args()


## Read config
cp = configparser.ConfigParser()
cp.read(args.config)

## Initialize dataset & preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = predefined_datasets.MNIST('data/MNIST', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=cp['Training'].getint('batch_size', 4))
do_val = cp['Training'].getboolean('validation', True)
if do_val:
    val_set = predefined_datasets.MNIST('data/MNIST', download=True, transform=transform, train=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cp['Training'].getint('batch_size', 4))


## Initialize model
model = hfnet.Classifier(pretrained_models.resnet18(pretrained=False))
#print(model.pt)



## Start training
train_model.train_classifier(model, dataloader, cp, args,
                            device = args.device, hostname=args.hostname, port=args.port,
                            exp_name = args.exp_name, val_set = val_set if do_val else None
                            )


