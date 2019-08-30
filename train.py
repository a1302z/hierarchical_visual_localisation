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
import numpy as np
import random

from models.cirtorch_utils.genericdataset import PointCloudImagesFromList, PointCloudSplit, PCDataLoader
from models.decoder import DecoderV2

import models.pointnet2_classification as ptnet
from models.cirtorch_network import init_network, extract_vectors
import torch_geometric
from torch_geometric.data import Data


## Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['Aachen', 'ShopFacade'], help='Dataset to choose')
parser.add_argument('--config', required=True, help='Config for training')
parser.add_argument('--device', default='0', help='Specify computing device')
parser.add_argument('--hostname', default='localhost', help='Hostname for visdom logging')
parser.add_argument('--port', default=8833, help='Port for visdom logging')
parser.add_argument('--exp_name', default='Experiment', help='Define experiment name')
parser.add_argument('--decode', action='store_true', help='Use autoencoder/decoder as additional model part')
args = parser.parse_args()


## Read config
cp = configparser.ConfigParser()
cp.read(args.config)

seed = cp['Training'].getint('random_seed', 0)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True




overfit = cp['Training'].getint('overfit', -1)

stats_file = cp['DataParams'].get('stats_file')
stats = np.loadtxt(stats_file)
normalize = transforms.Normalize(
   mean=stats[0],
   std=stats[1]
)
transform = transforms.Compose([
        transforms.Resize(cp['DataParams'].getint('input_size', 1024)),
        transforms.CenterCrop(cp['DataParams'].getint('input_size', 1024)),
        transforms.ToTensor(),
        normalize
])
## Initialize dataset & preprocessing
if args.dataset == 'Aachen':
    from dataset_loaders.txt_to_db import get_images, get_points
    images = get_images()
    points3d = get_points()
    dataset = PointCloudImagesFromList('data/AachenDayNight/images_upright', images, points3d,
                                       imsize=cp['DataParams'].getint('input_size', 1024), transform=transform, 
                                       triplet=True, min_num_points=cp['Training'].getint('min_num_3dpts', 100), 
                                       max_std_std=0.1, within_std=2.0,deterministic= overfit > 0
                                      )
elif args.dataset == 'ShopFacade':
    from dataset_loaders import cambridge
    dataset = cambridge.Cambridge(imsize=cp['DataParams'].getint('input_size', 1024), transform=transform, 
                                       triplet=True, deterministic=(overfit > 0))
    
do_val = cp['Training'].getboolean('validation', True)
if do_val:
    dataset_split = PointCloudSplit(dataset, val=False, split=cp['Training'].getint('val_split', 10))
    dataloader = PCDataLoader(dataset_split, batch_size=cp['Training'].getint('batch_size', 10) if overfit < 0 else 1, shuffle=True if overfit < 0 else False)
        
    if overfit >= 0:
        val_loader = dataloader # PCDataLoader(dataset_split, batch_size=1, shuffle=False)
    else:
        val_set = PointCloudSplit(dataset, val=True, split=cp['Training'].getint('val_split', 10))
        val_loader = PCDataLoader(val_set, batch_size=1, shuffle=False)
else:
    dataloader = PCDataLoader(dataset, batch_size=cp['Training'].getint('batch_size', 10), shuffle=True)
        
"""
elif args.dataset == 'MNIST': ## this was for validating training pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = predefined_datasets.MNIST('data/MNIST', download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cp['Training'].getint('batch_size', 4))
    do_val = cp['Training'].getboolean('validation', True)
    if do_val:
        val_set = predefined_datasets.MNIST('data/MNIST', download=True, transform=transform, train=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=cp['Training'].getint('batch_size', 4))
"""

## Initialize model
##model = hfnet.Classifier(pretrained_models.resnet18(pretrained=False))
#print(model.pt)
pointnet = ptnet.NetAachen()
cir = init_network({'architecture' : 'resnet34'})
decoder = DecoderV2(input_size=512) if args.decode else None


## Start training
train_model.train_triplet(pointnet, cir, dataloader, cp, args,
                            device = args.device, hostname=args.hostname, port=args.port,
                            exp_name = args.exp_name, val_set = val_loader if do_val else None,
                            decoder = decoder
                            )


