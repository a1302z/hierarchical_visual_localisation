"""
Create NetVLAD predictions for dataset
"""


import argparse
import numpy as np
import torch
from torchvision import transforms
import os
import h5py
import time
from datetime import timedelta

import models.netvlad_vd16_pitts30k_conv5_3_max_dag as netvlad
import dataset_loaders.aachen as aachen


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='data', help='Specify directory to save NetVLAD prediction')
parser.add_argument('--model_path', default='data/teacher_models/netvlad_pytorch/vd16_pitts30k_conv5_3_max_dag.pth', help='Path to trained NetVLAD weights')
parser.add_argument('--data_path', default='data/AachenDayNight', help='Specify path to AachenDayNight data')
parser.add_argument('--out_file', default='test', help='Specify name for output file')
parser.add_argument('--resize', type=int, default=1063, help='Size to resize and crop input images')
parser.add_argument('--overfit', type=int, default=None, help='Only use n data points')
args = parser.parse_args()


## params
CUDA = torch.cuda.is_available()

## Init model
model = netvlad.vd16_pitts30k_conv5_3_max_dag(weights_path=args.model_path)
if CUDA:
    model = model.cuda()

## Init dataset
transform = transforms.Compose([
    transforms.Resize(args.resize),
    transforms.CenterCrop(args.resize),
    transforms.ToTensor()])
dataset = aachen.AachenDayNight(args.data_path, True, train_split=-1,seed=0,input_types='img', output_types=[], real=True,transform=transform, verbose=False, overfit=args.overfit)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

print('%d datapoints'%len(dataloader))
predictions = []

out_file_name = os.path.join('data', args.out_file+'.hdf5')
out_file = h5py.File(out_file_name, "w")
width = model(dataset[0][0].unsqueeze(0).cuda()).detach().cpu().squeeze(0).size()[0] + 1
results = out_file.create_dataset("results", (len(dataloader), width), compression="gzip")
times = []
for i, (data, _) in enumerate(dataloader):
    if i % 20 == 0:
        print('%d/%d'%(i, len(dataloader)), end='\t')
        if len(times) > 0:
            t = np.median(times) * (len(dataloader)-i)
            print("Estimated remaining time: %s"%str(timedelta(seconds=t)))
        else:
            print('')
    t1 = time.time()
    if CUDA:
        data = data.cuda()
    pred = model(data)
    pred = pred.squeeze(0).detach().cpu().numpy()
    sh = pred.shape[0]
    #print(pred.shape)
    if sh + 1 > width:
        raise RuntimeError('Prediction larger than hdf5')
    results[i,0] = sh
    results[i, 1:sh+1] = pred
    times.append(time.time()-t1)
    
out_file.close()
"""
verification = np.load('data/test.npy', allow_pickle=True)
print(verification.shape)
for i, v in enumerate(verification):
    print(v.shape)
    print(np.linalg.norm(v-predictions[i]))
"""    
        
        
        
print('Fin')
    