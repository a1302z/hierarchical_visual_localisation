"""
Create NetVLAD predictions for dataset
"""


import argparse
import numpy as np
import torch
from torchvision import transforms
import os
import sqlite3
import time
from datetime import timedelta
import dataset_loaders.txt_to_db import get_images

import models.netvlad_vd16_pitts30k_conv5_3_max_dag as netvlad
from dataset_loaders.utils import load_image


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default='data', help='Specify directory to save NetVLAD prediction')
parser.add_argument('--model_path', default='data/teacher_models/netvlad_pytorch/vd16_pitts30k_conv5_3_max_dag.pth', help='Path to trained NetVLAD weights')
parser.add_argument('--data_path', default='data/AachenDayNight/images_upright/', help='Specify path to AachenDayNight data')
parser.add_argument('--out_file', default='data/netvlad', help='Specify name for output file')
parser.add_argument('--resize', type=int, default=256, help='Size to resize and crop input images')
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

c = sqlite3.connect(args.out_file+'.db')
try:
    res = c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='global_features';")
    if(res.fetchone()[0] == 1):
        print('Database already exists - Please either delete or use other name')
        exit()
    else:
        c.execute('''CREATE TABLE global_features (image_id INTEGER PRIMARY_KEY NOT NULL, cols INTEGER, data BLOB)''')
except sqlite3.Error as e:
    print(e)
    exit()

print('Loading image information')
images = get_images()
get_img = lambda i: load_image(args.data_path+images[i].name)

print('Found %d images'%len(images))
predictions = []

times = []
model.eval()
print('Start processing images')
for cnt, i in enumerate(images):
    if cnt % len(images) // 10 == 0:
        c.commit()
        print('%4d/%d'%(i, len(images)), end='\t')
        if len(times) > 0:
            t = np.median(times) * (len(images)-i)
            print("Estimated remaining time: %s"%str(timedelta(seconds=t)))
        else:
            print('')
    t1 = time.time()
    data = transform(get_img(i)).unsqueeze(0)
    if CUDA:
        data = data.cuda()
    pred = model(data)
    pred = pred.squeeze(0).detach().cpu().numpy()
    sh = pred.shape[0]
    c.execute("INSERT INTO global_features VALUES (?,?,?)", [i, sh, pred])
    times.append(time.time()-t1)
    if args.overfit is not None and cnt > args.overfit:
        break

c.commit()    
c.close()
print('Fin')
