import numpy as np
import cv2
import torch
import sqlite3
import argparse
from tqdm import tqdm
import models.demo_superpoint as superpoint
from models.d2net.extract_features import d2net_interface
from dataset_loaders.txt_to_db import get_images
from evaluate import keypoints_from_colmap_db

parser = argparse.ArgumentParser()
parser.add_argument('--local_method', default='Superpoint', type=str, choices=['Superpoint', 'D2'], help='Which model descriptors should be precalculated')
parser.add_argument('--model_path', default='data/teacher_models/superpoint_v1.pth', help='Path to trained superpoint weights')
parser.add_argument('--out_file', default='data/superpoint', help='Specify name for output file')
args = parser.parse_args()

c = sqlite3.connect(args.out_file+'.db')
try:
    res = c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='local_features';")
    if(res.fetchone()[0] == 1):
        print('Database {} already exists - Please either delete or use other name'.format(args.out_file))
        exit()
    else:
        c.execute('''CREATE TABLE local_features (image_id INTEGER PRIMARY_KEY NOT NULL, points BLOB, cols INTEGER, desc BLOB)''')
except sqlite3.Error as e:
    print(e)
    exit()

if args.local_method == 'Superpoint':
    model = superpoint.SuperPointFrontend(weights_path=args.model_path,
                          nms_dist=4, conf_thresh=0.015, nn_thresh=.7, cuda=torch.cuda.is_available())
elif args.local_method == 'D2':
    model = d2net_interface(model_file=args.model_path, use_relu=False)
else:
    raise NotImplementedError('Not implemented method')

print('Load images')
images = get_images()
keypoint_database = sqlite3.connect('data/AachenDayNight/aachen.db').cursor()
print('Calculate descriptors')
for db_id in tqdm(images.keys(), total=len(images.keys())):
    #if (i % 500) == 0:
    #    print('{}/{}'.format(i, len(images.keys())))
    img_name = images[db_id].name
    valid = images[db_id].point3D_ids > 0
    data_kpts = keypoints_from_colmap_db(keypoint_database, db_id)
    data_kpts = data_kpts[valid[:data_kpts.shape[0]]] - 0.5
    path_to_img = 'data/AachenDayNight/images_upright/'+img_name
    if args.local_method == 'Superpoint':
        cv_img = cv2.imread(path_to_img, 0).astype(np.float32)/255.0
        _, data_desc, _ = model.run(cv_img, points=data_kpts)
        data_desc = data_desc.T
    elif args.local_method == 'D2':
        fixed_kpts = np.flip(data_kpts.copy(), axis=1)
        data_desc = model.get_features(path_to_img, fixed_kpts)
        #print('Img: {}\tData kpts: {}\t Data descs: {}'.format(img_name, data_kpts.shape[0], data_desc.shape[0]))
    data_desc = data_desc.copy()
    c.execute("INSERT INTO local_features VALUES (?,?,?,?)", [db_id, data_kpts, data_kpts.shape[0], data_desc])
    #if i > 25:
    #    break
print('Store order')    
c.execute('CREATE INDEX tag_ids ON local_features (image_id);')

c.commit()    
c.close()
keypoint_database.close()