import numpy as np
import cv2
import torch
import sqlite3
import argparse
import models.demo_superpoint as superpoint
from dataset_loaders.txt_to_db import get_images

parser = argparse.ArgumentParser()
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


extractor = superpoint.SuperPointFrontend(weights_path=args.model_path,
                          nms_dist=4, conf_thresh=0.015, nn_thresh=.7, cuda=torch.cuda.is_available())

print('Load images')
images = get_images()
keypoint_database = sqlite3.connect('data/AachenDayNight/aachen.db').cursor()

for i, db_id in enumerate(images.keys()):
    if (i % (len(images.keys())//5)) == 0:
        print('{}/{}'.format(i, len(images.keys())))
    img_name = images[db_id].name
    valid = images[db_id].point3D_ids > 0 
    keypoint_database.execute('SELECT cols, data FROM keypoints WHERE image_id=?;',(db_id,))
    cols, blob = next(keypoint_database)
    data_kpts = np.frombuffer(blob, dtype=np.float32).reshape(-1, cols)
    data_kpts = data_kpts[valid[:data_kpts.shape[0]]] - 0.5
    path_to_img = 'data/AachenDayNight/images_upright/'+img_name
    cv_img = cv2.imread(path_to_img, 0).astype(np.float32)/255.0
    _, data_desc, _ = extractor.run(cv_img, points=data_kpts)
    data_desc = data_desc.T
    data_desc = data_desc.copy()
    c.execute("INSERT INTO local_features VALUES (?,?,?,?)", [db_id, data_kpts, data_kpts.shape[0], data_desc])
    

c.commit()    
c.close()
keypoint_database.close()