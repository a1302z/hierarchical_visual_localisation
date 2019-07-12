"""
Author: A. Ziller (Technical University of Munich)

This script evaluates position and orientation of given images in a scene. 
It follows the procedure of HF-NET which is shortly summarized as follows:
1) Find global neighbors
2) Cluster global neighbors
3) Find local features
4) Match local features of query image and neighboring cluster
5) Calculate 6-DoF pose

Please see Note in README
"""

import argparse
import os
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision import transforms
import cv2
import time
from collections import namedtuple
import sqlite3
from fnmatch import fnmatch
import nearpy

import read_model as rm
from dataset_loaders.utils import load_image
import models.netvlad_vd16_pitts30k_conv5_3_max_dag as netvlad


parser = argparse.ArgumentParser()
parser.add_argument('--global_method', default='NetVLAD', choices=['NetVLAD',], help='Which method to use for global features')
parser.add_argument('--local_method', default='Colmap', choices=['SiftOpenCV', 'Colmap'], help='Which method to use for local features')
parser.add_argument('--nearest_method', default='LSH', choices=['Exact', 'LSH'], help='Which method to use to find nearest global neighbors')
parser.add_argument('--global_resolution', default=256, type=int, help='Resolution on which nearest global neighbors are calculated')
parser.add_argument('--ratio_thresh', type=float, default=.75, help='Threshold for local feature matching in range [0.0, 1.0]. The higher it is the less similar matches have to be.')
parser.add_argument('--n_iter', type=int, default=5000, help='Number of iterations in RANSAC loop')
parser.add_argument('--reproj_error', type=float, default=8., help='Reprojection error of PnP-RANSAC loop')
parser.add_argument('--min_inliers', type=int, default=10, help='minimal number of inliers after PnP-RANSAC')
parser.add_argument('--n_neighbors', default=2, type=int, help='How many global neighbors are used')
parser.add_argument('--verify', action='store_true', help='Use given dataset to evaluate error on own images. Verifies that pipeline works')
parser.add_argument('--dataset_dir', default='data/AachenDayNight/images_upright/query/', help='Dataset directory')
parser.add_argument('--overfit', default=None, type=int, help='Limit number of queries')
parser.add_argument('--out_file', type=str, default='aachen_eval_.txt', help='Name of output file')
args = parser.parse_args()

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
        1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def colmap_image_to_pose(image):
    im_T_w = np.eye(4)
    im_T_w[:3, :3] = qvec2rotmat(image.qvec)
    im_T_w[:3, 3] = image.tvec
    w_T_im = np.linalg.inv(im_T_w)
    return w_T_im
def get_cursor(name):
    return sqlite3.connect(name).cursor()

def descriptors_from_colmap_db(cursor, image_id):
    cursor.execute('SELECT cols, data FROM descriptors WHERE image_id=?;',(image_id,))
    feature_dim, blob = next(cursor)
    desc = np.frombuffer(blob, dtype=np.uint8).reshape(-1, feature_dim)
    return desc


def keypoints_from_colmap_db(cursor, image_id):
    cursor.execute('SELECT cols, data FROM keypoints WHERE image_id=?;',(image_id,))
    cols, blob = next(cursor)
    kpts = np.frombuffer(blob, dtype=np.float32).reshape(-1, cols)
    return kpts

def get_kpts_desc(cursor, image_id):
    image_id = int(image_id)
    kpts = keypoints_from_colmap_db(cursor, image_id)[:, :2]
    desc = descriptors_from_colmap_db(cursor, image_id)
    return kpts, desc

def get_img_id(cursor, img_name):
    img_id, = next(cursor.execute('SELECT image_id FROM images WHERE name=?;',(img_name,)))
    return img_id

def get_img_id_dataset(cursor, dataset_id):
    db_query_name = 'db/%d.jpg'%dataset.get_img_id(dataset_id)
    return get_img_id(cursor, db_query_name)

def kpts_to_cv(kpts, kpt_size=1.0):
    cv_kpts = []
    for i, kpt in enumerate(kpts):
        cv_kpts.append(cv2.KeyPoint(x=kpt[0], y=kpt[1], _size=kpt_size))
    return cv_kpts

def get_files(path, pattern, not_pattern = None, printout=False):
    found = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern) and (not_pattern is None or not fnmatch(name, not_pattern)):
                found.append(os.path.join(path, name))
    if printout:
        print("Found %d files in path %s"%(len(found), path))
    return found

def time_to_str(t):
    out_str = ''
    if t > 86400:
        out_str += '%d days '%(t//86400)
        t %= 86400
    if t > 3600:
        out_str += '%d hours '%(t//3600)
        t %= 3600
    if t > 60:
        out_str += '%d minutes '%(t//60)
        t %= 60
    out_str += '%d seconds'%t
        
    return out_str


column_indents = [40, 1]
seperating_char = '| '

def print_column_entry(left_column, right_column, indents=column_indents, seperating_char=seperating_char):
    print('\t{:{}} {:>{}}{}'.format(left_column, indents[0], seperating_char, indents[1], right_column))



print('Configuration')
print_column_entry('Evaluation image directory', args.dataset_dir)
print_column_entry('Global method', args.global_method)
print_column_entry('Local method', args.local_method)
print_column_entry('Nearest neighbor method', args.nearest_method)
print_column_entry('Global resolution', args.global_resolution)
print_column_entry('k neighbors', args.n_neighbors)
print_column_entry('Matching threshold', args.ratio_thresh)
print_column_entry('Num iterations RANSAC', args.n_iter)
print_column_entry('Reprojection error', args.reproj_error)
print_column_entry('Minimum num inliers PnP', args.min_inliers)


print('Setup')
setup_time = time.time()
t = time.time()
points3d = rm.read_points3D_text('data/points3D.txt')
t = time.time() - t
print_column_entry('Read {} 3d points'.format(len(points3d)), time_to_str(t))

#cameras = rm.read_cameras_text('data/cameras.txt')
#print('\tRead %d cameras'%len(cameras))
t = time.time()
images = rm.read_images_text('data/images.txt')
t = time.time() - t
print_column_entry('Read {} images'.format(len(images)), time_to_str(t))
get_img = lambda i: np.array(load_image('data/AachenDayNight/images_upright/'+images[i].name))
database_cursor = get_cursor('data/AachenDayNight/aachen.db')
if args.local_method is 'Colmap':
    query_cursor = get_cursor('data/AachenDayNight/aachen.db') if args.verify else get_cursor('data/queries.db')
##create image clusters
t = time.time()
img_cluster = {}
for p_id in points3d.keys(): 
    img_ids = points3d[p_id].image_ids
    for img_id in img_ids:
        if img_id in img_cluster:
            img_cluster[img_id] |= set(img_ids)
        else:
            img_cluster[img_id] = set(img_ids)
t = time.time() - t
print_column_entry('Found {} cluster'.format(len(img_cluster)), time_to_str(t))
# Camera matrix
t = time.time()
camera_matrices = {}
query_intrinsics_files = ['data/AachenDayNight/queries/day_time_queries_with_intrinsics.txt',
                         'data/AachenDayNight/queries/night_time_queries_with_intrinsics.txt',
                         'data/AachenDayNight/database_intrinsics.txt']
for file_path in query_intrinsics_files:
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        for line in lines:
            # Format: `image_name SIMPLE_RADIAL w h f cx cy r`
            line = line.split(' ')
            img_path = line[0]
            f = float(line[4])
            cx = float(line[5])
            cy = float(line[6])
            rad_dist = float(line[7])
            A = np.array([[f, 0, cx],[0, f, cy], [0, 0, 1]])
            camera_matrices[img_path] = {'cameraMatrix': A, 'rad_dist':rad_dist}
t = time.time() - t
print_column_entry('Read camera matrices', time_to_str(t))
low_res_transform = transforms.Compose([transforms.Resize(args.global_resolution), transforms.CenterCrop(args.global_resolution), transforms.ToTensor() ])
setup_time = time.time() - setup_time

t = time.time()
if args.verify:
    query_images = [os.path.join(args.dataset_dir, images[i].name) for i in images]
    query_image_ids = [i for i in images]
else:
    query_images = get_files(args.dataset_dir, '*.jpg')
if args.overfit is not None:
    query_images = query_images[:args.overfit]
t = time.time() - t
print_column_entry('Found {} query images'.format(len(query_images)), time_to_str(t))
print_column_entry('Total time', time_to_str(setup_time))


print('Global Neighbors')

global_time = time.time()
t = time.time()
if args.global_method is 'NetVLAD':
    model = netvlad.vd16_pitts30k_conv5_3_max_dag(weights_path='data/teacher_models/netvlad_pytorch/vd16_pitts30k_conv5_3_max_dag.pth')
    model.eval()
    query_global_desc = []
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.cuda()
    print_column_entry('Calculating query descriptors', '')
    for cnt, img in enumerate(query_images):
        if cnt % (len(query_images)//5) == 0:
            print_column_entry('', '{}/{} query descriptors'.format(cnt, len(query_images)))
        if CUDA:
             query_global_desc.append(model(low_res_transform(load_image(img)).cuda().unsqueeze(0)).detach().cpu().squeeze(0).numpy())
        else:
            query_global_desc.append(model(low_res_transform(load_image(img)).unsqueeze(0)).detach().cpu().squeeze(0).numpy())
    query_global_desc = np.vstack(query_global_desc)
else:
    raise NotImplementedError('Global method not implemented')
t = time.time() - t

print_column_entry('{}-dim global query desc'.format(query_global_desc.shape), time_to_str(t))

t = time.time()
global_features_cursor = get_cursor('data/global_features.db')
global_features = []
image_ids = []
for row in global_features_cursor.execute('SELECT image_id, cols, data FROM global_features;'):
    global_features.append(np.frombuffer(row[2], dtype=np.float32).reshape(-1, row[1]))
    image_ids.append(row[0])
global_features = np.vstack(global_features)
global_features_cursor.close()
t = time.time() - t
print_column_entry('Database global features loaded', time_to_str(t))

t = time.time()
if args.nearest_method is 'LSH':
    engine = nearpy.Engine(global_features.shape[1], lshashes=[nearpy.hashes.RandomBinaryProjections('rbp', 10)])
    for i, v in enumerate(global_features):
        engine.store_vector(v, '%d'%i)
    indices = [engine.neighbours(d) for d in query_global_desc]
    indices = np.array([np.array([int(n[1]) for n in nbr])[:args.n_neighbors] for nbr in indices])
elif args.nearest_method is 'Exact':
    nbrs = NearestNeighbors(n_neighbors=args.n_neighbors).fit(global_features)
    distances, indices = nbrs.kneighbors(query_global_desc)
else:
    raise NotImplementedError('Nearest Method not implemented')
t = time.time() - t
print_column_entry('Nearest neighbors for all queries', time_to_str(t))


## delete data intense variables that are not needed anymore
del global_features, query_global_desc, model



## Local features matching and pose retrieval

image_times = []
if args.verify:
    errors = []
out_file = open(args.out_file, 'w')
print('Local feature matching and pose retrieval')
for query_id, query_name in enumerate(query_images):
    
    ## Make sure we have camera parameters.
    if args.verify:
        query_path = images[query_image_ids[query_id]].name
    else:
        query_path = os.path.join(*os.path.normpath(query_name).split(os.sep)[-4:])
    if query_path not in camera_matrices:
        continue
        
    individual_image_time = time.time()
    print_column_entry('Processing query image {}/{}'.format(query_id+1, len(query_images)), 'Expected remaining time: {}'.format(time_to_str(np.median(image_times)*(len(query_images)-query_id))) if query_id > 0 else '')
    print_column_entry('Query path', query_name)
    t = time.time()
    cluster_query = [img_cluster[image_ids[indices[query_id][0]]]]
    cluster_orig_ids = [image_ids[indices[query_id][0]]]
    for i, ind in enumerate(indices[query_id]):
        ind = image_ids[ind]
        if i == 0:
            continue
        point_set = img_cluster[ind]
        disjoint = False
        for j, c in enumerate(cluster_query):
            if ind in c:
                cluster_query[j] |= point_set
                disjoint = True
                break
        if not disjoint:
            cluster_orig_ids.append(ind)
            cluster_query.append(point_set)
    t = time.time() - t
    print_column_entry(' - Clustered neighbors', time_to_str(t))
    
    
    ## Local features
    t = time.time()
    if args.local_method is 'Colmap':
        ## query desc
        test_query_path = query_name.replace(args.dataset_dir, '')
        query_img_id = get_img_id(query_cursor, test_query_path)
        query_kpts, query_desc = get_kpts_desc(query_cursor, query_img_id)
        query_kpts = kpts_to_cv(query_kpts)
    else:
        raise NotImplementedError('Local feature extraction method not implemented')
    t = time.time() - t
    print_column_entry(' - Got query keypoints and descriptors', time_to_str(t))
    
    
    ## Matching
    t = time.time()
    matched_kpts_cv = []
    matched_pts = []
    matcher = cv2.BFMatcher.create(cv2.NORM_L2)
    pt_id_list = []
    data_descs = []
    for c in cluster_query:
        for img in c:
            db_id = img # get_img_id_dataset(database_cursor, img)
            img_name = images[db_id].name
            valid = images[db_id].point3D_ids > 0 
            data_kpts, data_desc = get_kpts_desc(database_cursor, db_id)
            data_kpts = kpts_to_cv(data_kpts[valid[:data_kpts.shape[0]]] - 0.5)
            pt_ids = images[db_id].point3D_ids[valid]
            pt_id_list.append(pt_ids)
            data_desc = data_desc[valid[:data_desc.shape[0]]]
            #print('Point Ids: {}\t Desc shape: {}'.format(len(pt_ids), data_desc.shape[0]))
            data_descs.append(data_desc)

    data_descs = np.concatenate(data_descs)
    pt_id_list = np.concatenate(pt_id_list)
    matches = matcher.knnMatch(data_descs,query_desc,k=2)
    good = []
    for i,(m,n) in enumerate(matches):
        if m.distance < args.ratio_thresh*n.distance:
            good.append(m)

    matched_kpts_cv = [query_kpts[i] for i in [g.trainIdx for g in good]]
    matched_pts = [pt_id_list[i] for i in [g.queryIdx for g in good]]

    t = time.time() - t
    matched_pts_xyz = np.stack([points3d[i].xyz for i in matched_pts])
    matched_keypoints = np.vstack([np.array([x.pt[0], x.pt[1]]) for x in matched_kpts_cv])
    print_column_entry(' - Number of matched points', matched_keypoints.shape[0])
    print_column_entry(' - Finished matching', time_to_str(t))
    
    
    ## Calculate pose
    t = time.time()
    cm = camera_matrices[query_path]
    camera_matrix = cm['cameraMatrix']
    distortion_coeff = cm['rad_dist']
    dist_vec = np.array([distortion_coeff, 0, 0, 0])
    
    success, R_vec, translation, inliers = cv2.solvePnPRansac(
        matched_pts_xyz, matched_keypoints, camera_matrix, dist_vec,
        iterationsCount=args.n_iter, reprojectionError=args.reproj_error,
        flags=cv2.SOLVEPNP_P3P)
    
    if success:
        inliers = inliers[:, 0] if len(inliers.shape) > 1 else inliers
        num_inliers = len(inliers)
        inlier_ratio = len(inliers) / len(matched_keypoints)
        success &= num_inliers >= args.min_inliers

        ret, R_vec, t = cv2.solvePnP(
                    matched_pts_xyz[inliers], matched_keypoints[inliers], camera_matrix,
                    dist_vec, rvec=R_vec, tvec=translation, useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE)
        success &= ret

        query_T_w = np.eye(4)
        query_T_w[:3, :3] = cv2.Rodrigues(R_vec)[0]
        query_T_w[:3, 3] = t[:, 0]
        w_T_query = np.linalg.inv(query_T_w)
    
    name = os.path.split(query_name)[-1]
    position = w_T_query[:3, 3]
    print_column_entry(' - Calculated position', position)
    
    if args.verify:
        gt = colmap_image_to_pose(images[query_image_ids[query_id]])[:3,3]
        error = np.linalg.norm(position-gt)
        error_str = '%.1f m'%error if error > 1e-1 else '%.1f cm'%(100.0*error)
        errors.append(error)
        print_column_entry(' - Groundtruth', gt)
        print_column_entry(' - Error', error_str)
        out_file.write('{} Error: {} CalcPos: {}\n'.format(name, error, position))
    else:
        if not success:
            print_column_entry(' - WARNING','Localization not successful!')
            out_file.write('{} {} {} {} {} {} {} {}\n'.format(name,0,0,0,0,0,0,0))
        else:
            quat = rotmat2qvec(w_T_query[:3,:3])
            out_file.write('{} {} {} {} {} {} {} {}\n'.format(name, quat[0], quat[1], quat[2], quat[3], position[0], position[1], position[2]))
    
    individual_image_time = time.time() - individual_image_time 
    print_column_entry('Finished image {}/{}'.format(query_id+1, len(query_images)), time_to_str(individual_image_time))
    image_times.append(individual_image_time)
out_file.close()
print('Stats')
print_column_entry('Setup time', time_to_str(setup_time))
print_column_entry('Average time per image', time_to_str(np.mean(image_times)))
print_column_entry('Median time per image', time_to_str(np.median(image_times)))
print_column_entry('Max image time', time_to_str(np.max(image_times)))
if args.verify:
    print_column_entry('Mean error', '{:.4f} m'.format(np.mean(errors)))
    print_column_entry('Median error', '{:.4f} m'.format(np.median(errors)))
    print_column_entry('Max error', '{:.4f} m'.format(np.max(errors)))