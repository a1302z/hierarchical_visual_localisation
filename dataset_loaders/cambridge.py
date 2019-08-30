import numpy as np
import collections
import os
import random
import torch
from torch.utils import data

from dataset_loaders.utils import load_image
from models.cirtorch_utils.datahelpers import default_loader, imresize


Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def list_to_np(l, dtype=float, sep=' '):
    return np.fromstring(' '.join(l), dtype=dtype, sep=sep)


class Cambridge(data.Dataset):
    def __init__(self, data_folder='data', scene='ShopFacade', demean=True,
                imsize=None, transform=None, triplet=False, deterministic=False
                ):
        self.demean = demean
        self.imsize = imsize
        self.transform = transform
        self.triplet = triplet
        self.deterministic=deterministic
        
        self.dir_path = os.path.join(data_folder, scene)
        self.nvm_file = os.path.join(self.dir_path, 'reconstruction.nvm')
        
        
        f = open(self.nvm_file, 'r')
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        self.num_cameras = int(lines[2])
        self.num_points = int(lines[self.num_cameras+4])
        point_lines = lines[self.num_cameras+5:self.num_points+self.num_cameras+5]
        camera_lines = lines[3:self.num_cameras+3]
        
        ## parse cameras
        # <Camera> = <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
        self.imgs = {}
        for i, line_str in enumerate(camera_lines):
            line = line_str.replace('\t',' ').split(' ')
            fn = str(line[0]).replace('.jpg', '.png')
            fl = float(line[1])
            quat = list_to_np(line[2:6])
            pos = list_to_np(line[6:9])
            dist = float(line[9])
            self.imgs[i] = Image(id=i, camera_id=i, qvec=quat, tvec=pos, name=fn, xys=[], point3D_ids=[])
        
        ## parse points
        # <Point>  = <XYZ> <RGB> <number of measurements> <List of Measurements>
        # <Measurement> = <Image index> <Feature Index> <xy>
        self.pts = {}
        #print(point_lines[0])
        for i, line_str in enumerate(point_lines):
            line = line_str.replace('\t',' ').split(' ')
            xyz = list_to_np(line[0:3])
            rgb = list_to_np(line[3:6], dtype=int)
            num_meas = int(line[6])
            img_ids = []
            pt_2d_idxs = []
            for meas in range(num_meas):
                offset = 7+meas*4
                img_id = int(line[offset])
                feat_id = int(line[offset+1])
                xy = list_to_np(line[offset+2:offset+4])
                self.imgs[img_id].xys.append(xy)
                self.imgs[img_id].point3D_ids.append(i)
                img_ids.append(img_id)
                pt_2d_idxs.append(feat_id)
            #"Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
            self.pts[i] = Point3D(id=i, xyz = xyz, rgb=rgb, error=-1, image_ids=img_ids, point2D_idxs=pt_2d_idxs)
        
    def __load_items__(self, index):
        img = load_image(os.path.join(self.dir_path, self.imgs[index].name))
        imfullsize = max(img.size)
        if self.imsize is not None:
            img = imresize(img, self.imsize)

        if self.transform is not None:
            img = self.transform(img)
        
        pt_ids = self.imgs[index].point3D_ids
        pts3d = np.vstack([self.pts[i].xyz for i in pt_ids])
        mean, std = np.mean(pts3d, axis=0), np.std(pts3d, axis=0)
        if self.demean:
            pts3d -= mean
            pts3d /= std
        pts3d = torch.from_numpy(pts3d).float()
        return img, pts3d
    
        
    def __getitem__(self, index):
        img, pts3d = self.__load_items__(index)
        
            
        if self.triplet:
            if self.deterministic:
                random.seed(index)
            j = random.randint(0, self.num_cameras - 1)
            while j == index:
                j = random.randint(0, self.num_cameras - 1)
            img_triplet, pts3d_triplet = self.__load_items__(j)
        else:
            img_triplet, pts3d_triplet = None, None
            
        
        return img, pts3d, img_triplet, pts3d_triplet
    
    def __len__(self):
        return self.num_cameras
    
    
if __name__ == '__main__':
    ct = Cambridge()
    print(len(ct))
    img, pts, img_triplet, pts_triplet = ct[0]
    import matplotlib.pyplot as plt
    print(img)
    plt.imshow(img)
    plt.show()
    print(pts.shape)
