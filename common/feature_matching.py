import cv2
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import nearpy


def to_unit_vector(x, method='approx_torch', cuda=False):
    if method == 'approx':
        return __to_unit_numpy__(x)
    elif method == 'approx_torch':
        return __to_unit_torch__(x, cuda)
    else:
        raise NotImplementedError('Method not implemented')
            
def __to_unit_numpy__(x):
    return x.astype(np.float32)/np.linalg.norm(x.astype(np.float32),axis=-1, keepdims=True)
    
    
def __to_unit_torch__(x, cuda):
    if cuda:
        x = torch.from_numpy(x).float().cuda()
    else:
        x = torch.from_numpy(x).float()
    return (x.transpose(0, 1) / torch.norm(x, p=2, dim=1)).transpose(0,1)

class GlobalMatcher:
    def __init__(self, method, n_neighbors, unit_vectors=False, buckets=5):
        if method == 'LSH':
            self.match = lambda x,y: self.__LSH__(x, y, buckets, n_neighbors)
        elif method == 'exact':
            self.match = lambda x,y: self.__exact(x, y, n_neighbors)
        elif method == 'approx':
            self.match = lambda x,y: self.__approx__(y, x, n_neighbors, unit_vectors, torch.cuda.is_available())
        elif method == 'approx_cpu':
            self.match = lambda x,y: self.__approx_np_global__(x, y, n_neighbors, unit_vectors)
        else:
            raise NotImplementedError('Global matching method not implemented')
            
    def __approx__(self, global_features, query_global_desc, n_neighbors, unit_vectors=False, cuda=False):
        if not unit_vectors:
            global_features = __to_unit_torch__(global_features, cuda=cuda)
            query_global_desc = __to_unit_torch__(query_global_desc, cuda=cuda)
        with torch.no_grad():
            d = 1. - torch.matmul(global_features, query_global_desc.transpose(0,1))
            values, indices = torch.topk(d, n_neighbors, dim=1, largest=False, sorted=True)
            return indices.cpu().numpy()
    def __approx_np_global__(self, x,y,n_neighbors, unit_vectors):
        ##normalize
        if not unit_vectors:
            x = __to_unit_numpy__(x)
            y = __to_unit_numpy__(y)
        x = x.astype(np.float32)/np.linalg.norm(x.astype(np.float32),axis=-1, keepdims=True)
        y = y.astype(np.float32)/np.linalg.norm(y.astype(np.float32),axis=-1, keepdims=True)
        d = 1.-np.matmul(x, y.T)
        k = np.argpartition(d, n_neighbors, axis=1)[:,:n_neighbors]
        intm = np.argsort(np.array([d[i, k[i]] for i in range(k.shape[0])]), axis=1)
        k = np.array([k[i,a] for i, a in enumerate(intm)])
        return k
            
    def __exact__(self, global_features, query_global_desc, n_neighbors):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(global_features)
        distances, indices = nbrs.kneighbors(query_global_desc)
        return indices
            
            
    def __LSH__(self, global_features, query_global_desc, buckets=5, n_neighbors=3):
        engines = {}
        engines[buckets] = nearpy.Engine(global_features.shape[1], lshashes=[nearpy.hashes.RandomBinaryProjections('rbp', buckets)])
        for i, v in enumerate(global_features):
            engines[buckets].store_vector(v, '%d'%i)
        indices = []
        for d in query_global_desc:
            nbr = engines[buckets].neighbours(d)
            if len(nbr) > (n_neighbors):
                indices.append(np.array([int(n[1]) for n in nbr]))
            else:
                b = buckets
                while (len(nbr) <= n_neighbors):
                    b = b // 2
                    if b not in engines:
                        engines[b] = nearpy.Engine(global_features.shape[1], lshashes=[nearpy.hashes.RandomBinaryProjections('rbp', b)])
                        for i, v in enumerate(global_features):
                            engines[b].store_vector(v, '%d'%i)
                    nbr = engines[b].neighbours(d)
                    indices.append(np.array([int(n[1]) for n in nbr]))    
        return np.array(indices)

class LocalMatcher:
    def __init__(self, ratio_thresh=.75, method='OpenCV', unit_vectors=False):
        self.ratio_thresh = ratio_thresh
        self.method = method
        if method == 'OpenCV':
            self.match = lambda x,y: self.__bf_matching__(x, y)
        elif method == 'approx':
            self.match = lambda x,y: self.__approx_np__(y, x, unit_vectors)
        elif method == 'approx_torch':
            self.match = lambda x,y: self.__approx_torch__(x,y,torch.cuda.is_available(), unit_vectors)
        else:
            raise NotImplementedError('Requested Matching method not implemented')
            

        

    """
    For exact kNN matching the OpenCV method was unbeatable regarding performance.
    """
    def __bf_matching__(self, x, y):
        matcher = cv2.BFMatcher.create(cv2.NORM_L2)
        matches = matcher.knnMatch(y, x, k=2)
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < self.ratio_thresh*n.distance:
                good.append(m)
        matches = np.array([[g.trainIdx, g.queryIdx] for g in good])
        return matches
    
    """
    An approximation that only considers direction of a feature vector but not its length
    """
    def __approx_np__(self, x, y, is_unit_vector=False):
        ##normalize
        if not is_unit_vector:
            x = __to_unit_numpy__(x)
            y = __to_unit_numpy__(y)
        d = 1.-np.matmul(y, x.T)
        ap = np.argpartition(d, 2, axis=1)[:,:2]
        intm = np.argsort(np.array([d[i, ap[i]] for i in range(ap.shape[0])]), axis=1)
        k = np.array([ap[i,a] for i, a in enumerate(intm)])
        matches = []
        for i, (m, n) in enumerate(k):
            if d[i, m] < self.ratio_thresh**2*d[i, n]:
                matches.append([i, m])
        return np.array(matches)
    

    """
    Same idea as approx numpy. Especially fast if gpu/cuda available.
    """
    def __approx_torch__(self, x,y,cuda,is_unit_vector=False):
        if not is_unit_vector:
            x = __to_unit_torch__(x, cuda)
            y = __to_unit_torch__(y, cuda)
        with torch.no_grad():
            y = y.transpose(0,1)
            #print('Memory needed: {:.1f} GiB'.format(round((x.element_size() * x.nelement())/(1024*1024*1024), 1)))
            #print('Memory needed: {:.1f} GiB'.format((y.element_size() * y.nelement())/(1024*1024*1024)))
            #d = 1. - torch.einsum('ik,kj->ij', [x, y])
            d = 1. - torch.matmul(x, y)
            values, indices = torch.topk(d, 2, dim=1, largest=False, sorted=True)
            valid = values[:,0] < self.ratio_thresh**2*values[:,1]
            if not torch.any(valid):
                return np.array([])
            indices_valid = indices[valid][:,0]
            if cuda:
                valid_indices = torch.arange(valid.size()[0]).cuda()[valid]
            else:
                valid_indices = torch.arange(valid.size()[0])[valid]
            ret = torch.stack([torch.Tensor([valid_indices[i], indices_valid[i]]) for i in range(valid_indices.shape[0])])
            return ret.cpu().numpy().astype(np.int64) if cuda else ret.numpy().astype(np.int64)
        
        