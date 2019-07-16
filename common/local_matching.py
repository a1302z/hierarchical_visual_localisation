import cv2
import numpy as np
import torch

class Matcher:
    def __init__(self, ratio_thresh=.75, method='OpenCV', unit_vectors=False):
        self.ratio_thresh = ratio_thresh
        self.method = method
        if method == 'OpenCV':
            self.match = lambda x,y: self.__bf_matching__(x, y)
        elif method == 'approx':
            self.match = lambda x,y: self.__approx_np__(x,y,unit_vectors)
        elif method == 'approx_torch':
            self.match = lambda x,y: self.__approx_torch__(x,y,torch.cuda.is_available(), unit_vectors)
        else:
            raise NotImplementedError('Requested Matching method not implemented')
            
    def to_unit_vector(self, x, cuda=False):
        if self.method == 'approx_numpy':
            return self.__to_unit_numpy__(x)
        elif self.method == 'approx_torch':
            return self.__to_unit_torch__(x, cuda)
            
    def __to_unit_numpy__(self, x):
        return x.astype(np.float32)/np.linalg.norm(x.astype(np.float32),axis=-1, keepdims=True)
    
    
    def __to_unit_torch__(self, x, cuda):
        if cuda:
            x = torch.from_numpy(x).float().cuda()
        else:
            x = torch.from_numpy(x).float()
        return (x.transpose(0, 1) / torch.norm(x, p=2, dim=1)).transpose(0,1)
        

    """
    For exact kNN matching the OpenCV method was unbeatable regarding performance.
    """
    def __bf_matching__(self, x, y):
        matcher = cv2.BFMatcher.create(cv2.NORM_L2)
        matches = matcher.knnMatch(x, y, k=2)
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < self.ratio_thresh*n.distance:
                good.append(m)
        matches = np.array([[g.trainIdx, g.queryIdx] for g in good])
        return matches
    
    """
    An approximation that only considers direction of a feature vector but not its length
    """
    def __approx_np__(self, x,y,is_unit_vector=False):
        ##normalize
        if not is_unit_vector:
            x = self.__to_unit_numpy__(x)
            y = self.__to_unit_numpy__(y)
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
            x = self.__to_unit_torch__(x, cuda)
            y = self.__to_unit_torch__(y, cuda)
        with torch.no_grad():
            d = 1. - torch.matmul(x, y.transpose(0,1))
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