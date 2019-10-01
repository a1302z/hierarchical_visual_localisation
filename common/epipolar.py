import numpy as np

def expand_homo_ones(arr2d, axis=1):
    """Raise 2D array to homogenous coordinates
    Args:
        - arr2d: (N, 2) or (2, N)
        - axis: the axis to append the ones
    """    
    if axis == 0:
        ones = np.ones((1, arr2d.shape[1]))
    else:
        ones = np.ones((arr2d.shape[0], 1))      
    return np.concatenate([arr2d, ones], axis=axis)

def symmetric_epipolar_distance(pts1, pts2, F, sqrt=False):
    """Calculate symmetric epipolar distance between 2 sets of points
    Args:
        - pts1, pts2: points correspondences in the two images, 
          each has shape of (num_points, 2)
        - F: fundamental matrix that fulfills x2^T*F*x1=0, 
          where x1 and x2 are the correspondence points in the 1st and 2nd image 
    Return:
        A vector of (num_points,), containing root-squared epipolar distances
          
    """
    
    # Homogenous coordinates
    pts1 = expand_homo_ones(pts1, axis=1)
    pts2 = expand_homo_ones(pts2, axis=1)
    
    # l2=F*x1, l1=F^T*x2
    l2 = np.dot(F, pts1.T) # 3,N
    l1 = np.dot(F.T, pts2.T)
    dd = np.sum(l2.T * pts2, 1)  # Distance from pts2 to l2
    
    if sqrt:
        # Adopted from  DFM paper.
        d = np.abs(dd) * (1.0 / np.sqrt(l1[0, :] ** 2 + l1[1, :] ** 2) + 1.0 / np.sqrt(l2[0, :] ** 2 + l2[1, :] ** 2))
    else: # Original version    
        d = dd ** 2 * (1.0 / (l1[0, :] ** 2 + l1[1, :] ** 2) + 1.0 /(l2[0, :] ** 2 + l2[1, :] ** 2))
    return d

