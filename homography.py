#!/usr/bin/python3

import numpy as np
import scipy.linalg as linalg

def homography(xy_src, xy_dst):
    '''
    xy_src  : Nx2 Matrix corrsponding to source points
                N is equal to the number of points
                Each row contains [x, y]
                
    xy_dst  : Nx2 Matrix corrsponding to destination points
                N is equal to the number of points
                Each row contains [x, y]  

    returns : 3x3 homography matrix such that [ xy_dst = H * xy_src ]
    
    Reference: https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf
    '''
    
    src = np.asarray(xy_src, dtype=np.float32)
    dst = np.asarray(xy_dst, dtype=np.float32)
    
    if src.shape != dst.shape:
        raise Exception('Source and Destination dimensions must be same')
    if src.shape[0] < 4:
        raise Exception('At least 4 set of points is required to compute homography')
    if src.shape[1] != 2:
        raise Exception('Each row in Source and Destination matrices should contain [x, y] points')
        
    n_points = src.shape[0]
    
    # Form matrix A
    A = np.zeros((n_points*2, 9), dtype=np.float32)
    for i in range(n_points):
        # A[i] = [-x1, -y1, -1, 0, 0, 0, x2x1, x2y1, x2]
        A[2*i] = [-1.0*src[i][0], -1.0*src[i][1], -1, 0, 0, 0, dst[i][0]*src[i][0], dst[i][0]*src[i][1], dst[i][0]]
        # A[i+1] = [0, 0, 0, -x1, -y1, -1, y2x1, y2y1, y2]
        A[2*i+1] = [0, 0, 0, -1.0*src[i][0], -1.0*src[i][1], -1, dst[i][1]*src[i][0], dst[i][1]*src[i][1], dst[i][1]]
        
    U, Sigma, V_transpose = linalg.svd(A)
    
    ## Form homography matrix
    # homography matrix corresponds to the column of V
    # corresponding to the smallest value of Sigma.
    # linalg.svd() returns Sigma in decreasing order
    # hence homography matrix will can be chosesn as
    # the last column of V or last row of V_transpose
    H = np.reshape(V_transpose[-1], (3,3))
    
    # Compute inverse of homography
    H_inverse = linalg.inv(H)
    
    return H, H_inverse
    
def applyHomography(xy, H):
    '''
    xy      : 1x2 Matrix corresponding to the source point in the form [x1, y1]
    H       : Homography Matrix
    returns : [x2, y2] points mapped on the destination plane corresponding 
                to the source points such that { [x2, y2] = H * [x1, y1] }
    '''
    xyz = xy
    if len(xyz) == 2:
        # Add z=1
        xyz.append(1)
    else:
        raise Exception('expected input of form [x, y]') 
        
    # convert to a column vector
    xyz_transpose = np.transpose(np.asarray(xyz, dtype=np.float32))
    
    # Apply homography matrix
    new_xyz = np.matmul(H, xyz_transpose)
    
    # Homogeneous to Cartesian conversion
    z_ = new_xyz[2]
    x = int(new_xyz[0]/z_)
    y = int(new_xyz[1]/z_)
    
    return [x, y]