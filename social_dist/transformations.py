import numpy as np
import cv2

def getWarpingMatrix(pts1, pts2):
    """
    Returns warping matrix according to the datapoints
    """
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return matrix

def warpPerspective(image, matrix, shape):
    """
    Warps the perspective in an image 
    according to the warping matrix
    """
    out_shape = shape
    result = cv2.warpPerspective(image, matrix, out_shape, flags = cv2.INTER_LINEAR)
    return result

def getTransformedPoint(p, matrix):
    """
    Transform the points according to the transformation matrix
    """
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    p_after = (int(px), int(py)) 
    return p_after

def getBoundingBoxCenter(bbs):
    """
    Bounding Box bottom center from the bounding box
    """
    bbs = np.array(bbs)
    bbs_center = np.zeros((len(bbs),2), np.int32)
    if len(bbs_center) ==0: return bbs_center
    bbs_center[:,0] = (bbs[:,0] + bbs[:,2]) // 2
    bbs_center[:,1] = (bbs[:,3])
    return bbs_center

def getPaiWiseDistance(points):
    if len(points) == 0: return np.int32([])
    return np.int32(
      np.linalg.norm(
          points[:, None, :] - points[None, :, :], axis=-1
          )
      )