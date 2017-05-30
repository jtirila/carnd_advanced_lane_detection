import cv2
import numpy as np
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import rgb_to_grayscale


def calibrate_camera(calibration_images, dims=(9, 6)):
    """FIXME: returns 
    :param calibration_images: An array-like collection of grayscale images
    :param dims: the number of corners, y-by-x
    
    :return: FIXME ret, mtx, dist, rvecs, tvecs"""

    objpoints, imgpoints = _find_imgpoints(calibration_images, dims)
    return cv2.calibrateCamera(objpoints, imgpoints, calibration_images[0].shape[::-1], None, None)


def _find_chessboard_corners(gray, dims):
    return cv2.findChessboardCorners(gray, dims, None)


def _find_imgpoints(calibration_images, dims):

    objpoints = []
    imgpoints = []

    objp = np.zeros((dims[0] * dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1, 2)

    for img in calibration_images:
        ret, corners = _find_chessboard_corners(img, dims)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    return objpoints, imgpoints

