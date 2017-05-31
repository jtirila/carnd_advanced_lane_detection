import cv2
import numpy as np
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import mass_rgb_to_grayscale


def calibrate_camera(calibration_images, dims=(9, 6)):
    """Processes the calibration images and comes up with the distortion parameters.  
    :param calibration_images: An array-like collection of grayscale images
    :param dims: the number of corners, y-by-x
    
    :return: ret, mtx, dist, rvecs, tvecs as per cv2.calibrateCamera"""

    gray_calibration_images = mass_rgb_to_grayscale(calibration_images)
    objpoints, imgpoints = _find_imgpoints(gray_calibration_images, dims)
    return cv2.calibrateCamera(objpoints, imgpoints, gray_calibration_images[0].shape[::-1], None, None)


def _find_chessboard_corners(gray, dims):
    """Just a convenience wrapper for cv2's findChessboardCorners"""
    return cv2.findChessboardCorners(gray, dims, None)


def _find_imgpoints(calibration_images, dims):
    """Compilile the object and image point arrays from calibration images
    
    :param calibration_images: the set of calibration images, in grayscale
    :param dims: the tuple containing the column and row numbers of the chessboard intersection grid.
    
    :return: object point array and image point array"""

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

