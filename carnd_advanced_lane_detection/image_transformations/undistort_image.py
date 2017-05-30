import cv2


def undistort_image(image, mtx, dist):
    """FIXME: given the camera calibration parameters, undistort an image"""
    return cv2.undistort(image, mtx, dist, None, mtx)
