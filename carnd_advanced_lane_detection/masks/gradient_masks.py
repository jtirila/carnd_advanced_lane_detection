import cv2
import numpy as np


def _abs_sobel(sobelx, sobely):
    return np.sqrt(np.square(sobelx) + np.square(sobely))


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = _sobel(gray, 'x', sobel_kernel)
    sobely = _sobel(gray, 'y', sobel_kernel)

    sobel_abs = _abs_sobel(sobelx, sobely)
    scaled_sobel = np.uint8(255 * sobel_abs /np.max(sobel_abs))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return binary_output


def _sobel(image, orient='x', ksize=3):
    assert orient in ('x', 'y')
    if orient == 'x':
        orient_tuple = (1, 0)
    elif orient == 'y':
        orient_tuple = (0, 1)
    return cv2.Sobel(image, cv2.CV_64F, *orient_tuple, ksize=ksize)


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    orient = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(abs_sobelx)
    binary_output[(orient >= thresh[0]) & (orient <= thresh[1])] = 1
    return binary_output


def fixme_first_combined_threshold():
    pass