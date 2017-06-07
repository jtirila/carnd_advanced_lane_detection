import os
import glob
import cv2
import numpy as np
from carnd_advanced_lane_detection import ROOT_DIR

_CALIBRATION_IMAGE_PATHS = glob.glob(os.path.join(ROOT_DIR, 'images', 'calibration*.jpg'))


def read_calibration_images():
    return np.array(list(map(cv2.imread, _CALIBRATION_IMAGE_PATHS)))

