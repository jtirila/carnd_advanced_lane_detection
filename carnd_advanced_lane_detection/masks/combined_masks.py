import numpy as np
import cv2
from carnd_advanced_lane_detection.masks.color_masks import saturation_mask
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import rgb_to_s_channel
from carnd_advanced_lane_detection.masks.gradient_masks import dir_threshold, mag_thresh


def submission_combined(image):
    s_image = rgb_to_s_channel(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(s_image)
    saturation_masked = saturation_mask(equalized, (220, 255))
    grad_mag_thresholded = mag_thresh(image, 5, (30, 255))
    combined = np.zeros_like(saturation_masked)
    combined[(saturation_masked == 1) & (grad_mag_thresholded == 1)] = 1
    return combined
