import numpy as np
from carnd_advanced_lane_detection.masks.color_masks import saturation_mask
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import rgb_to_s_channel
from carnd_advanced_lane_detection.masks.gradient_masks import dir_threshold, mag_thresh


def first_combined(image):
    s_image = rgb_to_s_channel(image)
    saturation_masked = saturation_mask(s_image, (150, 255))
    # saturation_masked = np.zeros_like(image)
    # dir_gradient_thresholded = dir_threshold(image, 15, (0.7, 1.3))
    grad_mag_thresholded = mag_thresh(image, 5, (30, 255))
    combined = np.zeros_like(saturation_masked)
    combined[(saturation_masked == 1) & (grad_mag_thresholded== 1)] = 1
    return combined
