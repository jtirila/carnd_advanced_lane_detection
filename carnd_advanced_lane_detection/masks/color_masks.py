import numpy as np


def saturation_mask(s_image, thresh=(0, 255)):
    binary = np.zeros_like(s_image)
    binary[(s_image > thresh[0]) & (s_image <= thresh[1])] = 1
    return binary