import numpy as np
import cv2


def mass_rgb_to_grayscale(images):
    return np.array(list(map(rgb_to_grayscale, images)))


def mass_brg_to_grayscale(images):
    return np.array(list(map(brg_to_grayscale, images)))


def rgb_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def brg_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def brg_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def gray_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def scale_grayscale_to_255(image):
    return np.uint8(255 * image / np.max(image))






def rgb_to_s_channel(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 2]


def normalize_brightness(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)