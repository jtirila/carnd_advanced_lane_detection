import cv2
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from carnd_advanced_lane_detection.preparations.calibrate_camera import calibrate_camera
from carnd_advanced_lane_detection.preparations.read_calibration_images import read_calibration_images
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import mass_rgb_to_grayscale, rgb_to_s_channel, scale_grayscale_to_255, normalize_brightness, brg_to_rgb
from carnd_advanced_lane_detection.image_transformations.undistort_image import undistort_image
from carnd_advanced_lane_detection.utils.visualize_images import one_by_two_plot
from carnd_advanced_lane_detection import ROOT_DIR
from carnd_advanced_lane_detection.utils.visualize_images import open_visualize_single_image
from carnd_advanced_lane_detection.image_transformations.perspective_transform import road_perspective_transform
from carnd_advanced_lane_detection.masks.color_masks import saturation_mask
import os

PERSPECTIVE_CALIBRATION_IMAGE_PATH = os.path.join(ROOT_DIR, 'images', 'perspective_calibration_image.png')

# These are some simple scripts to obtain images for the written report

def visualize_undistortion():
    calibration_images = read_calibration_images()
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calibration_images)

    for img in calibration_images[:3]:
        dst = undistort_image(img, mtx, dist)
        one_by_two_plot(img, dst, None, None, "Original", "Undistorted")


def visualize_undistorted_perspective_calibration_image():
    orig = mpimg.imread(PERSPECTIVE_CALIBRATION_IMAGE_PATH)
    udist = undistort_image(orig)
    one_by_two_plot(orig, udist, None, None, "Original image", "Undistorted image")


def visualize_perspective_transformed_image():
    img = plt.imread(PERSPECTIVE_CALIBRATION_IMAGE_PATH)
    img_transformed = road_perspective_transform(img)
    plt.imshow(img_transformed)
    plt.show()
    mpimg.imsave(os.path.join(ROOT_DIR, 'images', 'perspective_transformed_image.png'), img_transformed)


def visualize_s_channel_normalization():
    img = cv2.imread(PERSPECTIVE_CALIBRATION_IMAGE_PATH)
    img_rgb = brg_to_rgb(img)
    img_normalized = normalize_brightness(img_rgb)
    img_transformed = road_perspective_transform(img_normalized)
    img_transformed_2 = road_perspective_transform(img)
    one_by_two_plot(img_rgb, img_normalized, None, None, "Original", "Normalized")
    s_image = rgb_to_s_channel(img_transformed)
    s_image_2 = rgb_to_s_channel(img_transformed_2)
    # scaled_s_image = scale_grayscale_to_255(s_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(s_image)
    one_by_two_plot(s_image, equalized, 'gray', 'gray', "Raw s channel", "Equalized s channel")
    masked_normalized = saturation_mask(equalized, (254, 255))
    masked_raw = saturation_mask(s_image, (254, 255))
    masked_original_raw = saturation_mask(s_image_2, (254, 254))
    masked_original_raw_2 = saturation_mask(s_image_2, (100, 254))
    masked_original_raw_3 = saturation_mask(s_image_2, (170, 254))
    # equalized = cv2.equalizeHist(s_image)
    one_by_two_plot(masked_raw, masked_normalized, 'gray', 'gray', "S channel masked", "Normalized s channel masked")
    one_by_two_plot(masked_raw, masked_original_raw, 'gray', 'gray', "S channel masked", "Original image, not brightness normalized, s channel masked, with same threshold")
    one_by_two_plot(masked_original_raw_2, masked_original_raw_3, 'gray', 'gray', "Not brightness normalized or equalized, s channel lower thresh 60", "Not brightness normalized or equalized, s channel lower thresh 170")


if __name__ == "__main__":

    # visualize_undistortion()
    visualize_undistorted_perspective_calibration_image()
    # visualize_perspective_transformed_image()
    # visualize_s_channel_normalization()

