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
import os

PERSPECTIVE_CALIBRATION_IMAGE_PATH = os.path.join(ROOT_DIR, 'calibration_images', 'perspective_calibration_image.png')

# These are some simple scripts to obtain images for the written report

def visualize_undistortion():
    calibration_images = read_calibration_images()
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calibration_images)

    for img in calibration_images[:3]:
        dst = undistort_image(img, mtx, dist)
        one_by_two_plot(img, dst, None, None, "Original", "Undistorted")


def visualize_undistorted_perspective_calibration_image():
    open_visualize_single_image(PERSPECTIVE_CALIBRATION_IMAGE_PATH)


def visualize_perspective_transformed_image():
    img = plt.imread(PERSPECTIVE_CALIBRATION_IMAGE_PATH)
    img_transformed = road_perspective_transform(img)
    plt.imshow(img_transformed)
    plt.show()
    mpimg.imsave(os.path.join(ROOT_DIR, 'calibration_images', 'perspective_transformed_image.png'), img_transformed)


def visualize_s_channel_normalization():
    img = cv2.imread(PERSPECTIVE_CALIBRATION_IMAGE_PATH)
    img_rgb = brg_to_rgb(img)
    img_normalized = normalize_brightness(img_rgb)
    img_transformed = road_perspective_transform(img_normalized)
    one_by_two_plot(img_rgb, img_normalized, None, None, "Original", "Normalized")
    s_image = rgb_to_s_channel(img_transformed)
    # scaled_s_image = scale_grayscale_to_255(s_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(s_image)
    # equalized = cv2.equalizeHist(s_image)
    one_by_two_plot(s_image, equalized, 'gray', 'gray', "Raw s channel", "Equalized s channel")


if __name__ == "__main__":

    # visualize_undistortion()
    # visualize_undistorted_perspective_calibration_image()
    # visualize_perspective_transformed_image()
    visualize_s_channel_normalization()

