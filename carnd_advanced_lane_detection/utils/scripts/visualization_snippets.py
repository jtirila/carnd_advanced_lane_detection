from carnd_advanced_lane_detection.preparations.calibrate_camera import calibrate_camera
from carnd_advanced_lane_detection.preparations.read_calibration_images import read_calibration_images
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import mass_rgb_to_grayscale
from carnd_advanced_lane_detection.image_transformations.undistort_image import undistort_image
from carnd_advanced_lane_detection.utils.visualize_images import one_by_two_plot


# These are some simple scripts to obtain images for the written report

def visualize_undistortion():
    calibration_images = read_calibration_images()
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calibration_images)

    for img in calibration_images[:3]:
        dst = undistort_image(img, mtx, dist)
        one_by_two_plot(img, dst, None, None, "Original", "Undistorted")

