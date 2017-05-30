from carnd_advanced_lane_detection.preparations.read_calibration_images import read_calibration_images
from carnd_advanced_lane_detection.calibrate_camera.calibrate_camera import calibrate_camera
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import mass_rgb_to_grayscale
from carnd_advanced_lane_detection.utils.visualize_images import one_by_two_plot

import cv2



calibration_images = read_calibration_images()
gray_calibration_images = mass_rgb_to_grayscale(calibration_images)
ret, mtx, dist, rvecs, tvecs = calibrate_camera(gray_calibration_images)
print(ret, mtx, dist, rvecs, tvecs)

dst = cv2.undistort(calibration_images[0], mtx, dist, None, mtx)
dst2 = cv2.undistort(calibration_images[1], mtx, dist, None, mtx)

one_by_two_plot(calibration_images[0], dst, None, None, "Original", "Undistorted")
one_by_two_plot(calibration_images[2], dst2, None, None, "Original", "Undistorted")




