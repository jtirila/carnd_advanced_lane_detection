from carnd_advanced_lane_detection.preparations.read_calibration_images import read_calibration_images
from carnd_advanced_lane_detection.calibrate_camera.calibrate_camera import calibrate_camera
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import mass_rgb_to_grayscale
from carnd_advanced_lane_detection.utils.visualize_images import one_by_two_plot

import cv2



# TODO: This is just some early experimentation

calibration_images = read_calibration_images()
gray_calibration_images = mass_rgb_to_grayscale(calibration_images)
ret, mtx, dist, rvecs, tvecs = calibrate_camera(gray_calibration_images)
print(ret, mtx, dist, rvecs, tvecs)

dst2 = cv2.undistort(calibration_images[1], mtx, dist, None, mtx)


for img in calibration_images[:3]:
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    one_by_two_plot(img, dst, None, None, "Original", "Undistorted")



# The real thing, READ FROM THE PROJECT RUBRIC


# 1) Calibrate camera using the 9x6 chessboard images
#    - This will produce the undistortion matrix etc use
# 2) Undistort an example calibration image and display it in writeup
# 3) Undistort an example video frame and display in writeup
# 4) Some methods to be performed on individual images:
# 4.1) The perspective transformation is performed successfully to
#      the image to obtain a bird's eye view
# 4.2) (An own addition: a region of interest mask is applied
# 4.3) A binary mask is applied to the bird's eye picture to
#      identify lane line pixels. This mask is some kind of a combination of
#      color and gradient thresholding.
# 4.4) With the collection of lane line pixels at hand, a 2nd degree polynomial
#      fit is performed and the corresponding curvature calculated.
#      Note that tha curvature is needed in terms of real world measurements,
#      in metres / yards or so
# 4.5. The location of the camera relative to the edges of the lane is calculated,
#      again in metres.
# 4.6. The fitted polynomial must be superimposed on the original image again
#
# NOTES:
#
# * Keep in mind the Line class and some kind of a memory effect to smooth things
#   out in situations where things go wrong for a frame or two
