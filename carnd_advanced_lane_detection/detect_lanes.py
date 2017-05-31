import os
import sys
import cv2

from carnd_advanced_lane_detection import ROOT_DIR
from carnd_advanced_lane_detection.preparations.calibrate_camera import calibrate_camera
from carnd_advanced_lane_detection.preparations.read_calibration_images import read_calibration_images
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import mass_rgb_to_grayscale, \
    rgb_to_s_channel, gray_to_rgb
from carnd_advanced_lane_detection.image_transformations.perspective_transform import road_perspective_transform
from carnd_advanced_lane_detection.image_transformations.undistort_image import undistort_image
from carnd_advanced_lane_detection.masks.color_masks import saturation_mask
from carnd_advanced_lane_detection.fit_functions.fit_polynomial import sliding_window_polyfit
from moviepy.editor import VideoFileClip
from carnd_advanced_lane_detection.utils.visualize_images import one_by_two_plot, return_superimposed_polyfits

_TRANSFORMED_VIDEO_OUTPUT_PATH = os.path.join(ROOT_DIR, 'transformed.mp4')
_PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'project_video.mp4')


def _draw_lane_visualization(image):
    """Augments the image with superimposed lane detection results. This function performs most of the heavy lifting 
    to detect lane lines. The steps performed to achieve this result are: 
       - Perspective transform the image to obtain a bird's eye view of the lane ahead 
       - Apply a mask combining gradient and saturation criteria to the image, with the aim of extracting pixels that 
         belong to the lane line
       - Apply a histogram based search algorithm to approximately locate the bases of left and right lane lines in the
         image 
       - Using the approximate locations, segment the lane pixels to left and right lane pixels, then fit a quadratic
         function to these pixels to estimate the smoothed center of the lane line
       - Inverse perspective transform the fitted polynomials back to the vehicle camera perspective, and draw them 
         superimposed on the original image with some added stylistic effects
         
    :param image: A rbg image. Use distortion corrected images. 
    :return: A rgb image, otherwise the same as input but now lane line illustrations have been superimposed on top
    """
    perspective_image = road_perspective_transform(image)
    s_image = rgb_to_s_channel(perspective_image)
    masked = saturation_mask(s_image, (150, 255))
    left_fit, right_fit, out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = sliding_window_polyfit(masked)
    if out_img is not None:
        out_img = return_superimposed_polyfits(masked, left_fit, right_fit)
        ret_img = road_perspective_transform(out_img, inverse=True)
    else:
        ret_img = gray_to_rgb(masked)
    result = cv2.addWeighted(image, 1, ret_img, 0.3, 0)
    return result


def _process_image(image, mtx, dist):
    """First undistorts and image, then runs it through the lane detection algorithm and returns the resulting 
    stacked image.
    
    :param image: the original undistorted rgb image
    :param mtx: the camera distortion matrix
    :param dist: the camera distance metric
    :return: an rgb image containing lane line visualizations"""

    return _draw_lane_visualization(undistort_image(image, mtx, dist))


def process_video(video_path=_PROJECT_VIDEO_PATH, mtx=None, dist=None, output_path=_TRANSFORMED_VIDEO_OUTPUT_PATH):
    """Processes the input video, frame by frame, and saves the output video in the specified output path. 
    :param video_path: The path of the original video
    :param mtx: the camera distortion matrix
    :param dist: the camera distance metric
    :param output_path: The path of the output video
    """
    clip = VideoFileClip(video_path)
    transformed_clip = clip.fl_image(lambda image: _process_image(image, mtx, dist))
    transformed_clip.write_videofile(output_path, audio=False)


def detect_lanes():
    """Orchestrates the video processing procedure. No params, no return value, but if everything goes as expected, 
    a resulting video file has been saved on disk"""

    # Collect the camera calibration parameters
    ret, mtx, dist, _, _ = calibrate_camera(read_calibration_images())

    # Process video frame by frame, undistorting the images using the calibration params if the calibration was
    # successful.
    if ret:
        process_video(mtx=mtx, dist=dist)
    else:
        # Cannot do anything if calibration did not succeed.
        print("Unable to determine camera calibration parameters, quitting")
        sys.exit(1)

if __name__ == "__main__":
    detect_lanes()


# The real thing, READ FROM THE PROJECT RUBRIC


# 1) Calibrate camera using the 9x6 chessboard images
#    - This will produce the undistortion matrix etc use
# 2) Undistort an example calibration image and display it in writeup
# 3) Undistort an example video frame and display in writeup
# 4) Some methods to be performed on individual images:
# 4.1) The perspective transformation is performed successfully to
#      the image to obtain a bird's eye view
# 4.2) A binary mask is applied to the bird's eye picture to
#      identify lane line pixels. This mask is some kind of a combination of
#      color and gradient thresholding.
# 4.3) With the collection of lane line pixels at hand, a 2nd degree polynomial
#      fit is performed and the corresponding curvature calculated.
#      Note that tha curvature is needed in terms of real world measurements,
#      in metres / yards or so
# 4.4. The location of the camera relative to the edges of the lane is calculated,
#      again in metres.
# 4.5. The fitted polynomial must be superimposed on the original image again
#
# NOTES:
#
# * Keep in mind the Line class and some kind of a memory effect to smooth things
#   out in situations where things go wrong for a frame or two
