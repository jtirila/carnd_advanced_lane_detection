import os
import sys

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

from carnd_advanced_lane_detection import ROOT_DIR
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import rgb_to_s_channel, \
    normalize_brightness
from carnd_advanced_lane_detection.image_transformations.perspective_transform import road_perspective_transform
from carnd_advanced_lane_detection.image_transformations.undistort_image import undistort_image
from carnd_advanced_lane_detection.masks.color_masks import saturation_mask
from carnd_advanced_lane_detection.masks.combined_masks import submission_combined
from carnd_advanced_lane_detection.models.line import Line
from carnd_advanced_lane_detection.models.line import XM_PER_PIX
from carnd_advanced_lane_detection.preparations.calibrate_camera import calibrate_camera
from carnd_advanced_lane_detection.preparations.read_calibration_images import read_calibration_images
from carnd_advanced_lane_detection.utils.visualize_images import return_superimposed_polyfits

_TRANSFORMED_VIDEO_OUTPUT_PATH = os.path.join(ROOT_DIR, 'transformed.mp4')

TEST = False
# TEST = True
COUNTER = 0

_PROJECT_VIDEO_PATH = os.path.join(ROOT_DIR, 'project_video.mp4') \
    if not TEST \
    else os.path.join(ROOT_DIR, 'test_videos', 'middle_5_sec.mp4')


def _process_image(image, mtx, dist, left_line, right_line):
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
    :param mtx: the camera distortion matrix
    :param dist: the camera distance metric
    :return: an rgb image containing lane line visualizations"""

    udist = undistort_image(image, mtx, dist)
    enhanced = normalize_brightness(udist)
    perspective_image = road_perspective_transform(enhanced)
    # masked = first_combined(perspective_image)

    s_image = rgb_to_s_channel(perspective_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(s_image)
    # equalized = cv2.equalizeHist(s_image)
    # masked = first_combined(frame)
    # masked = dir_threshold(masked, 5, (0.76, 0.84), need_to_gray=False)
    masked = saturation_mask(equalized, (254, 255))
    # masked = scale_grayscale_to_255(masked)
    # masked = first_combined(frame)
    # scaled_masked = scale_grayscale_to_255(masked)

    # s_image = rgb_to_s_channel(perspective_image)
    # normalized_s_image = cv2.equalizeHist(s_image)
    # masked = saturation_mask(normalized_s_image, (253, 255))
    if left_line.previous_fit_succeeded() and right_line.previous_fit_succeeded():
        left, right, nonzerox, nonzeroy, out_img = Line.detect_line_pixels_based_on_previous_fit(masked, left_line.get_smoothed_coeffs(), right_line.get_smoothed_coeffs())
    else:
        print("Need to start from scratch")
        left, right, nonzerox, nonzeroy, out_img = Line.find_lane_lines(masked)
    if out_img is not None:
        left_line.fit_polynomial(left.y, left.x)
        right_line.fit_polynomial(right.y, right.x)
    else:
        print("lane line pixel identification failed, need to just proceed to next frame.")
        pass

    out_img = return_superimposed_polyfits(masked, left_line, right_line)
    ret_img = road_perspective_transform(out_img, inverse=True)
    result = cv2.addWeighted(udist, 1, ret_img, 0.3, 0)
    avg_curverad = 0.5 * (left_line.calculate_curverad() + left_line.calculate_curverad())
    offset = _fixme_compute_camera_offset(left_line, right_line)
    offset_dir = 'left' if offset > 0 else 'right'
    cv2.putText(result, "Curvature: {:.2f} m".format(avg_curverad), (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(result, "Camera offset from lane center: {:.2f} m {}".format(
        np.absolute(offset),
        offset_dir), (100, 150), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return result


def _fixme_compute_camera_offset(left_line, right_line):
    return XM_PER_PIX * (0.5 * (left_line.compute_line_position_at_bottom() + right_line.compute_line_position_at_bottom()) - 640)


def process_video(video_path=_PROJECT_VIDEO_PATH, mtx=None, dist=None, output_path=_TRANSFORMED_VIDEO_OUTPUT_PATH):
    """Processes the input video, frame by frame, and saves the output video in the specified output path. 
    :param video_path: The path of the original video
    :param mtx: the camera distortion matrix
    :param dist: the camera distance metric
    :param output_path: The path of the output video
    """

    left = Line()
    right = Line()

    clip = VideoFileClip(video_path)
    transformed_clip = clip.fl_image(lambda image: _process_image(image, mtx, dist, left, right))
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


