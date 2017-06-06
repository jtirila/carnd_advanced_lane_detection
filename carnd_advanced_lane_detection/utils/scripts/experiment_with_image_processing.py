import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

from carnd_advanced_lane_detection import ROOT_DIR
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import brg_to_rgb, rgb_to_s_channel, \
    scale_grayscale_to_255, gray_to_rgb
from carnd_advanced_lane_detection.image_transformations.perspective_transform import perspective_transform, \
    road_perspective_transform
from carnd_advanced_lane_detection.masks.color_masks import saturation_mask
from carnd_advanced_lane_detection.masks.gradient_masks import mag_thresh, dir_threshold
from carnd_advanced_lane_detection.masks.combined_masks import first_combined
from carnd_advanced_lane_detection.utils.visualize_images import one_by_two_plot, visualize_lanes_with_polynomials, \
    return_superimposed_polyfits
from carnd_advanced_lane_detection.fit_functions.fit_polynomial import sliding_window_polyfit

PERSPECTIVE_CALIBRATION_PATH = os.path.join(ROOT_DIR, 'calibration_images', 'perspective_calibration_image.png')
PERSPECTIVE_TEST_PATH = os.path.join(ROOT_DIR, 'calibration_images', 'perspective_test_image.png')

TRANSFORMED_VIDEO_OUTPUT_PATH = os.path.join(ROOT_DIR, 'transformed.mp4')


# TODO: early experiments
SRC = np.float32([[285, 720], [578, 473], [721, 473], [1172, 720]])
DST = np.float32([[285, 720], [285, 0], [1172, 0], [1000, 720]])


def display():
    """Just display an image"""
    img = cv2.imread(PERSPECTIVE_CALIBRATION_PATH)
    plt.imshow(brg_to_rgb(img))
    print(img.shape)
    plt.show()


def saturation_mask_image(image):
    """Convert image to rgb, extract s channel and apply saturation mask
    
    TODO: it seems clear from this usage that channel conversions should be performed inside the saturation_mask 
          function"""
    image = brg_to_rgb(image)
    s_image = rgb_to_s_channel(image)
    masked = saturation_mask(s_image, (150, 255))
    return masked


def display_single_saturation_masked_image(image):
    """FIXME: Partly redundant code, repeated to be able to plot images side by side"""
    image = brg_to_rgb(image)
    s_image = rgb_to_s_channel(image)
    scaled_masked = scale_grayscale_to_255(saturation_mask(s_image, (150, 255)))
    one_by_two_plot(s_image, scaled_masked, 'gray', 'gray')


def display_single_saturation_masked_transformed_image(image):
    """FIXME: Again there is some redundancy"""
    image = brg_to_rgb(image)
    image = road_perspective_transform(image)
    s_image = rgb_to_s_channel(image)
    masked = saturation_mask(s_image, (150, 255))
    left_fit, right_fit, out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = sliding_window_polyfit(masked)
    visualize_lanes_with_polynomials(masked, left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, out_img)
    scaled_masked = scale_grayscale_to_255(masked)
    # one_by_two_plot(s_image, masked, 'gray', 'gray')
    cv2.imshow('masked', scaled_masked)
    cv2.imwrite(os.path.join(ROOT_DIR, 'images', 'binary_masked.png'), scaled_masked)
    cv2.waitKey()


def display_single_saturation_masked_transformed_image_with_polyfit(image):
    """FIXME: Again there is some redundancy"""
    image = brg_to_rgb(image)
    image = road_perspective_transform(image)
    s_image = rgb_to_s_channel(image)
    masked = saturation_mask(s_image, (150, 255))
    left_fit, right_fit, out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = sliding_window_polyfit(masked)
    visualize_lanes_with_polynomials(masked, left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, out_img)


def _convert_color_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)


def transform_and_saturation_mask_image(image):
    """FIXME: Again there is some redundancy"""
    # image = brg_to_rgb(image)
    image = _convert_color_image(image)
    perspective_image = road_perspective_transform(image)
    s_image = rgb_to_s_channel(perspective_image)
    equalized = cv2.equalizeHist(s_image)
    masked = saturation_mask(equalized, (254, 255))
    left_fit, right_fit, out_img, nonzerox, nonzeroy, left_lane_inds, right_lane_inds = sliding_window_polyfit(masked)
    if out_img is not None:
        out_img = return_superimposed_polyfits(masked, left_fit, right_fit)
        ret_img = road_perspective_transform(out_img, inverse=True)
    else:
        ret_img = cv2.cvtColor(masked, cv2.COLOR_GRAY2RGB)
    result = cv2.addWeighted(image, 1, ret_img, 0.3, 0)
    return result


def display_transformed_frames():
    cap = cv2.VideoCapture(os.path.join(ROOT_DIR, 'project_video.mp4'))

    frame_count = 1
    while cap.isOpened():
        print("Frame number {}".format(frame_count))
        frame_count += 1
        ret, frame = cap.read()

        # rgb = brg_to_rgb(frame)


        frame = _convert_color_image(frame)
        frame = road_perspective_transform(frame)
        # s_image = rgb_to_s_channel(frame)
        # equalized = cv2.equalizeHist(s_image)
        masked = mag_thresh(frame, 13, (100, 255))
        masked = dir_threshold(masked, 5, (0.76, 0.84), need_to_gray=False)
        # masked = saturation_mask(equalized, (253, 255))
        # masked = scale_grayscale_to_255(masked)
        # masked = first_combined(frame)
        scaled_masked = scale_grayscale_to_255(masked)


        # inversed = road_perspective_transform(scaled_masked, inverse=True)

        scaled_masked = gray_to_rgb(scaled_masked)
        cv2.imshow('frame', scaled_masked)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def display_partly_transformed_frames():
    cap = cv2.VideoCapture(os.path.join(ROOT_DIR, 'project_video.mp4'))

    frame_count = 1
    while cap.isOpened():
        print("Frame number {}".format(frame_count))
        frame_count += 1
        ret, frame = cap.read()

        # rgb = brg_to_rgb(frame)

        frame = transform_and_saturation_mask_image(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def load_video_capture_frames():
    cap = cv2.VideoCapture(os.path.join(ROOT_DIR, 'project_video.mp4'))

    frame_count = 1
    while cap.isOpened():
        print("Frame number {}".format(frame_count))
        frame_count += 1
        ret, frame = cap.read()
        if frame_count == 358:
            cv2.imwrite(PERSPECTIVE_CALIBRATION_PATH, frame)
        elif frame_count == 50:
            cv2.imwrite(PERSPECTIVE_TEST_PATH, frame)


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # rgb = brg_to_rgb(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def perspective_transform_single_image(img):
    transformed_img = perspective_transform(img, SRC, DST)
    return transformed_img


if __name__ == "__main__":

    # This is just throwaway code, have one block at a time commented out. Also, load_video_capture_frames must
    # be run first once before the others work.

    # load_video_capture_frames()
    # display()

    # img = cv2.imread(PERSPECTIVE_TEST_PATH)
    # transf_img = perspective_transform_single_image(img)
    # one_by_two_plot(brg_to_rgb(img), brg_to_rgb(transf_img))
    # display_single_saturation_masked_transformed_image_with_polyfit(img)

    display_transformed_frames()
    # display_partly_transformed_frames()

    # display_single_saturation_masked_image(img)
    # display_single_saturation_masked_transformed_image(img)

    # transf_img = transform_and_saturation_mask_image(img)
    # plt.imshow(transf_img, cmap='gray')
    # plt.show()


    # clip = VideoFileClip(os.path.join(ROOT_DIR, 'project_video.mp4'))
    # transformed_clip = clip.fl_image(transform_and_saturation_mask_image)
    # transformed_clip.write_videofile(TRANSFORMED_VIDEO_OUTPUT_PATH, audio=False)





