import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import image as mpimg
from moviepy.editor import VideoFileClip

from carnd_advanced_lane_detection import ROOT_DIR
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import brg_to_rgb, rgb_to_s_channel, \
    scale_grayscale_to_255, gray_to_rgb, normalize_brightness, rgb_to_grayscale
from carnd_advanced_lane_detection.image_transformations.perspective_transform import perspective_transform, \
    road_perspective_transform
from carnd_advanced_lane_detection.masks.color_masks import saturation_mask
from carnd_advanced_lane_detection.masks.gradient_masks import mag_thresh, dir_threshold
from carnd_advanced_lane_detection.masks.combined_masks import submission_combined
from carnd_advanced_lane_detection.utils.visualize_images import one_by_two_plot, visualize_lanes_with_polynomials, \
    return_superimposed_polyfits
from carnd_advanced_lane_detection.fit_functions.fit_polynomial import sliding_window_polyfit

PERSPECTIVE_CALIBRATION_PATH = os.path.join(ROOT_DIR, 'images', 'perspective_calibration_image.png')
PERSPECTIVE_TEST_PATH = os.path.join(ROOT_DIR, 'images', 'perspective_test_image.png')

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



def transform_and_saturation_mask_image(image):
    """FIXME: Again there is some redundancy"""
    # image = brg_to_rgb(image)
    image = normalize_brightness(image)
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter()
    out.open(os.path.join(ROOT_DIR, 'mask_comparison.mp4'), fourcc, 20.0, (900, 900), True)

    frame_count = 1
    while cap.isOpened():

        print("Frame number {}".format(frame_count))

        frame_count += 1

        ret, frame = cap.read()
        if not ret:
            break

        # if 576 != frame_count:
        #     continue

        # rgb = brg_to_rgb(frame)



        # frame = normalize_brightness(frame)
        frame = road_perspective_transform(frame)
        if frame is not None:
            frame_normalized = normalize_brightness(frame)
            # if 400 < frame_count < 600:
            #     mpimg.imsave("/tmp/frame-{}.png".format(frame_count - 2), frame)
            s_image = rgb_to_s_channel(frame_normalized)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(s_image)


            s_image_nonnorm = rgb_to_s_channel(frame)


            equalized_nonnorm = clahe.apply(s_image_nonnorm)
            combined_nonnorm = np.hstack((s_image_nonnorm, equalized_nonnorm))
            combined_nonnorm = cv2.resize(combined_nonnorm, (900, 300))
            masked_nonnorm = saturation_mask(combined_nonnorm, (170, 255))

            equalized2 = normalize_brightness(gray_to_rgb(s_image))
            equalized3 = normalize_brightness(gray_to_rgb(s_image_nonnorm))
            combined_luminosity_normalized = np.hstack((equalized2, equalized3))
            combined_luminosity_normalized = cv2.resize(combined_luminosity_normalized, (900, 300))
            combined_luminosity_normalized = rgb_to_grayscale(combined_luminosity_normalized)
            masked_3 = saturation_mask(combined_luminosity_normalized, (254,255))

            # mpimg.imsave("/tmp/problematic_s_channel.png", s_image)
            # mpimg.imsave("/tmp/problematic_s_channel_normalized.png", equalized)
            # equalized = cv2.equalizeHist(s_image)
            # masked = first_combined(frame)
            # masked = dir_threshold(masked, 5, (0.76, 0.84), need_to_gray=False)
            combined = np.hstack((s_image, equalized))
            combined = cv2.resize(combined, (900, 300))
            masked_normalized = saturation_mask(combined, (254, 255))

            ultimate_combined = np.vstack((masked_normalized, masked_nonnorm, masked_3))
            # masked_raw = saturation_mask(s_image, (254, 255))
            # mpimg.imsave("/tmp/problematic_masked_s_channel.png", masked_raw)
            # mpimg.imsave("/tmp/problematic_masked_s_channel_normalized.png", masked_normalized)
            # masked = scale_grayscale_to_255(masked)
            # masked = first_combined(frame)
            #scaled_masked = scale_grayscale_to_255(masked)
            ultimate_combined = scale_grayscale_to_255(ultimate_combined)

            # if frame_count == 620:
            #     cv2.imwrite('/tmp/sample_1.png', ultimate_combined)
            # elif frame_count == 866:
            #     cv2.imwrite('/tmp/sample_2.png', ultimate_combined)
            # elif frame_count == 1001:
            #     cv2.imwrite('/tmp/sample_3.png', ultimate_combined)
            # elif frame_count == 1061:
            #     cv2.imwrite('/tmp/sample_4.png', ultimate_combined)


            # inversed = road_perspective_transform(scaled_masked, inverse=True)

            # scaled_masked = gray_to_rgb(scaled_masked)
            cv2.imshow('frame', ultimate_combined)
            out.write(gray_to_rgb(ultimate_combined))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
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





