import numpy as np
import matplotlib.pyplot as plt
from carnd_advanced_lane_detection.image_transformations.colorspace_conversions import brg_to_rgb
import cv2


def stack_images(first, second):
    # TODO: just copy-pasting for now
    np.dstack(( np.zeros_like(first), first, second))


def one_by_two_plot(first, second, first_cmap=None, second_cmap=None, first_title="First image", second_title="Second image"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title(first_title)
    if first_cmap is not None:
        ax1.imshow(first, cmap=first_cmap)
    else:
        ax1.imshow(first)

    ax2.set_title(second_title)
    if second_cmap is not None:
        ax2.imshow(second, cmap=second_cmap)
    else:
        ax2.imshow(second)
    plt.show()


# TODO: finish this if needed
def two_by_two_plot(first, second, third, fourth):
    pass


def visualize_lanes_with_polynomials(binary_warped, left_fit, right_fit, nonzerox, nonzeroy, left_lane_inds, right_lane_inds, out_img):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def open_visualize_single_image(path):
    image = cv2.imread(path)
    plt.imshow(brg_to_rgb(image))
    plt.show()


def return_superimposed_polyfits(warped, left_line, right_line):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    left_fit = left_line.get_smoothed_coeffs()
    right_fit = right_line.get_smoothed_coeffs()

    try:
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        # Create an image to draw the lines on

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))


    except IndexError:
        # FIXME something went wrong with the polynomial fits, need to account for this somehow
        pass
    return color_warp