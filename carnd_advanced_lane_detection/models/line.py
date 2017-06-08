import numpy as np
import cv2
from collections import namedtuple
from collections import deque

YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

PointCollections = namedtuple("point_collections", ['x', 'y', 'lane_inds'])

# Define a class to keep track of the characteristics of line detection
class Line():
    WEIGHTS = np.array([1/2, 1/2, 1/2, 1/3, 1/4, 1/4, 1/5, 1/5, 1/6, 1/6, 1/7, 1/7, 1/8, 1/10, 1/10])
    # TODO: this is a copy-paste from lecture material.
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        # Recent polyfits successful?
        self.fits_successful = deque([])
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        self.recent_coeffs = np.array(np.empty((15,), dtype=object))
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        #

        self.current_left_lane_inds = None



    def convert_warped_polynomial_to_original_perspective(self):
        ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
        y_eval = np.max(ploty)
        coeffs = self.get_smoothed_coeffs()
        xs = np.array([coeffs[0] * y**2 + coeffs[1] ** y + coeffs[2] for y in ploty])


    def calculate_curverad(self):

        ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        coeffs = self.get_smoothed_coeffs()
        xs = np.array([coeffs[0] * y**2 + coeffs[1] ** y + coeffs[2] for y in ploty])
        # xs = np.array([200 + (y ** 2) * self.get_smoothed_coeffs() + np.random.randint(-50, high=51)
        #                   for y in ploty])

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(ploty * YM_PER_PIX, xs * XM_PER_PIX, 2)
        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * YM_PER_PIX + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])
        # print("Curvature is {} m".format(curverad))
        return curverad

    @staticmethod
    def detect_line_pixels_based_on_previous_fit(binary_warped, left_fit, right_fit):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 37
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_vals = PointCollections(x=leftx, y=lefty, lane_inds=left_lane_inds)
        right_vals = PointCollections(x=rightx, y=righty, lane_inds=right_lane_inds)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        return left_vals, right_vals, nonzerox, nonzeroy, out_img

    def _detect_line_pixels(self, img):
        """TODO: Using the information from previous rounds stored in the member variables, detect the lane line 
        pixels from the input img
        
        :param img: A rgb image.
        :return: Nothing, just update member variables"""
        pass

    def compute_line_position_at_bottom(self):
        coeffs = self.get_smoothed_coeffs()
        pos = coeffs[0] * 720**2 + coeffs[1] * 720 + coeffs[2]
        return pos

    def previous_fit_succeeded(self):
        return self.recent_coeffs[0] is not None

    def fit_polynomial(self, y, x, degree=2):
        # TODO: fit polynomial
        # TODO: update member variables
        try:
            new_fit = np.polyfit(y, x, degree)
            self._append_last_coefficients(new_fit)
        except TypeError:
            self._append_last_coefficients(None)

    @staticmethod
    def find_lane_lines(binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_vals = PointCollections(x=leftx, y=lefty, lane_inds=left_lane_inds)
        right_vals = PointCollections(x=rightx, y=righty, lane_inds=right_lane_inds)
        return left_vals, right_vals, nonzerox, nonzeroy, out_img

    def _append_last_coefficients(self, coeffs):
        self.recent_coeffs[-1] = coeffs
        self.recent_coeffs = np.roll(self.recent_coeffs, 1)

    def get_smoothed_coeffs(self):
        idx = self.recent_coeffs != np.array(None)
        real_weights = Line.WEIGHTS[idx]
        scaled_real_weights = real_weights / np.sum(real_weights)
        return np.sum(self.recent_coeffs[idx] * scaled_real_weights, axis=0)

