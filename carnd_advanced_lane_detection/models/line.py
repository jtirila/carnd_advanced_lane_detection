import numpy as np
import cv2
from collections import namedtuple
from collections import deque


PointCollections = namedtuple("point_collections", ['x', 'y', 'lane_inds'])

# Define a class to keep track of the characteristics of line detection
class Line():
    WEIGHTS = np.array([1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9, 1/10, 1/11])
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
        self.recent_coeffs = np.array(np.empty((10,), dtype=object))
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


    def _detect_line_pixels(self, img):
        """TODO: Using the information from previous rounds stored in the member variables, detect the lane line 
        pixels from the input img
        
        :param img: A rgb image.
        :return: Nothing, just update member variables"""
        pass

    def process_new_image(self):
        """"""
        # Detect lane line pixels
        # Fit polynomial
        # What else?
        pass

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

