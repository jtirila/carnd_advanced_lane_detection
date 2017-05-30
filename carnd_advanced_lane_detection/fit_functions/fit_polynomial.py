YM_PER_PIX = 30 / 720
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension


def fit_second_order_polynomial(img):
    # TODO: this is just plain copy-paste
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(ploty * YM_PER_PIX, rightx * XM_PER_PIX, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
