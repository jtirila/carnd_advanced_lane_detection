import cv2
import numpy as np

# TODO : determine good points for perspective transform, these are just the result of some early experiments
ROAD_SRC = np.float32([[250, 720], [589, 463], [701, 463], [1030, 720]])
ROAD_DST = np.float32([[250, 720], [250, 0], [1030, 0], [1030, 720]])


def road_perspective_transform(img, inverse=False):
    """Performs the perspective transform specific to mapping a vehicle
    camera image into a bird's eye view.
    
    :param img: An image object
    :param inverse: Boolean, whether to compute inverse transform 
    :return: An image object with the perspective transform applied, either forward transform if param inverse is 
             missing or False, or inverse transform if inverse param is set to True"""

    if img is None:
        return None
    transform_points_params = (ROAD_DST, ROAD_SRC) if inverse else (ROAD_SRC, ROAD_DST)
    return perspective_transform(img, *transform_points_params)


def perspective_transform(img, src, dst):
    """Performs generic perspective transform for any src and dst trapezoids
    
    :param img: source image
    :param src: the source trapezoid
    :param dst: the destination trapezoid:
    :return: the image transformed with the input parameters using openCV's 
             warpPerspective function"""

    assert img is not None
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

