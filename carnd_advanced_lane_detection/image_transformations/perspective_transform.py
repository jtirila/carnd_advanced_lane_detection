import cv2

# TODO : determine good points for perspective transform
ROAD_SRC = ((1, 2))
ROAD_DST = ((1, 2))


def road_perspective_transform(img):
    """Performs the perspective transform specific to mapping a vehicle
    camera image into a bird's eye view.
    
    :param img: An image object
    :return: An image object with the perspective transform applied"""

    return perspective_transform(img, ROAD_SRC, ROAD_DST)


def perspective_transform(img, src, dst):
    """Performs generic perspective transform for any src and dst trapezoids
    
    :param img: source image
    :param src: the source trapezoid
    :param dst: the destination trapezoid:
    :return: the image transformed with the input parameters using openCV's 
             warpPerspective function"""

    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

