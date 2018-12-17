import cv2 as cv
import numpy as np
from math import acos, pi
from config import SKIN_LOWER_BOUND, SKIN_UPPER_BOUND


class HandDetector:
    def __init__(self):
        self.fgbg = cv.createBackgroundSubtractorMOG2()

    def get_hand_position(self, frame):
        position = None
        fingers_count = 0

        mask = self._get_interesting_pixels_mask(frame)
        transformed = self._get_transformed_pixels_mask(mask)

        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#contours-getting-started
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
        # max(contours, key=cv2.contourArea)

        return position, fingers_count

    def _get_interesting_pixels_mask(self, frame):
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
        # (blurring)
        blur = cv.GaussianBlur(frame, (5, 5), 0)
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, SKIN_LOWER_BOUND, SKIN_UPPER_BOUND)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)
        res = cv.bitwise_and(frame, frame, mask=mask)
        # http://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html
        # (subtract background)
        fgmask = self.fgbg.apply(res)
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
        # ret, thresh = cv.threshold(fgmask, 100, 255, cv.THRESH_BINARY)
        return fgmask

    @staticmethod
    def _get_transformed_pixels_mask(mask):
        # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        kern = np.ones((5,5), 'uint8')
        morph = cv.morphologyEx(mask,cv.MORPH_CLOSE, kern)
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
        bil = cv.bilateralFilter(morph, 5, 160, 160)

        return bil

    @staticmethod
    def _get_fingers_count(hand_contour):
        # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        # https://en.wikipedia.org/wiki/Law_of_cosines

        return 0


