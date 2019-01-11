from sklearn.metrics import pairwise
import cv2 as cv
import numpy as np
from enum import Enum
from math import acos, pi
from config import SKIN_LOWER_BOUND, SKIN_UPPER_BOUND

lower_blue = np.array([78,158,124])
upper_blue = np.array([138,255,255])

lower_green = np.array([50, 100, 50])
upper_green = np.array([80,255,255])

class Color(Enum):
    GREEN = 0
    BLUE = 1

class MarkDetector:
    def __init__(self, color):
        if color == Color.GREEN:
            self.lower_bound = lower_green
            self.upper_bound = upper_green
        elif color == Color.BLUE:
            self.lower_bound = lower_blue
            self.upper_bound = upper_blue
        self.fgbg = cv.createBackgroundSubtractorMOG2()

    def get_mark_position(self, frame):
        center_position = None
        fingers_count = 0

        mask = self._get_interesting_pixels_mask(frame)
        transformed = self._get_transformed_pixels_mask(mask)
        cv.imshow('tra', mask)
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#contours-getting-started
        img, contours, hierarchy = cv.findContours(transformed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            max_area = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(max_area)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_position = (int(x + (w / 2)), int(y+(h / 2)))
            cv.circle(frame, center_position, 5, (0, 0, 255), -1)

        return center_position

    def _get_interesting_pixels_mask(self, frame):
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
        # (blurring)
        blur = cv.GaussianBlur(frame, (5, 5), 0)
        frame_bw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
        hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        # mask = cv.inRange(hsv, lower_green, upper_green)

        # mask = cv.inRange(hsv, SKIN_LOWER_BOUND, SKIN_UPPER_BOUND)
        mask = cv.inRange(hsv, self.lower_bound, self.upper_bound)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)
        res = cv.bitwise_and(frame_bw, mask, mask=mask)
        # http://docs.opencv.org/3.1.0/db/d5c/tutorial_py_bg_subtraction.html
        # (subtract background)
        fgmask = self.fgbg.apply(res)
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
        val, ret = cv.threshold(fgmask, 100, 255, cv.THRESH_OTSU|cv.THRESH_BINARY)
        return ret

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


