import cv2 as cv
import numpy as np
import detector.hand_detector as hd


cam = cv.VideoCapture(0)
fgbg = cv.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cam.read()
    frame = cv.flip(frame,1)
    # bil = cv.bilateralFilter(frame,5,160,160)
    # blur = cv.GaussianBlur(frame, (5, 5), 0)
    #hls = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    handdetect = hd.HandDetector()
    res = handdetect._get_interesting_pixels_mask(frame)
    res = handdetect._get_transformed_pixels_mask(res)
    # print (res)
    cv.imshow('before', frame)
    cv.imshow('after', res)

    if not ret:
        break
    k = cv.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
cam.release()
cv.destroyAllWindows()

'''
    #red
    red = np.uint8([[[0,0,255]]])
    hsv_red = cv.cvtColor(red, cv.COLOR_BGR2HSV)
    cv.imshow('red', hsv_red)


    # show only blue colours
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv.inRange(hls, lower_blue, upper_blue)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow("webcam", res)
    
        fgmask = fgbg.apply(blur)
    ret, thresh = cv.threshold(fgmask, 100, 255, cv.THRESH_BINARY)

    kern = np.ones((5, 5), 'uint8')
    morph = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kern)
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
    bil = cv.bilateralFilter(morph, 5, 160, 160)

    img, hand_con, hierarchy = cv.findContours(bil, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_con = max(hand_con, key=cv.contourArea)
    # cv.drawContours(frame, max_con, -1,(0,255,0), 3)
    print(hand_con[0])
    pts = np.array(hand_con)
    hull = cv.convexHull(pts, returnPoints=False)
    defects = cv.convexityDefects(pts, hull)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(pts[s][0])
        end = tuple(pts[e][0])
        far = tuple(pts[f][0])
        cv.line(frame, start, end,[0,255,0], 2)
        cv.circle(frame, far, 5, [0,0,255],-1)
'''

