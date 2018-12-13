import cv2 as cv


class PongGame:
    def draw_left_slider(self, frame):
        cv.rectangle(frame, (0,0), (frame.shape[0], frame.shape[1]),(255,0,0), 3)

    def draw_right_slider(self,frame):
        cv.rectangle(frame,(),(frame.shape[0], frame.shape[1]), (0, 0, 255), 3)
# press SPACE to take picture, ESC to quit
def take_a_photo():
    # auf Patricia's Laptop 1 ist logitech-webcam
    cam = cv.VideoCapture(0)

    img_counter = 0
    while True:
        ret, frame = cam.read()

        cv.rectangle(frame,(384,0),(510,128),(0,255,0),3)

        cv.imshow("before", frame)
        # cv.imshow("edge", edge)

        if not ret:
            break
        k = cv.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    cam.release()
    cv.destroyAllWindows()

take_a_photo()
