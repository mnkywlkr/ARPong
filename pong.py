import imutils
import cv2

from config import SHOW_DETECTIONS
from detector.marker_detector import MarkDetector
from game_engine.pong_game import PongGame
from utils.kalman import KalmanTracker
from utils.positions_display import PositionsDisplay


def run_game(display_detections=False, input_path=None):
    green_detector = MarkDetector()
    blue_detector = MarkDetector()
    # find color green
    green_detector.get_color()
    blue_detector.get_color()
    # find color blue: blue_detector.find_color()
    if input_path is None:
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(input_path)
    pong_game = PongGame()

    while True:
        (grabbed, frame) = video_capture.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=1000)
        frame = cv2.flip(frame, 1)

        green_position = green_detector.get_mark_position(frame)
        blue_position = blue_detector.get_mark_position(frame)

        if green_position is not None:
            player1_position = green_position

        if blue_position is not None:
            player2_position = blue_position

        pong_game.make_move_two_players(player1_position[1], player2_position[1], frame.shape[0])
        pong_game.draw(frame)

        cv2.imshow("result", frame)

        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            pong_game.reset_game()

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_game(SHOW_DETECTIONS, 0)
