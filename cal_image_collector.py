import cv2 as cv
from cv2 import aruco
import numpy as np
import threading
import queue

def frame_getter(cap, frame_buffer, running):
    # cap is opencv capture object
    # frames_buffer is a synchronized queue
    # running is a boolean

    while running[0]:
        current_frame = cap.read()
        frame_buffer.put(current_frame)


if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print('Couldnt open video stream')

    frame_getter_running = [True]
    frame_buffer = queue.Queue()
    frame_getter_thread = threading.Thread(target=frame_getter, args=(cap, frame_buffer, frame_getter_running))
    frame_getter_thread.start()


    i = 0
    while True:
        try:
            ret, frame = frame_buffer.get_nowait()
        except queue.Empty:
            continue

        cv.imshow('frame', frame)
        cv.waitKey(1)
        cv.imwrite('cal_images/' + str(i) + '.jpg', frame)
        i += 1



