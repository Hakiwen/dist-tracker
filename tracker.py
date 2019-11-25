import cv2 as cv
from cv2 import aruco
import numpy as np
import threading
import queue

def frame_getter(cap, cal_mtx, dist, frame_buffer, running):
    # cap is opencv capture object
    # frames_buffer is a synchronized queue
    # running is a boolean

    while running[0]:
        ret, current_frame = cap.read()
        # gray = cv.cvtColor(np.float32(current_frame), cv2.COLOR_RGB2GRAY)
        if ret:
            h, w  = current_frame[1].shape[:2]
            dim = (w, h)
            new_cal_mtx = cal_mtx.copy()
            new_cal_mtx = cv.getOptimalNewCameraMatrix(cal_mtx, dist, (w,h), 1, (w,h))
            # corrected_frame = current_frame
            corrected_frame = []
            # print(current_frame)
            corrected_frame = cv.undistort(current_frame, cal_mtx, dist)
            # x, y, w, h = roi
            # corrected_frame = corrected_frame[y:y+h, x:x+w]
            frame_buffer.put(corrected_frame)

def aruco_tagger(frame_buffer,  aruco_dict, parameters, running):


    while running[0]:
        current_frame = frame_buffer.get()
        if current_frame is not None:
            # cv.imshow('frame', current_frame)
            # cv.waitKey(1)
            gray_frame = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)
            corners, ids, rejected_points = aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)

            # cv.imshow('before_frame', current_frame)
            tagged_frame = current_frame.copy()
            tagged_frame = aruco.drawDetectedMarkers(tagged_frame, corners, ids)
            cv.imshow('frame', tagged_frame)
            cv.waitKey(1)


            return corners, ids

if __name__ == '__main__':
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print('Couldnt open video stream')

    cal_file = cv.FileStorage("cal_matrix.xml", cv.FILE_STORAGE_READ)
    cal_mtx = cal_file.getNode("cal_mtx").mat()
    dist = cal_file.getNode("dist").mat()
    cal_file.release()

    tag_length = 0.03


    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    aruco_parameters = aruco.DetectorParameters_create()

    frame_getter_running = [True]
    frame_buffer = queue.Queue()
    frame_getter_thread = threading.Thread(target=frame_getter, args=(cap, cal_mtx, dist, frame_buffer, frame_getter_running))
    frame_getter_thread.start()

    aruco_tagger_running = [True]
    tagged_buffer = queue.Queue()
    # aruco_tagger_thread = threading.Thread(target=aruco_tagger, args=(frame_buffer, tagged_buffer, aruco_dict, aruco_parameters, aruco_tagger_running))

    displayed = False
    # while not displayed:
    while True:
        try:
            aruco_corners, aruco_ids = aruco_tagger(frame_buffer, aruco_dict, aruco_parameters, aruco_tagger_running)
            # frame = frame_buffer.get_nowait()
            # https://www.docs.opencv.org/trunk/d9/d6a/group__aruco.html#ga84dd2e88f3e8c3255eb78e0f79571bd1
            # rotations are in euler angles, translations in the same unit as tag_length
            aruco_rotations, aruco_translations, object_points = aruco.estimatePoseSingleMarkers(aruco_corners, tag_length, cal_mtx, dist)
            print(aruco_rotations)

            # displayed = True
        except queue.Empty:
            continue



