import cv2 as cv
from cv2 import aruco
import numpy as np
import threading
import queue
import time
import sys
from mpi4py import MPI

##############################################################################################################

dt = 0.0015
A = np.matrix('1 0 0; 0 1 0; 0 0 1') # stationary target
B = np.matrix('0; 0; 0')
Q = np.matrix('1 0; 0 1')
xs = np.matrix('0; 0; 0')
H = np.matrix('1 0 0; 0 1 0; 0 0 1')
th = 0
n = 2
E = np.array([[0,1]])

class Sensor_Init(object):
    def __init__(self, ind, H, dt, A, B, Q, n, xs, th, E):
        super(Sensor_Init, self).__init__()
        self.xs = xs
        self.P = np.matrix('1 0; 0 1')
        self.z = np.matrix('0; 0')
        self.th = th
        self.u = np.matrix('0; 0')
        self.dt = dt
        self.A = A
        self.B = B
        self.Q = Q
        self.H = H
        self.R = 20*np.sqrt(ind+1)
        self.U = 1/self.R * H.transpose() * H
        self.x_bar = np.matrix(str(20*(np.random.random_sample() - 0.5)) + ';' + str(20*(np.random.random_sample() - 0.5))) + self.P*np.random.randn(2, 1)
        self.message_out = {'u':1/self.R * H.transpose() * self.z, 'U':self.U, 'x_bar':self.x_bar}
        self.message_in = [{'u':self.message_out['u'], 'U':self.message_out['U'], 'x_bar':self.message_out['x_bar']} for i in range(len(E))]
        print(self.x_bar)

def filter_update(s):
    m = len(s.message_in) # number of neighbots
    s.u = 1/s.R * s.H.transpose() * s.z # filtered measurement
    y = s.u
    S = s.U
    w_sum = np.matrix('0; 0')
    for i in range(m):
        y = y + s.message_in[i]['u']
        S = S + s.message_in[i]['U']
        w_sum = w_sum + (s.message_in[i]['x_bar'] - s.x_bar)

    M = np.linalg.inv(np.linalg.inv(s.P) + S)
    x_hat = s.x_bar + M*(y - S*s.x_bar) + s.dt*M*w_sum # prediction
#   print(str(rank) + ',' + str(x_hat)) 
    
    s.P = s.A*M*s.A + s.B*s.Q*s.B.transpose() # covariance update
    s.x_bar = s.A*x_hat # corection
    
#   print(str(rank) + ',' + str(s.x_bar))   
#   time.sleep(5)
    s.message_out['u'] = s.u
    s.message_out['U'] = s.U
    s.message_out['x_bar'] = s.x_bar

    return s


##############################################################################################################

def get_all_states(comm, state):
    size = comm.Get_size()
    all_states = np.zeros((size, len(state)), dtype = np.float)
    comm.Allgather([state, MPI.FLOAT], [all_states, MPI.FLOAT])
    return all_states

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
            # cv.imshow('frame', tagged_frame)
            cv.waitKey(1)

            return corners, ids

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # state = np.zeros(6, dtype = np.float) # [x, y, z, row, pitch, yaw]
    state = np.zeros(3, dtype = np.float) # [x, y, yaw]
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
            start_time = time.time()
            aruco_corners, aruco_ids = aruco_tagger(frame_buffer, aruco_dict, aruco_parameters, aruco_tagger_running)
            find_tags_time = time.time()
            # frame = frame_buffer.get_nowait()
            # https://www.docs.opencv.org/trunk/d9/d6a/group__aruco.html#ga84dd2e88f3e8c3255eb78e0f79571bd1
            # rotations are in euler angles (rad), translations in the same unit as tag_length
            aruco_rotations, aruco_translations, object_points = aruco.estimatePoseSingleMarkers(aruco_corners, tag_length, cal_mtx, dist)
            pose_estimate_time = time.time()
            # print("rot", aruco_rotations)
            # print("trans", aruco_translations)
            for i in range(0, len(state)):
                # if i >= 0 and i < 3 and aruco_translations is not None:
                #     state[i] = aruco_translations[0][0][i]
                # elif i >= 3 and aruco_rotations is not None:
                #     state[i] = aruco_rotations[0][0][i-3]
                if i >= 0 and i < 2 and aruco_translations is not None:
                    state[i] = aruco_translations[0][0][i]
                elif i >= 2 and aruco_rotations is not None:
                    state[i] = aruco_rotations[0][0][5]
                else:
                    state[i] = float('nan')

            sensor.z = state[i]
            sensor = Sensor_Init(rank, H, dt, A, B, Q, n, xs, th, E)
            sensor = filter_update(sensor)

            # call this when 'state' is updated
            all_states = get_all_states(comm, state)
            comm_time = time.time()
            # print("my rank:", rank, ", my state:", state, "\nall states:", all_states)
            if rank == 0:
                print("all states:", all_states)
                # print("find tags time:", find_tags_time - start_time)
                # print("pose estimate time", pose_estimate_time - find_tags_time)
                # print("comm time", comm_time - pose_estimate_time)
                # print("total time", comm_time - start_time)
            sys.stdout.flush()
            # displayed = True
        except queue.Empty:
            continue



