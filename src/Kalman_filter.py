import numpy as np
import cv2

def createKalmanFilter(kf=None):
    stateSize = 4
    measSize = 2
    contrSize = 0
    if kf is None:
        kf = cv2.KalmanFilter(stateSize, measSize, contrSize)

    # initialize measurement matrix
    measure_matrix = np.zeros((measSize, stateSize), dtype=np.float32)
    measure_matrix[0, 0] = 1.
    measure_matrix[1, 1] = 1.
    kf.measurementMatrix = measure_matrix
    kf.transitionMatrix = np.eye(stateSize, dtype=np.float32)
    # initialize process noise covariance matrix
    kf.processNoiseCov[0, 0] = 1e-2
    kf.processNoiseCov[1, 1] = 1e-2
    kf.processNoiseCov[2, 2] = 7.
    kf.processNoiseCov[3, 3] = 7.
    
    kf.measurementNoiseCov = np.eye(measSize, dtype=np.float32)*1e-1
    return kf