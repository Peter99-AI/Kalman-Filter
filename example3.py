# ball tracker using kalman filter

import  cv2
import numpy as np

stateSize = 6
measSize = 4
contrSize = 0

type = cv2.CV_32F;
kf = cv2.KalmanFilter(stateSize, measSize, contrSize, type)

state = cv2.Mat(stateSize, 1, type)
meas = cv2.Mat(measSize, 1, type)

print(state)
print(meas)
