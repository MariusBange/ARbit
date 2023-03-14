import numpy as np
from filterpy.kalman import KalmanFilter

class GlobeKalmanFilter:
    def __init__(self):
        self.kf = KalmanFilter(12, 12)
        self.kf.x = np.zeros(12)
        self.kf.F = np.eye(12, 12)
        self.kf.H = np.eye(12, 12)
        #self.kf.P *= 100
        self.kf.R = np.eye(12, 12)
        self.kf.Q = np.eye(12, 12)
        self.kf.Q[:3] *= 0.1
        self.kf.Q[3:6] *= 0.01
        self.kf.Q[6:9] *= 0.01
        self.kf.Q[9:12] *= 0.001
        self.initialized = False

    def predict(self, dt):
        for i in range(0, 6):
            self.kf.F[i, i+6] = dt
            self.kf.predict()
        return self.kf.x

    def update(self, rvecs, tvecs, rvelocity, tvelocity):
        z = np.zeros(12)
        for i in range(0, 3):
            z[i] = tvecs[i, 0]
            z[i + 3] = rvecs[i, 0]
            z[i + 6] = rvelocity[i, 0]
            z[i + 9] = tvelocity[i, 0]
        if not self.initialized:
            self.kf.x = z
            self.initialized = True
        self.kf.update(z)

    def getPose(self):
        rvecs = np.zeros((3, 1))
        tvecs = np.zeros((3, 1))
        for i in range(0, 3):
            tvecs[i, 0] = self.kf.x[i]
            rvecs[i, 0] = self.kf.x[i + 3]
        return (rvecs, tvecs)

if __name__ == "__main__":
    k = GlobeKalmanFilter()
    test = k.predict(0.1)
    k.getPose()
    z = np.zeros(6)
    z[0] = 1
    k.kf.update(z)
    while(True):
        test = k.predict(0.1)
        print(test)