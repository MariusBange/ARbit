
import numpy as np
import cv2
import time
import multiprocessing

from CameraCalibrator import CameraCalibrator
from PreProcessor import PreProcessor
from GlobeDetector import GlobeDetector
from GlobeKalmanFilter import GlobeKalmanFilter

class Processor:

    def __init__(self, standalone=False):
        self.mtx, self.dist = CameraCalibrator.load_coefficients('calibration_chessboard_vid.yml')

        self.axis = np.float32([[0, 0, 0], [10,0,0], [0,10,0], [0,0,10]]).reshape(-1,3)
        self.axisColors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

        self.preProcessor = PreProcessor(self.mtx, self.dist)
        #self.globeDetector = GlobeDetector(self.mtx, self.dist, 'textured_miller_projection_2_test.png')
        self.globeDetector = GlobeDetector(self.mtx, self.dist, 'globe_fully_colored_half_res.png')
        #self.globeDetector = GlobeDetector(self.mtx, self.dist, 'textured_globe_test.png')
        self.globeKalmanFilter = GlobeKalmanFilter()

        self.rvecs = None
        self.tvecs = None
        self.features = None
        self.features3D = None
        self.lastFrame = None

        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.prev_update_time = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.frameTimes = [100]

        self.counter = 0

        self.standalone = standalone

    def process(self, frame):
        rvecs = None
        tvecs = None
        features = None
        features3D = None
        hueMasked = self.preProcessor.process(frame)
        detected = False
        if self.features is not None:
            if self.counter != 0:
                response = self.globeDetector.trackGlobe(frame, self.lastFrame, self.features, self.features3D, self.rvecs, self.tvecs)
                if response is not None:
                    print("lk")
                    rvecs, tvecs, features, features3D = response
                    detected = True
                else:
                    self.counter = 0
        if self.counter == 0:
            response = self.globeDetector.detectGlobe(hueMasked, self.rvecs, self.tvecs)
            if response is not None:
                print("orb")
                rvecs, tvecs, features, features3D = response
                detected = True
        self.new_frame_time = time.time()
        dt = self.new_frame_time - self.prev_frame_time
        self.globeKalmanFilter.predict(dt)
        if rvecs is not None and detected:
            #print(np.reshape(tvecs, -1))
            dtu = self.new_frame_time - self.prev_update_time
            self.prev_update_time = self.new_frame_time
            if self.rvecs is not None:
                rvelocity = (rvecs - self.rvecs) / dtu
                tvelocity = (tvecs - self.tvecs) / dtu
            else:
                rvelocity = np.zeros(rvecs.shape)
                tvelocity = np.zeros(tvecs.shape)
            self.globeKalmanFilter.update(rvecs, tvecs, rvelocity, tvelocity)
            #rvecs, tvecs = self.globeKalmanFilter.getPose()
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.features = features
            self.features3D = features3D
        self.frameTimes.append(dt)
        if len(self.frameTimes) > 10:
            del(self.frameTimes[0])
        self.frameTime = sum(self.frameTimes) / len(self.frameTimes)
        if self.frameTime != 0:
            fps = str(int(1/(self.frameTime)))
        else:
            fps = "0"
        self.prev_frame_time = self.new_frame_time
        if self.standalone:
            cv2.putText(frame, fps, (7, 70), self.font, 2, (100, 255, 0), 3, cv2.LINE_AA)
            if self.rvecs is not None:
                imgpts, jac = cv2.projectPoints(self.axis, self.rvecs, self.tvecs, self.mtx, self.dist)
                zero = (int(imgpts[0,0,0]), int(imgpts[0,0,1]))
                imgpts = imgpts[1:]
                try:
                    for i, p in enumerate(imgpts):
                        img = cv2.line(frame, zero, (int(p[0,0]), int(p[0,1])),self.axisColors[i],3)
                except Exception as e:
                    pass

            cv2.imshow('hue',hueMasked)
            cv2.imshow('frame',frame)
        self.lastFrame = frame
        self.counter = (self.counter + 1) % 5

        return self.rvecs, self.tvecs

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    p = Processor(standalone=True)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            print(frame.shape)
            p.process(frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()