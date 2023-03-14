import numpy as np
import cv2
from CameraCalibrator import CameraCalibrator

cap = cv2.VideoCapture(0)
calibrator = CameraCalibrator()
mtx, dist = calibrator.load_coefficients('calibration_chessboard.yml')


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    undistorted = cv2.undistort(gray, mtx, dist, None, None)

    cv2.imshow('gray',gray)
    cv2.imshow('undistorted',undistorted)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
