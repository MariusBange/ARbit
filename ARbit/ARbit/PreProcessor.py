import numpy as np
import cv2

class PreProcessor:
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist
        self.center = (0, 0)
        self.radius = 800

    def process(self, frame):
        # Transform Frame into HSV-Color and extract hue image
        hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,0]
        #hue = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        red = frame[:,:,2]
        # Undistort and blur the hue image as preparation for circle detection
        hueUndistorted = cv2.undistort(hue, self.mtx, self.dist, None, None)
        #hueUndistorted = cv2.GaussianBlur(hueUndistorted, (5,5), 0)
        #hue = cv2.medianBlur(hue, 5)
        # Undistort and blur the red image as preparation for circle detection
        redUndistorted = cv2.undistort(red, self.mtx, self.dist, None, None)
        redUndistorted = cv2.GaussianBlur(redUndistorted, (5,5), 0)

        rows = hueUndistorted.shape[0]
        circles = cv2.HoughCircles(redUndistorted, cv2.HOUGH_GRADIENT, 1, rows / 4,
                                   param1=170, param2=38,
                                   minRadius=80, maxRadius=200)
        #canny = cv2.Canny(hueUndistorted, 200, 100)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
            self.center = (circles[0,0,0], circles[0,0,1])
            self.radius = circles[0,0,2]
        else:
            self.radius = self.radius * 1.03
        if self.radius < 800:
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.circle(mask, self.center, int(self.radius * 0.95), 255, -1)
            #frame = cv2.bitwise_and(frame, frame, mask=mask)
            hueMasked = cv2.bitwise_and(hue, hue, mask=mask)
            hueMasked = cv2.bitwise_and(hue, hue, mask=mask)
            return hueMasked
        else:
            self.radius = 800
            return hue