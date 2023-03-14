import numpy as np
import cv2

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

cap = cv2.VideoCapture(0)
mtx, dist = load_coefficients('calibration_chessboard.yml')


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.undistort(gray, mtx, dist, None, None)
    hue = hsv[:, :, 0]
    hue = cv2.GaussianBlur(hue, (7,7), 0)
    saturation = hsv[:, :, 1]
    #saturation = cv2.GaussianBlur(saturation, (5,5), 0)
    v = hsv[:, :, 2]
    blue = frame[:, :, 0]
    green = frame[:, :, 1]
    red = frame[:, :, 2]

    #gray = cv2.medianBlur(gray, 5)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    #blue = cv2.medianBlur(blue, 5)
    #blue = cv2.medianBlur(blue, 5)
    #red = cv2.medianBlur(red, 5)
    #red = cv2.GaussianBlur(red, (5,5), 0)
    #_, saturation = cv2.threshold(saturation, 50, 255, 0)
    
    
    rows = gray.shape[0]
    #rows = blue.shape[0]
    circles = cv2.HoughCircles(hue, cv2.HOUGH_GRADIENT, 1, rows / 4,
                               param1=200, param2=35,
                               minRadius=80, maxRadius=200)
    
    canny = cv2.Canny(hue, 60, 30)


    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        #i = circles[0][0]
        #center = (i[0], i[1])
        ## circle center
        #cv2.circle(frame, center, 1, (0, 100, 100), 3)
        ## circle outline
        #radius = i[2]
        #cv2.circle(frame, center, radius, (255, 0, 255), 3)
        
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 255), 3)
        center = (circles[0,0,0], circles[0,0,1])
        radius = circles[0,0,2]
        print(gray.shape)
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.circle(mask, center, radius, 255, -1)
        frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow("detected circles", frame)
    #cv2.waitKey(0)

    # Display the resulting frame
    cv2.imshow('hue',hue)
    cv2.imshow('saturation',saturation)
    cv2.imshow('v',v)
    cv2.imshow('red',red)
    cv2.imshow('green',green)
    cv2.imshow('blue',blue)
    #cv2.imshow('gray',gray)
    cv2.imshow('canny',canny)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
