import cv2
import numpy as np
import pathlib

class CameraCalibrator:
    def calibrate_chessboard(dir_path, image_format, square_size, width, height):
        '''Calibrate a camera using chessboard images.'''
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
        objp = np.zeros((height*width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

        objp = objp * square_size

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = pathlib.Path(dir_path).glob(f'*.{image_format}')
        # Iterate through all images
        for fname in images:
            img = cv2.imread(str(fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                cv2.drawChessboardCorners(gray, (width,height), corners2, ret)
                cv2.imshow('img', gray)
                cv2.waitKey(500)

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return [ret, mtx, dist, rvecs, tvecs]

    def calibrate_chessboard_vid(dir_path, image_format, square_size, width, height):
        '''Calibrate a camera using chessboard images.'''
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
        objp = np.zeros((height*width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

        objp = objp * square_size

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        #images = pathlib.Path(dir_path).glob(f'*.{image_format}')
        # Iterate through all images
        #for fname in images:

        cap = cv2.VideoCapture(0)
        num_pics = 10

        while True:
            _, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cv2.imshow('img', gray)
            if cv2.waitKey(50) != -1:
                print("yeah")
                    # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

                # If found, add object points, image points (after refining them)
                if ret:
                    objpoints.append(objp)

                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)

                    cv2.drawChessboardCorners(gray, (width,height), corners2, ret)
                    cv2.imshow('img', gray)
                    cv2.waitKey(500)

                    num_pics -= 1
                    if num_pics == 0:
                        break
        cap.release()
        cv2.destroyAllWindows()

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        return [ret, mtx, dist, rvecs, tvecs]

    def save_coefficients(mtx, dist, path):
        '''Save the camera matrix and the distortion coefficients to given path/file.'''
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
        cv_file.write('K', mtx)
        cv_file.write('D', dist)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()

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

if __name__ == '__main__':
    IMAGES_DIR = 'C:/Users/Yannick/Pictures/CheckerBoardNew'
    IMAGES_FORMAT = 'jpg'
    #SQUARE_SIZE = 20.7 / 8
    #SQUARE_SIZE = 1.87
    SQUARE_SIZE = 16.7 / 9
    #WIDTH = 3
    #HEIGHT = 5
    WIDTH = 6
    HEIGHT = 8

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = CameraCalibrator.calibrate_chessboard_vid(IMAGES_DIR, IMAGES_FORMAT, SQUARE_SIZE, WIDTH, HEIGHT)
    # Save coefficients into a file
    CameraCalibrator.save_coefficients(mtx, dist, "calibration_chessboard_vid.yml")

