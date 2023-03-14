import re
import os.path

import numpy as np
import cv2

class GlobeDetector:
    def __init__(self, mtx, dist, fileName, numDescriptors = 2000):
        self.mtx = mtx
        self.dist = dist
        self.globeDescriptors, self.keypoints3D = self.makeGlobeDescriptor(fileName, numDescriptors)
        self.runtimeOrb = cv2.ORB_create(int(numDescriptors/2))
        self.flannMatcher = self.initFlannMatcher()
        self.lkParams = dict( winSize  = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def initFlannMatcher(self):
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6, # 12
                           key_size = 12,     # 20
                           multi_probe_level = 1) #2
        search_params = dict(checks=50)   # or pass empty dictionary
        return cv2.FlannBasedMatcher(index_params,search_params)

    def makeGlobeDescriptor(self, fileName, numDescriptors):
        # Check if file is JPG or PNG
        if re.search(r"\.([pP][nN][gG]|[jJ][pP][gG])$", fileName):
            # Check if file exists
            if os.path.isfile(fileName):
                # Load file, transform into HSV color space and extract hue
                globeImg = cv2.imread(fileName)
                globeImg = cv2.GaussianBlur(globeImg, (5,5), 0)
                #globeImgHue = cv2.cvtColor(globeImg, cv2.COLOR_BGR2GRAY)
                globeImgHue = cv2.cvtColor(globeImg, cv2.COLOR_BGR2HSV)[:, :, 0]
                # Create detector and compute descriptors and keypoints
                orb = cv2.ORB_create(int(numDescriptors))
                keypoints, descriptors = orb.detectAndCompute(globeImgHue, None)
                # Calculate 3D coordinates for keypoints
                keypoints3D = []
                imgShape = globeImgHue.shape
                for kp in keypoints:
                    p3D = self.millerProjection(kp.pt, imgShape)
                    keypoints3D.append(p3D)
                return descriptors, keypoints3D
        return None, None

    def millerProjection(self, pt2D, imgShape):
        # Calculate miller Projection and return 3D coordinate corresponding to 2D point
        x2D = pt2D[0] / imgShape[1] * 2 * np.pi
        y2D = (2 * pt2D[1] / imgShape[0] - 1) * 2.03
        lambdaVal = x2D - np.pi
        phiVal = - 5/4 * np.arctan(np.sinh(4/5 * y2D))
        z = np.sin(phiVal) * 17.5
        p = np.cos(phiVal) * 17.5
        y = np.sin(lambdaVal) * p
        x = np.cos(lambdaVal) * p
        return [x, y, z]

    def calculateMatches(self, runtimeDescriptors):
        # Calculate matches
        knnMatches = self.flannMatcher.knnMatch(self.globeDescriptors, runtimeDescriptors, k=2)
        # Filter matches using the lowe's ratio test
        ratio_thresh = 0.7
        matches = []
        if len(knnMatches) > 0:
            try:
                for m,n in knnMatches:
                    if m.distance < ratio_thresh * n.distance:
                        matches.append(m)
            except ValueError:
                pass
        if len(matches) > 0:
            return sorted(matches, key = lambda x:x.distance)[:100]
        return matches

    def detectGlobe(self, hueMasked, rvec=None, tvec=None):
        # Detect and compute ORB for incoming frame
        test = self.runtimeOrb.detectAndCompute(hueMasked, None)
        runtimeKeypoints, runtimeDescriptors = self.runtimeOrb.detectAndCompute(hueMasked, None)
        # Calculate Matches
        if runtimeDescriptors is not None:
            matches = self.calculateMatches(runtimeDescriptors)
            if len(matches) > 30:
                # Create arrays for the matching 3D-Coordinates and 2D-Points on frame
                matchPoints = np.zeros((len(matches), 2), dtype=np.float32)
                matchPoints3D = np.zeros((len(matches), 3), dtype=np.float32)
                for i, match in enumerate(matches):
                    matchPoints[i, :] = runtimeKeypoints[match.trainIdx].pt
                    matchPoints3D[i,:] = self.keypoints3D[match.queryIdx]

                ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(matchPoints3D, matchPoints, self.mtx, self.dist, rvec, tvec)
                if ret:
                    goodMatchPoints = np.zeros((inliers.shape[0], 2), dtype=np.float32)
                    goodMatchPoints3D = np.zeros((inliers.shape[0], 3), dtype=np.float32)
                    for i, index in enumerate(inliers):
                        goodMatchPoints[i,:] = matchPoints[index]
                        goodMatchPoints3D[i,:] = matchPoints3D[index]
                    return (rvecs, tvecs, goodMatchPoints, goodMatchPoints3D)
        return None

    def trackGlobe(self, oldFrame, newFrame, features, features3D, rvecs, tvecs):
        featuresReshaped = features.reshape(-1,1,2)
        features3DReshaped = features3D.reshape(-1,1,3)
        newPoints, status, _ = cv2.calcOpticalFlowPyrLK(oldFrame, newFrame, featuresReshaped, None, **self.lkParams)
        if newPoints is not None:
            if newPoints.shape[0] > 5:
                goodFeatures = newPoints[status == 1]
                goodFeatures3D = features3DReshaped[status == 1]
                ret, rvecs, tvecs, _ = cv2.solvePnPRansac(features3D, features, self.mtx, self.dist, rvecs, tvecs)
                if ret:
                    return (rvecs, tvecs, goodFeatures, goodFeatures3D)
        return None