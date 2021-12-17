import cv2
import numpy as np
import glob

################################## Intialization #################################

boardsize = (7,10)      #board size excluding outermost edge: 7x10
framesize = (4096,2160) #size of the images

# Termination Criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Object,Image Coordinates
objpoints, imgpoints = [],[]

# Defining World Coordinates
objp = np.zeros((boardsize[0]*boardsize[1],3),np.float32)
objp[:,:2] = np.mgrid[0:boardsize[0], 0:boardsize[1]].T.reshape(-1,2)


##################### Find corners #############################

# Getting all the images from directory
images = glob.glob('sports science project/*.jpeg')

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray,boardsize,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, boardsize, corners2, ret)
        cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

##################################### Calibration ####################################

ret, intrinsicmatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, framesize, None, None)

print("camera calibrated: ", ret)
print("\nIntrinsic Matrix: \n", intrinsicmatrix)
print(("\nDistortion parameters: \n",dist))
print("\nRotational Vectors: \n", rvecs)
print("\nTranslational Vectors: \n", tvecs)
