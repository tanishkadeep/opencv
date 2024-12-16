import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('chessboard/*.jpg')

for fname in images:
    img = cv.imread(fname)
    
    if img is None:
        print(f"Error loading image: {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    if ret:
        objpoints.append(objp)
        
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        imgpoints.append(corners2)
        
        cv.drawChessboardCorners(img, (7, 6), corners2, ret)
        
        cv.imshow('Chessboard Corners', img)
        cv.waitKey(0)

if len(objpoints) > 0 and len(imgpoints) > 0:
    h, w = img.shape[:2]
    ret, matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        (w, h),
        None,
        None
    )

    if ret:
        print("Camera matrix:")
        print(matrix)
        print("Distortion coefficients:")
        print(distortion)
        print("Rotation vectors:")
        print(r_vecs)
        print("Translation vectors:")
        print(t_vecs)
    else:
        print("Calibration failed.")
else:
    print("Insufficient data for calibration.")

cv.destroyAllWindows()
