import cv2
import numpy as np
import glob
import os

checkerboard_size = (10, 7)
square_size = 0.024
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_dir = "checkerboard"
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
# Shape: (N, 3) where N = checkerboard_size[0] * checkerboard_size[1]
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : checkerboard_size[0], 0 : checkerboard_size[1]].T.reshape(
    -1, 2
)
objp *= square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(os.path.join(image_dir, "*.jpg"))
images += glob.glob(os.path.join(image_dir, "*.png"))

gray = None
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        # Refine corners
        # corners2 shape: (N, 1, 2)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

if len(objpoints) == 0:
    print("No checkerboard corners found in the images.")
    # raise ValueError("No checkerboard corners found in the images.")

# Calibrate camera
# mtx shape: (3, 3)
# dist shape: (1, 5)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Ensure that the "intrinsics" directory exists
os.makedirs("intrinsics", exist_ok=True)
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)
# Save the camera matrix and distortion coefficients as .npy files
np.save("intrinsics/intrinsics.npy", mtx)
np.save("intrinsics/distortions.npy", dist)

print("\nSaved to intrinsics/intrinsics.npy and intrinsics/distortions.npy")